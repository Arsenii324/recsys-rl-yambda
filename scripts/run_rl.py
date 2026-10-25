#!/usr/bin/env python3
"""One-step REINFORCE fine-tuning against a frozen conditional-completion proxy."""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from recsys_tt2.batching import batches, make_batch, open_event_arrays
from recsys_tt2.model import ConditionalCompletionRM, ItemGPT
from run_experiments import device_from, eval_policy, seed_all


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def data_provenance(data_root: Path, train_indices: np.ndarray, valid_indices: np.ndarray) -> dict:
    manifest_path = data_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    return {
        'data_runtime_path': str(data_root.resolve()),
        'data_source': manifest.get('source'),
        'data_revision': manifest.get('revision'),
        'data_source_sha256': manifest.get('source_sha256'),
        'data_manifest_sha256': sha256(manifest_path),
        'train_indices_file_sha256': sha256(data_root / 'train_indices.npy'),
        'train_anchor_count_used': int(len(train_indices)),
        'train_anchor_indices_used_sha256': hashlib.sha256(np.asarray(train_indices, dtype=np.uint32).tobytes()).hexdigest(),
        'validation_event_count_used': int(len(valid_indices)),
        'validation_event_indices_used_sha256': hashlib.sha256(np.asarray(valid_indices, dtype=np.uint32).tobytes()).hexdigest(),
    }


def load_policy(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    c = payload['config']
    model = ItemGPT(n_items=int(c['n_items']), width=int(c['width']), layers=int(c['layers']),
                    heads=int(c['heads']), max_context=int(c['max_context']), dropout=float(c['dropout'])).to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, c, payload


def make_item_durations(indices, arrays, n_items):
    tokens = arrays['item_token'][indices].astype(np.int64)
    duration = arrays['track_length_seconds'][indices].astype(np.float64)
    keep = tokens > 0
    classes = tokens[keep] - 1
    sums = np.bincount(classes, weights=duration[keep], minlength=n_items)
    counts = np.bincount(classes, minlength=n_items)
    return np.divide(sums, counts, out=np.full(n_items, 180.0), where=counts > 0).astype(np.float32)


def eval_proxy(policy, reference, rm, indices, arrays, item_duration, *, batch_size, max_context, temperature, device):
    policy.eval(); reference.eval(); rm.eval()
    kl_sum = entropy_sum = reward_sum = duration_sum = rank_sum = repeat_sum = supported_sum = 0.0
    agree = {k: 0.0 for k in (10, 50, 100)}
    n = 0
    with torch.inference_mode():
        for id_batch in batches(indices, batch_size):
            b = make_batch(id_batch, arrays, n_items=policy.n_items, max_context=max_context, device=device)
            logits = policy(b['tokens'], b['padding_mask']) / temperature
            ref_logits = reference(b['tokens'], b['padding_mask']) / temperature
            logp = torch.log_softmax(logits, -1)
            ref_logp = torch.log_softmax(ref_logits, -1)
            p = logp.exp()
            kl = (p * (logp - ref_logp)).sum(-1)
            entropy = -(p * logp).sum(-1)
            top = torch.topk(logits, k=100, dim=-1).indices
            ref_top = torch.topk(ref_logits, k=100, dim=-1).indices
            top1 = top[:, 0]
            top1_rank = torch.as_tensor(arrays['candidate_popularity_rank'][top1.cpu().numpy()].astype(np.int64), device=device)
            reward = rm(b['rm_history'], top1, torch.as_tensor(item_duration, device=device)[top1], top1_rank)
            kl_sum += float(kl.sum()); entropy_sum += float(entropy.sum())
            for k in agree:
                overlap = (top[:, :k, None] == ref_top[:, None, :k]).any(dim=-1).sum(dim=-1)
                agree[k] += float((overlap / k).sum())
            duration_sum += float(torch.as_tensor(item_duration, device=device)[top1].sum())
            rank_sum += float((top1_rank + 1).sum())
            repeat = b['supported'] & b['tokens'].eq(top1[:, None] + 1).any(dim=1)
            repeat_sum += float(repeat.sum()); supported_sum += float(b['supported'].sum())
            reward_sum += float(reward.sum()); n += len(id_batch)
    return {'queries': n, 'mean_kl_policy_to_base': kl_sum / max(1,n), 'mean_entropy_nats': entropy_sum / max(1,n),
            'topk_agreement_with_base': {str(k): agree[k] / max(1,n) for k in agree},
            'top1_predicted_completion_proxy': reward_sum / max(1,n),
            'top1_mean_duration_seconds': duration_sum / max(1,n),
            'top1_mean_catalog_rank': rank_sum / max(1,n), 'top1_repeat_fraction_all': repeat_sum / max(1,n),
            'top1_repeat_fraction_supported': repeat_sum / max(1,supported_sum)}


def train_supervised_control(args, config, model_config, arrays, train_indices, valid_indices,
                             reference, rm, item_duration, device):
    """Matched-step/data-exposure continuation control from the same frozen base."""
    seed_all(int(config['seed']))
    model, _, _ = load_policy(args.base_checkpoint, device)
    model.eval()  # keep the same disabled-dropout convention as the sampled policy arms
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['supervised_learning_rate']), weight_decay=0.0)
    run_dir = Path(args.output) / 'supervised-data-matched'; run_dir.mkdir(parents=True, exist_ok=True)
    permutation = np.random.default_rng(int(config['seed'])).permutation(train_indices)
    cursor = 0; started = time.time(); best_score = -math.inf; best_step = None
    with (run_dir / 'metrics.jsonl').open('w') as log:
        for step in range(1, int(config['steps']) + 1):
            bs = int(config['batch_size'])
            if cursor + bs > len(permutation):
                permutation = np.random.default_rng(int(config['seed']) + step).permutation(train_indices); cursor = 0
            ids = permutation[cursor:cursor + bs]; cursor += bs
            b = make_batch(ids, arrays, n_items=model.n_items, max_context=model.max_context, device=device)
            logits = model(b['tokens'], b['padding_mask'])
            loss = F.cross_entropy(logits[b['supported']], b['target'][b['supported']])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step == 1 or step % int(config['log_every']) == 0:
                rec = {'step':step,'arm':'supervised-data-matched','train_ce':float(loss.detach()),'elapsed_seconds':time.time()-started}
                log.write(json.dumps(rec)+'\n'); log.flush(); print(json.dumps(rec),flush=True)
            if step % int(config['eval_every']) == 0 or step == int(config['steps']):
                metrics = eval_policy(model,valid_indices,arrays,batch_size=bs,device=device)
                diagnostics = eval_proxy(model,reference,rm,valid_indices,arrays,item_duration,batch_size=bs,
                                         max_context=model.max_context,temperature=float(config['temperature']),device=device)
                rec={'step':step,'arm':'supervised-data-matched','validation':metrics,'policy_diagnostics':diagnostics,'elapsed_seconds':time.time()-started}
                log.write(json.dumps(rec)+'\n'); log.flush()
                score=metrics['all']['ndcg@100']
                if score>best_score:
                    best_score,best_step=score,step
                    torch.save({'state_dict':model.state_dict(),'config':config,'model_config':model_config,'step':step,
                                'selection_metric':'validation all-event NDCG@100','selection_score':score},run_dir/'supervised-best.pt')
                if step==int(config['steps']):
                    torch.save({'state_dict':model.state_dict(),'config':config,'model_config':model_config,'step':step},run_dir/'supervised-final.pt')
                print(json.dumps(rec),flush=True)
    return {'arm':'supervised-data-matched','best_step':best_step,'best_validation_ndcg100':best_score,
            'elapsed_seconds':time.time()-started,'training_examples_seen':int(config['steps'])*int(config['batch_size']),
            'learning_rate':float(config['supervised_learning_rate'])}


def train_beta(beta, args, config, model_config, arrays, train_indices, valid_indices, reference, rm, item_duration, device):
    seed_all(int(config['seed']))
    policy, _, _ = load_policy(args.base_checkpoint, device)
    # Deterministic policy probabilities are used for both sampled log-probability and KL.
    policy.eval(); reference.eval(); rm.eval()
    for p in reference.parameters(): p.requires_grad_(False)
    for p in rm.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
    run_dir = Path(args.output) / f'beta-{beta:g}'
    run_dir.mkdir(parents=True, exist_ok=True)
    best_score = -math.inf; best_step = None
    permutation = np.random.default_rng(int(config['seed'])).permutation(train_indices)
    cursor = 0; started = time.time()
    with (run_dir / 'metrics.jsonl').open('w') as log:
        for step in range(1, int(config['steps']) + 1):
            bs = int(config['batch_size'])
            if cursor + bs > len(permutation):
                permutation = np.random.default_rng(int(config['seed']) + step).permutation(train_indices); cursor = 0
            ids = permutation[cursor:cursor + bs]; cursor += bs
            b = make_batch(ids, arrays, n_items=policy.n_items, max_context=policy.max_context, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = policy(b['tokens'], b['padding_mask']) / float(config['temperature'])
            with torch.no_grad():
                ref_logits = reference(b['tokens'], b['padding_mask']) / float(config['temperature'])
            logp = torch.log_softmax(logits, dim=-1)
            ref_logp = torch.log_softmax(ref_logits, dim=-1)
            dist = Categorical(logits=logits)
            action = dist.sample()
            action_logp = dist.log_prob(action)
            action_duration = torch.as_tensor(item_duration, device=device)[action]
            action_rank = torch.as_tensor(arrays['candidate_popularity_rank'][action.detach().cpu().numpy()].astype(np.int64), device=device)
            with torch.no_grad():
                sampled_reward = rm(b['rm_history'], action, action_duration, action_rank)
            # A fixed state-independent baseline reduces variance without rescaling reward.
            advantage = sampled_reward - float(config['reward_baseline'])
            kl_per_query = (logp.exp() * (logp - ref_logp)).sum(dim=-1)
            reinforce = -(advantage.detach() * action_logp).mean()
            kl_penalty = kl_per_query.mean()
            loss = reinforce + float(beta) * kl_penalty
            if not torch.isfinite(loss): raise FloatingPointError(f'non-finite RL loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % int(config['log_every']) == 0:
                record = {'step': step, 'beta': beta, 'loss': float(loss.detach()), 'reinforce_loss': float(reinforce.detach()),
                          'kl_penalty': float(kl_penalty.detach()), 'mean_sampled_proxy_reward': float(sampled_reward.mean()),
                          'mean_entropy_nats': float(dist.entropy().mean().detach()), 'elapsed_seconds': time.time()-started}
                log.write(json.dumps(record)+'\n'); log.flush(); print(json.dumps(record), flush=True)
            if step % int(config['eval_every']) == 0 or step == int(config['steps']):
                policy.eval()
                val_metrics = eval_policy(policy, valid_indices, arrays, batch_size=bs, device=device)
                proxy_metrics = eval_proxy(policy, reference, rm, valid_indices, arrays, item_duration,
                                           batch_size=bs, max_context=policy.max_context,
                                           temperature=float(config['temperature']), device=device)
                record = {'step': step, 'beta': beta, 'validation': val_metrics, 'policy_diagnostics': proxy_metrics,
                          'elapsed_seconds': time.time()-started}
                log.write(json.dumps(record)+'\n'); log.flush()
                score = val_metrics['all']['ndcg@100']
                if score > best_score:
                    best_score, best_step = score, step
                    torch.save({'state_dict':policy.state_dict(), 'config':config, 'model_config':model_config, 'step':step, 'beta':beta,
                                'selection_metric':'validation all-event NDCG@100','selection_score':score}, run_dir/'rl-best.pt')
                if step == int(config['steps']):
                    torch.save({'state_dict':policy.state_dict(), 'config':config, 'model_config':model_config, 'step':step,'beta':beta}, run_dir/'rl-final.pt')
                print(json.dumps(record), flush=True)
                policy.train(False)
    return {'arm':'reinforce','beta':beta,'best_step':best_step,'best_validation_ndcg100':best_score,
            'elapsed_seconds':time.time()-started,'training_examples_seen':int(config['steps'])*int(config['batch_size']),
            'learning_rate':float(config['learning_rate']),
            'base_checkpoint_sha256':sha256(args.base_checkpoint),'rm_checkpoint_sha256':sha256(args.rm_checkpoint)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True); p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--rm-checkpoint',type=Path,required=True); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='auto')
    args=p.parse_args(); config=json.loads(args.config.read_text()); device=device_from(args.device)
    arrays=open_event_arrays(args.data)
    train_indices=np.load(args.data/'train_indices.npy',mmap_mode='r')
    valid_all=np.load(args.data/'valid_indices.npy',mmap_mode='r')
    eval_count=int(config.get('eval_event_count',0))
    valid_indices=(np.random.default_rng(991).choice(valid_all,size=eval_count,replace=False) if eval_count and len(valid_all)>eval_count else valid_all)
    reference, base_config, base_payload=load_policy(args.base_checkpoint,device)
    if int(config['n_items']) != reference.n_items: raise ValueError('RL config/base candidate size mismatch')
    rm_payload=torch.load(args.rm_checkpoint,map_location=device,weights_only=False); rmc=rm_payload['config']
    rm=ConditionalCompletionRM(int(config['n_items']),width=int(rmc['width']),variant='combined').to(device)
    rm.load_state_dict(rm_payload['state_dict']); rm.eval()
    item_duration=make_item_durations(train_indices,arrays,reference.n_items)
    results=[train_supervised_control(args,config,base_config,arrays,train_indices,valid_indices,reference,rm,item_duration,device)]
    for beta in config['kl_coefficients']:
        results.append(train_beta(float(beta),args,config,base_config,arrays,train_indices,valid_indices,reference,rm,item_duration,device))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'run.json').write_text(json.dumps({'mode':'one-step-reinforce','config':config,'results':results,
        **data_provenance(args.data,train_indices,valid_indices),
        'base_checkpoint_sha256':sha256(args.base_checkpoint),'rm_checkpoint_sha256':sha256(args.rm_checkpoint),
        'base_step':base_payload.get('step'),'base_config':base_config,'device':str(device),'torch':torch.__version__},indent=2)+'\n')

if __name__=='__main__': main()
