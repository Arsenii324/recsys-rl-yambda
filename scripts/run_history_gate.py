#!/usr/bin/env python3
"""Train a one-pass gate over frozen GPT and recent-item repeat distributions."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from recsys_tt2.batching import batches, make_batch, open_event_arrays
from recsys_tt2.history_gate import (HistoryGate, frozen_last_state, history_repeat_stats,
                                     mixture_target_logprob, mix_distributions)
from recsys_tt2.model import ItemGPT

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def seed_all(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_base(path: Path, device: torch.device):
    payload=torch.load(path,map_location=device,weights_only=False)
    c=payload.get("model_config",payload["config"])
    base=ItemGPT(n_items=int(c["n_items"]),width=int(c["width"]),layers=int(c["layers"]),
                 heads=int(c["heads"]),max_context=int(c["max_context"]),dropout=float(c["dropout"])).to(device)
    base.load_state_dict(payload["state_dict"]); base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    return base,payload,c

def _rank(probs,target,supported):
    n=probs.shape[1]; top=torch.topk(probs,k=min(100,n),dim=-1).indices
    hitpos=top.eq(target[:,None]) & supported[:,None]
    found=hitpos.any(dim=1); rank=hitpos.to(torch.int64).argmax(dim=1)+1
    return torch.where(found,rank,n+1)

def _metric(rank,mask,supported):
    support=mask & supported
    out={"queries":int(mask.sum()),"supported_queries":int(support.sum()),
         "support_fraction":float(support.sum()/max(1,mask.sum()))}
    for name,denom in (("all_target_denominator",mask),("supported_target_denominator",support)):
        for k in (10,50,100):
            hit=rank[denom] <= k
            out[f"{name}_recall@{k}"]=float(hit.mean()) if hit.size else None
            out[f"{name}_ndcg@{k}"]=float(np.mean(np.where(hit,1.0/np.log2(rank[denom]+1.0),0.0))) if hit.size else None
    return out

def evaluate(model,gate,indices,arrays,*,n_items,batch_size,device,collect):
    names=("gpt","history_50","history_gate","history_only")
    rank_parts={n:[] for n in names}
    col_parts={n:[] for n in ("uid","event_index","organic","supported","seen_in_context","gate")}
    model.eval(); gate.eval()
    with torch.inference_mode():
        for idx in batches(indices,batch_size):
            b=make_batch(idx,arrays,n_items=n_items,max_context=model.max_context,device=device)
            state=frozen_last_state(model,b["tokens"],b["padding_mask"])
            logits=F.linear(state,model.item_embedding.weight[1:n_items+1],model.output_bias)
            gpt=torch.softmax(logits.float(),dim=-1)
            stats=history_repeat_stats(b["tokens"],n_items); gamma=gate(state,stats)
            distributions={"gpt":gpt,
                "history_50":mix_distributions(gpt,b["tokens"],n_items,0.5),
                "history_gate":mix_distributions(gpt,b["tokens"],n_items,gamma),
                "history_only":mix_distributions(gpt,b["tokens"],n_items,1.0)}
            for name,p in distributions.items():
                rank_parts[name].append(_rank(p,b["target"],b["supported"]).cpu().numpy().astype(np.int32))
            if collect:
                seen=b["supported"] & b["tokens"].eq(b["target"][:,None]+1).any(dim=1)
                event=torch.as_tensor(idx,dtype=torch.long,device=device)
                for name,val in (("uid",b["uid"]),("event_index",event),("organic",b["organic"]),
                                 ("supported",b["supported"]),("seen_in_context",seen),("gate",gamma)):
                    col_parts[name].append(val.cpu().numpy())
    ranks={n:np.concatenate(v) for n,v in rank_parts.items()}
    columns={n:np.concatenate(v) for n,v in col_parts.items()} if collect else {}
    organic=columns["organic"].astype(bool) if collect else np.asarray(arrays["is_organic"][indices],dtype=bool)
    supported=columns["supported"].astype(bool) if collect else np.asarray(arrays["item_token"][indices]>0)
    masks={"all":np.ones(len(indices),dtype=bool),"organic":organic,"recommended":~organic}
    report={g:{n:_metric(ranks[n],m,supported) for n in names} for g,m in masks.items()}
    if collect:
        seen=columns["seen_in_context"].astype(bool)
        report["seen_supported"]={n:_metric(ranks[n],seen,supported) for n in names}
        report["unseen_supported"]={n:_metric(ranks[n],supported & ~seen,supported) for n in names}
        columns.update({f"rank_{n}":v for n,v in ranks.items()})
    return report,columns

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True); p.add_argument("--base-checkpoint",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="auto"); args=p.parse_args()
    cfg=json.loads(args.config.read_text()); seed_all(int(cfg["seed"]))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else torch.device(args.device)
    arrays=open_event_arrays(args.data)
    train=np.load(args.data/"train_indices.npy",mmap_mode="r")
    valid_all=np.load(args.data/"valid_indices.npy",mmap_mode="r")
    if not np.all(arrays["split"][train]==0) or np.any(arrays["item_token"][train]<=0):
        raise ValueError("training indices must be training split with supported targets")
    valid=np.asarray(valid_all)
    if cfg["eval_event_count"] and len(valid)>int(cfg["eval_event_count"]):
        valid=np.random.default_rng(int(cfg["selection_seed"])).choice(valid,size=int(cfg["eval_event_count"]),replace=False)
    if not np.all(arrays["split"][valid]==2): raise ValueError("validation selection contains non-validation rows")
    base,base_payload,base_cfg=load_base(args.base_checkpoint,device)
    if base.n_items != int(cfg["n_items"]): raise ValueError("gate/base candidate size mismatch")
    gate=HistoryGate(base.width,hidden=int(cfg["hidden_width"])).to(device)
    opt=torch.optim.AdamW(gate.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]))
    order=np.random.default_rng(int(cfg["seed"])).permutation(train)
    nsteps=math.ceil(len(train)/int(cfg["batch_size"]))
    if cfg.get("single_pass",True) and int(cfg["steps"])!=nsteps:
        raise ValueError(f"registered steps {cfg['steps']} do not equal one-pass steps {nsteps}")
    args.output.mkdir(parents=True,exist_ok=True); start=time.time(); cursor=0
    best=-math.inf; best_step=None
    with (args.output/"metrics.jsonl").open("w") as log:
        for step in range(1,int(cfg["steps"])+1):
            idx=order[cursor:cursor+int(cfg["batch_size"])]
            if not len(idx): raise RuntimeError("one-pass anchors exhausted")
            cursor+=len(idx)
            b=make_batch(idx,arrays,n_items=base.n_items,max_context=base.max_context,device=device)
            if not bool(b["supported"].all()): raise ValueError("unsupported train anchor")
            with torch.no_grad():
                state=frozen_last_state(base,b["tokens"],b["padding_mask"])
                logits=F.linear(state,base.item_embedding.weight[1:base.n_items+1],base.output_bias)
                stats=history_repeat_stats(b["tokens"],base.n_items)
            gamma=gate(state.detach(),stats)
            loss=-mixture_target_logprob(logits,b["target"],b["tokens"],gamma,base.n_items).mean()
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"nonfinite gate CE step {step}")
            opt.zero_grad(set_to_none=True); loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(gate.parameters(),1.0)
            if not bool(torch.isfinite(grad)) or float(grad)<=0: raise FloatingPointError(f"invalid gate gradient step {step}")
            if any(p.grad is not None for p in base.parameters()): raise AssertionError("frozen base received gradients")
            opt.step()
            if step==1 or step%int(cfg["log_every"])==0:
                rec={"step":step,"train_ce":float(loss.detach()),"mean_history_weight":float(gamma.detach().mean()),
                     "gradient_norm":float(grad),"examples_seen":int(cursor),"elapsed_seconds":time.time()-start}
                log.write(json.dumps(rec)+"\n"); log.flush(); print(json.dumps(rec),flush=True)
            if step%int(cfg["eval_every"])==0 or step==int(cfg["steps"]):
                metrics,_=evaluate(base,gate,valid,arrays,n_items=base.n_items,batch_size=int(cfg["eval_batch_size"]),device=device,collect=False)
                score=metrics["all"]["history_gate"]["all_target_denominator_ndcg@100"]
                rec={"step":step,"validation":metrics,"elapsed_seconds":time.time()-start}
                log.write(json.dumps(rec)+"\n"); log.flush(); print(json.dumps(rec),flush=True)
                if score>best:
                    best,best_step=score,step
                    torch.save({"state_dict":gate.state_dict(),"config":cfg,"step":step,
                                "selection_metric":cfg["select_by"],"selection_score":score,
                                "base_checkpoint_sha256":sha256(args.base_checkpoint)},args.output/"gate-best.pt")
            if step==int(cfg["steps"]):
                torch.save({"state_dict":gate.state_dict(),"config":cfg,"step":step,
                            "base_checkpoint_sha256":sha256(args.base_checkpoint)},args.output/"gate-final.pt")
    if cfg.get("single_pass", True) and cursor!=len(order):
        raise AssertionError(f"consumed {cursor}/{len(order)} anchors")
    selected=torch.load(args.output/"gate-best.pt",map_location=device,weights_only=False)
    gate.load_state_dict(selected["state_dict"]); gate.eval()
    report,per_query=evaluate(base,gate,valid,arrays,n_items=base.n_items,batch_size=int(cfg["eval_batch_size"]),device=device,collect=True)
    np.savez_compressed(args.output/"per-query-validation.npz",**per_query)
    data_manifest=json.loads((args.data/"manifest.json").read_text())
    run={"method":"one-pass learned history gate over frozen GPT and recent-history frequency","config":cfg,
         "train_anchor_count":int(len(train)),"examples_seen":int(cursor),"steps":int(cfg["steps"]),
         "final_batch_size":int(min(int(cfg["batch_size"]),len(order)-int(cfg["batch_size"])*(int(cfg["steps"])-1))),
         "train_anchor_permutation_sha256":hashlib.sha256(np.asarray(order,dtype=np.uint32).tobytes()).hexdigest(),
         "validation_event_indices_sha256":hashlib.sha256(np.asarray(valid,dtype=np.uint32).tobytes()).hexdigest(),
         "data_manifest_sha256":sha256(args.data/"manifest.json"),"data_source":data_manifest.get("source"),
         "base_checkpoint":str(args.base_checkpoint),"base_checkpoint_sha256":sha256(args.base_checkpoint),
         "base_training_step":base_payload.get("step"),"base_model_config":base_cfg,
         "best_step_by_validation_ndcg100":best_step,"best_validation_ndcg100":best,"selected_validation":report,
         "device":str(device),"torch_version":torch.__version__,
         "scope":"observational next-item retrieval; gate is not a response model or causal policy",
         "history_source":"frozen train-supported catalog; last 64 strict-earlier context positions; unknown/BOS/PAD skipped",
         "output_arrays":"per-query uid, event index, support/organic/seen flags, gate, and ranks for GPT/constant mixture/learned gate/history-only"}
    (args.output/"run.json").write_text(json.dumps(run,indent=2)+"\n")

if __name__=="__main__": main()
