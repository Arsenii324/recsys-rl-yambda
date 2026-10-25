#!/usr/bin/env python3
"""Run one paired recurrent item-backbone comparison on Kaggle T4x2.

Inputs are staged and hash-verified before CUDA initialization. Both arms use
identical full-supported data and shared fresh initial weights for each seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_tt2.batching import make_batch, open_event_arrays  # noqa: E402
from recsys_tt2.huginn_item_backbone import HuginnItemGPT  # noqa: E402
from scripts.run_experiments import eval_policy, seed_all  # noqa: E402
from recsys_tt2.live_monitor import LiveMonitor  # noqa: E402

FEATURE_MANIFEST_SHA = "ecc0aa56ac64c3d2a0351f4e02c60977035855a41b83ed8d88ad947c00d949b6"
FULL_SOURCE_MANIFEST_SHA = "aa5570c4e16cbd61e33457bd31970b356c8a49b9938de4ba90c773109bb85570"
SUBSET_INDICES_SHA = "90a38ccfc2098bd78299a92c62b6cfffdceab5b354283fe9f095d3bf81fbae3c"
FULL_INDICES_SHA = "948a6d33e6a5055c868d65c59b2b81ce60ff856111c68d7659f57008f595c474"
VALIDATION_EVENT_SHA = "950a0d20892fec4b383091752af797bee9d4096a49be97577edc25ed0c7f0a12"
TEST_INDICES_SHA = "40dec33232ce369bb3e0b9f67dcafc1afd298899543de065b216501944297083"
INDEX_INPUT_MANIFEST_SHA = "0e8efb8408ceb9714160b2dacce0a34779c33d0e320fb63176f6ed7514871a61"
WORKER_CPU_THREADS = 1
ARRAY_FILES = ("uid", "timestamp", "item_token", "played_ratio_pct", "track_length_seconds",
               "is_organic", "split", "context_end", "history_start", "tie_group_size",
               "candidate_item_ids")


def copy_verified(source: Path, target: Path, expected_sha: str, expected_bytes: int) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    started = time.monotonic()
    with source.open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            digest.update(block)
            dst.write(block)
            size += len(block)
        dst.flush()
        os.fsync(dst.fileno())
    actual = digest.hexdigest()
    if size != expected_bytes or actual != expected_sha:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"staged input mismatch for {source.name}: bytes={size}, sha256={actual}")
    result = {"source": str(source), "staged": str(target), "bytes": size,
              "sha256": actual, "copy_seconds": round(time.monotonic() - started, 3)}
    print(json.dumps({"event": "input_file_staged", "name": source.name,
                      "bytes": size, "copy_seconds": result["copy_seconds"],
                      "sha256": actual}), flush=True)
    return result


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def common_init_sha256(model: torch.nn.Module) -> str:
    """Hash parameters shared across modes; omit cloned untied core copies."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("core.") and name.split(".", 2)[1] != "0":
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_structure(model: HuginnItemGPT) -> dict:
    params = list(model.parameters())
    core_block_ids = []
    seen: dict[int, int] = {}
    for step in range(model.default_depth):
        block = model.core[step] if model.untied else model.core[0]
        key = id(block)
        if key not in seen:
            seen[key] = len(seen)
        core_block_ids.append(seen[key])
    return {"trainable_parameter_count": sum(p.numel() for p in params),
            "core_stored_block_count": len(model.core),
            "core_pass_block_sharing_ids": core_block_ids,
            "output_logits_shape_contract": "[batch,100000] from final sequence position only"}


def cpu_model_preflight(feature_root: Path, train_indices_path: Path,
                        config: dict, seed: int) -> dict:
    """Exercise actual feature batches, all planned modes, masks, CE and gradients."""
    torch.set_num_threads(1)
    arrays = open_event_arrays(feature_root)
    indices = np.load(train_indices_path, mmap_mode="r")
    batch_ids = np.asarray(indices[:4], dtype=np.uint32)
    batch = make_batch(batch_ids, arrays, n_items=int(config["architecture"]["n_items"]),
                       max_context=int(config["architecture"]["max_context"]),
                       device=torch.device("cpu"))
    if int(batch["supported"].sum()) != len(batch_ids):
        raise RuntimeError("CPU preflight selected a non-supported training target")

    models: dict[str, HuginnItemGPT] = {}
    records = []
    common_hashes = set()
    for arm in config["active_arms"]:
        seed_all(seed)
        model = HuginnItemGPT(**config["architecture"], mode=arm, seed=seed)
        models[arm] = model
        common_hashes.add(common_init_sha256(model))
        record = {"arm": arm, **model_structure(model),
                  "initial_state_sha256": state_dict_sha256(model),
                  "common_initial_state_sha256": common_init_sha256(model)}
        model.eval()
        torch.manual_seed(1031)
        logits = model(batch["tokens"], batch["padding_mask"])
        if tuple(logits.shape) != (len(batch_ids), int(config["architecture"]["n_items"])):
            raise RuntimeError(f"unexpected catalog output shape for {arm}: {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"non-finite initial logits for {arm}")
        loss = F.cross_entropy(logits.float(), batch["target"])
        loss.backward()
        grad_sq = torch.zeros((), dtype=torch.float64)
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite gradient for {arm}")
                grad_sq += parameter.grad.detach().double().square().sum()
        record.update({"initial_train_ce": float(loss.detach()),
                       "gradient_norm": float(grad_sq.sqrt()),
                       "finite_logits_and_gradients": True})
        records.append(record)
        model.zero_grad(set_to_none=True)
    if len(common_hashes) != 1:
        raise RuntimeError("shared initial tensors differ across the four arms")
    if models["R1source"].default_depth != 1:
        raise RuntimeError("R1source is not a one-pass model")
    if len(models["R4source"].core) != 1 or len(models["R4self"].core) != 1:
        raise RuntimeError("shared R4 controls do not store exactly one recurrent block")
    if len(models["R4untiedsource"].core) != 4:
        raise RuntimeError("untied R4 control does not store four recurrent blocks")
    for block in models["R4untiedsource"].core[1:]:
        for left, right in zip(models["R4untiedsource"].core[0].state_dict().values(),
                               block.state_dict().values()):
            if not torch.equal(left, right):
                raise RuntimeError("untied recurrent blocks do not start as exact clones")

    # R1source and R4self at recurrence depth one execute the identical first pass.
    first = models["R1source"].eval()
    self_control = models["R4self"].eval()
    with torch.no_grad():
        torch.manual_seed(9921)
        r1_logits = first(batch["tokens"], batch["padding_mask"])
        torch.manual_seed(9921)
        self_logits = self_control(batch["tokens"], batch["padding_mask"], recurrence_depth=1)
    parity_error = float((r1_logits - self_logits).abs().max())
    if parity_error > 1e-6:
        raise RuntimeError(f"R1source/R4self first-pass parity failed: max_abs={parity_error}")

    # Verify production key-padding behavior with actual left-padded feature batches.
    pad_model = models["R4source"].eval()
    padding_mask = batch["padding_mask"]
    changed = batch["tokens"].clone()
    changed[padding_mask] = 1
    with torch.no_grad():
        torch.manual_seed(553)
        pad_a = pad_model(batch["tokens"], padding_mask)
        torch.manual_seed(553)
        pad_b = pad_model(changed, padding_mask)
    padding_error = float((pad_a - pad_b).abs().max())
    if padding_error > 1e-6:
        raise RuntimeError(f"padding-mask parity failed: max_abs={padding_error}")

    # Inspect the actual attention module's causal mask by changing future states.
    attention = pad_model.prelude.attention.eval()
    x = torch.randn(1, 4, int(config["architecture"]["width"]))
    changed_x = x.clone()
    changed_x[:, 3] += 100.0
    unpadded = torch.zeros(1, 4, dtype=torch.bool)
    with torch.no_grad():
        out_a = attention(x, unpadded)
        out_b = attention(changed_x, unpadded)
    causal_error = float((out_a[:, :3] - out_b[:, :3]).abs().max())
    if causal_error > 1e-6:
        raise RuntimeError(f"causal attention mask failed: max_abs={causal_error}")
    return {"status": "passed", "seed": seed, "actual_train_event_indices": batch_ids.tolist(),
            "common_initial_state_sha256": next(iter(common_hashes)),
            "arms": records, "r1_vs_self_first_pass_max_abs_error": parity_error,
            "padding_mask_max_abs_error": padding_error,
            "causal_mask_max_abs_error": causal_error,
            "full_catalog_last_position_only_logits": True}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def memory_snapshot() -> dict:
    mem_kib = next(int(x.split()[1]) for x in Path("/proc/meminfo").read_text().splitlines()
                   if x.startswith("MemAvailable:"))
    return {"mem_available_bytes": mem_kib * 1024}


def resource_gate(path: Path) -> dict:
    snapshot = memory_snapshot()
    snapshot["free_disk_bytes"] = __import__("shutil").disk_usage(path).free
    if snapshot["mem_available_bytes"] < 500 * 1024**2:
        raise RuntimeError(f"MemAvailable below 500 MiB: {snapshot}")
    if snapshot["free_disk_bytes"] < 5 * 1024**3:
        raise RuntimeError(f"free disk below 5 GiB: {snapshot}")
    return snapshot


def validate_inputs(config_path: Path, config: dict, feature: Path, index_root: Path,
                    aux: Path, output: Path, stage_root: Path,
                    expected_config_sha: str) -> dict:
    if stage_root.exists():
        raise FileExistsError(f"refusing preexisting staged-input directory: {stage_root}")
    if sha256(config_path) != expected_config_sha:
        raise RuntimeError("matched continuation config SHA mismatch")
    if sha256(feature / "manifest.json") != FEATURE_MANIFEST_SHA:
        raise RuntimeError("event feature manifest SHA mismatch")
    if sha256(index_root / "manifest.json") != INDEX_INPUT_MANIFEST_SHA:
        raise RuntimeError("full-support index input manifest SHA mismatch")
    index_manifest = json.loads((index_root / "manifest.json").read_text())
    source_manifest_path = index_root / "source-data-manifest.json"
    if (sha256(source_manifest_path) != FULL_SOURCE_MANIFEST_SHA
            or index_manifest.get("source_data_manifest_sha256") != FULL_SOURCE_MANIFEST_SHA
            or index_manifest.get("feature_manifest_sha256") != FEATURE_MANIFEST_SHA):
        raise RuntimeError("full-supported source/feature manifest chain mismatch")
    aux_manifest_path = aux / "manifest.json"
    aux_manifest = json.loads(aux_manifest_path.read_text())
    if (aux_manifest.get("dataset_id") != "arsen4ikvar/recsys-core-aux-20261019"
            or aux_manifest.get("feature_manifest_sha256") != FEATURE_MANIFEST_SHA):
        raise RuntimeError("checkpoint auxiliary manifest does not match this feature input")
    # Aux manifest is lineage-only; this from-scratch study does not load pretrained weights.
    feature_manifest = json.loads((feature / "manifest.json").read_text())
    expected_bytes = sum(int(feature_manifest["files"][f"{name}.npy"]["bytes"])
                         for name in ARRAY_FILES)
    expected_bytes += int(feature_manifest["files"]["train_indices.npy"]["bytes"])
    expected_bytes += int(feature_manifest["files"]["valid_indices.npy"]["bytes"])
    expected_bytes += (index_root / "train_indices.npy").stat().st_size
    expected_bytes += (feature / "manifest.json").stat().st_size + source_manifest_path.stat().st_size
    stage_disk_free = __import__("shutil").disk_usage(stage_root.parent).free
    if stage_disk_free - expected_bytes < 5 * 1024**3:
        raise RuntimeError(f"staging {expected_bytes} bytes would violate the 5 GiB disk floor; free={stage_disk_free}")
    stage_root.mkdir(parents=True)
    staged_feature = stage_root / "features"
    staged_indices = stage_root / "indices"
    staged_aux = stage_root / "aux"
    staged_feature.mkdir()
    staged_indices.mkdir()
    staged_aux.mkdir()
    staged_files = []
    for name in ARRAY_FILES:
        record = feature_manifest["files"][f"{name}.npy"]
        staged_files.append(copy_verified(feature / f"{name}.npy", staged_feature / f"{name}.npy",
                                          record["sha256"], int(record["bytes"])))
    for name, source, expected, expected_bytes in (
            ("subset_train_indices.npy", feature / "train_indices.npy", SUBSET_INDICES_SHA,
             int(feature_manifest["files"]["train_indices.npy"]["bytes"])),
            ("valid_indices.npy", feature / "valid_indices.npy",
             feature_manifest["files"]["valid_indices.npy"]["sha256"],
             int(feature_manifest["files"]["valid_indices.npy"]["bytes"])),
            ("full_train_indices.npy", index_root / "train_indices.npy", FULL_INDICES_SHA,
             (index_root / "train_indices.npy").stat().st_size)):
        staged_files.append(copy_verified(source, staged_indices / name, expected, expected_bytes))
    staged_files.append(copy_verified(feature / "manifest.json", staged_feature / "manifest.json",
                                      FEATURE_MANIFEST_SHA, (feature / "manifest.json").stat().st_size))
    staged_files.append(copy_verified(source_manifest_path, staged_indices / "source-data-manifest.json",
                                      FULL_SOURCE_MANIFEST_SHA, source_manifest_path.stat().st_size))
    copied_bytes = sum(x["bytes"] for x in staged_files)
    if copied_bytes != int(config["data"]["expected_staged_payload_bytes"]):
        raise RuntimeError(f"staged payload byte total mismatch: {copied_bytes}")
    subset_path = staged_indices / "subset_train_indices.npy"
    full_path = staged_indices / "full_train_indices.npy"
    subset_indices = np.load(subset_path, mmap_mode="r")
    full_indices = np.load(full_path, mmap_mode="r")
    if (len(subset_indices) != 4_110_754 or len(full_indices) != 41_142_628
            or index_manifest.get("train_indices_sha256") != FULL_INDICES_SHA
            or index_manifest.get("train_anchor_count") != len(full_indices)):
        raise RuntimeError("training-index count/manifest mismatch")
    valid = np.load(staged_indices / "valid_indices.npy", mmap_mode="r")
    val_ids = np.random.default_rng(991).choice(valid, 10_000, replace=False).astype(np.uint32)
    if hashlib.sha256(val_ids.tobytes()).hexdigest() != VALIDATION_EVENT_SHA:
        raise RuntimeError("fixed seed-991 validation event order mismatch")
    missing = [name for name in (f"{item}.npy" for item in ARRAY_FILES)
        if not (staged_feature / name).is_file()]
    if missing:
        raise RuntimeError(f"feature input missing required arrays: {missing}")
    disk_free = __import__("shutil").disk_usage(output.parent).free
    resources = memory_snapshot()
    if disk_free < 5 * 1024**3:
        raise RuntimeError(f"free disk below 5 GiB: {disk_free}")
    if resources["mem_available_bytes"] < 500 * 1024**2:
        raise RuntimeError(f"MemAvailable below 500 MiB: {resources}")
    torch.set_num_threads(WORKER_CPU_THREADS)
    torch.set_num_interop_threads(1)
    return {"feature_manifest_sha256": FEATURE_MANIFEST_SHA,
            "index_manifest_sha256": INDEX_INPUT_MANIFEST_SHA,
            "full_source_manifest_sha256": FULL_SOURCE_MANIFEST_SHA,
            "aux_manifest_sha256": sha256(aux_manifest_path),
            "initialization_mode": "fresh random; no pretrained checkpoint read or loaded",
            "subset_indices_sha256": SUBSET_INDICES_SHA, "full_indices_sha256": FULL_INDICES_SHA,
            "validation_event_indices_sha256_uint32": VALIDATION_EVENT_SHA,
            "staged_feature_root": str(staged_feature), "staged_index_root": str(staged_indices),
            "staged_aux_root": str(staged_aux), "staged_files": staged_files,
            "stage_copy_bytes": copied_bytes,
            "stage_copy_seconds": round(sum(x["copy_seconds"] for x in staged_files), 3),
            "train_anchor_counts": {"subset": len(subset_indices), "full_supported": len(full_indices)},
            "free_disk_bytes": disk_free, **resources}


def cuda_device_preflight() -> dict:
    # Called only after stage and every staged-input SHA check has completed.
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(f"exactly two visible CUDA devices required, got {torch.cuda.device_count()}")
    devices = []
    for i in range(2):
        prop = torch.cuda.get_device_properties(i)
        devices.append({"index": i, "name": prop.name, "total_memory_bytes": prop.total_memory})
    if any("T4" not in item["name"] for item in devices):
        raise RuntimeError(f"expected two T4 devices, got {devices}")
    return {"device_count": 2, "devices": devices}


def worker_test_export(job: dict) -> int:
    """Score every frozen test event once, only after all mode doses are locked."""
    seed = int(job["seed"])
    seed_all(int(job["evaluation_seed"]))
    device = torch.device("cuda:0")
    test_path = Path(job["test_indices_path"])
    if sha256(test_path) != TEST_INDICES_SHA:
        raise RuntimeError("frozen full-test event index SHA mismatch")
    indices = np.load(test_path, mmap_mode="r")
    if len(indices) != 232_049:
        raise RuntimeError(f"unexpected frozen full-test count: {len(indices)}")
    arrays = open_event_arrays(Path(job["feature_root"]))
    config = job["recipe"]
    model = HuginnItemGPT(**config["architecture"], mode=job["arm"], seed=seed).to(device)
    checkpoint = torch.load(Path(job["final_checkpoint_path"]), map_location=device, weights_only=False)
    expected = job["expected_common_initial_state_sha256"]
    if checkpoint["config"]["common_initial_state_sha256"] != expected:
        raise RuntimeError("test checkpoint common initialization SHA mismatch")
    if checkpoint["config"]["mode"] != job["arm"] or int(checkpoint["config"]["seed"]) != seed:
        raise RuntimeError("test checkpoint arm or seed mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    scores, rows = eval_policy(model, indices, arrays,
                               batch_size=int(job["batch_size"]), device=device, return_rows=True)
    out = Path(job["rows_path"])
    np.savez_compressed(out, **rows)
    result = {"status": "complete", "arm": job["arm"], "seed": seed,
              "test_event_count": len(indices), "test_indices_sha256": TEST_INDICES_SHA,
              "test_event_order_sha256_uint32": hashlib.sha256(np.asarray(indices, dtype=np.uint32).tobytes()).hexdigest(),
              "checkpoint_sha256": sha256(Path(job["final_checkpoint_path"])),
              "common_initial_state_sha256": expected, "evaluation_seed": int(job["evaluation_seed"]),
              "metrics": scores, "rows_path": str(out), "rows_sha256": sha256(out),
              "test_scored_only_after_all_training": True}
    atomic_json(Path(job["result_path"]), result)
    return 0


def resolve_worker_deadline(job: dict, seed: int, max_context: int) -> tuple[str | None, str | None, str | None]:
    deadline_utc = job.get("deadline_utc")
    deadline_source = "job.deadline_utc" if deadline_utc else None
    deadline_source_sha256 = None
    if not deadline_utc and job.get("mode") == "final":
        # The already-loaded A100 controller may still create legacy jobs
        # without a deadline field. Read its pinned study config directly so
        # replacing the worker after R1 does not silently lose the cutoff.
        candidates = []
        for ancestor in Path(__file__).resolve().parents:
            config_dir = ancestor / "infra" / "a100" / "huginn-context256-20261025" / "configs"
            if config_dir.is_dir():
                candidates.extend(sorted(config_dir.glob(f"context{max_context}*-seed{seed}.json")))
        for candidate in candidates:
            deadline_config = json.loads(candidate.read_text())
            source_recipe = dict(deadline_config.get("recipe", {}))
            job_recipe = job.get("recipe", {})
            # Recreate only the runner's deterministic dose/batch edits, so the
            # fallback identifies the exact loaded config (e.g. context256c).
            for key in ("global_batch_size", "steps", "steps_per_arm", "examples_per_arm"):
                if key in job_recipe:
                    source_recipe[key] = job_recipe[key]
            if "evaluations" in job_recipe and "evaluations" in source_recipe:
                source_recipe["evaluations"] = dict(source_recipe["evaluations"])
                for key in ("steps", "interval_steps"):
                    if key in job_recipe["evaluations"]:
                        source_recipe["evaluations"][key] = job_recipe["evaluations"][key]
            if source_recipe != job_recipe:
                continue
            deadline_utc = deadline_config.get("resources", {}).get("deadline_utc")
            if deadline_utc:
                deadline_source = str(candidate)
                deadline_source_sha256 = sha256(candidate)
                break
        output_hint = str(job.get("progress_path", job.get("final_checkpoint_path", "")))
        if (not deadline_utc and job.get("mode") == "final"
                and "/infra/a100/huginn-context256-20261025/output/" in output_hint):
            raise RuntimeError("A100 final worker job has no matching deadline-bearing study config")
    return deadline_utc, deadline_source, deadline_source_sha256


def worker_train(args: argparse.Namespace) -> int:
    torch.set_num_threads(WORKER_CPU_THREADS)
    torch.set_num_interop_threads(1)
    job = json.loads(args.job_config.read_text())
    if job.get("mode") == "test_export":
        return worker_test_export(job)
    cfg = job["recipe"]
    seed = int(job["seed"])
    seed_all(seed)
    device = torch.device("cuda:0")
    feature = Path(job["feature_root"])
    indices_path = Path(job["train_indices_path"])
    indices = np.load(indices_path, mmap_mode="r")
    valid_all = np.load(Path(job["valid_indices_path"]), mmap_mode="r")
    valid_indices = np.random.default_rng(991).choice(valid_all, 10_000, replace=False).astype(np.uint32)
    if hashlib.sha256(valid_indices.tobytes()).hexdigest() != VALIDATION_EVENT_SHA:
        raise RuntimeError("worker validation event set/order differs from the frozen seed-991 sample")
    arrays = open_event_arrays(feature)
    arch = cfg["architecture"]
    model = HuginnItemGPT(**arch, mode=job["arm"], seed=seed).to(device)
    initial_state_sha256 = state_dict_sha256(model)
    common_initial_state_sha256 = common_init_sha256(model)
    structure = model_structure(model)
    # Keep the base module as optimizer/checkpoint owner so state_dict keys
    # remain unprefixed; use the compiled wrapper only for training forwards.
    forward_model = (torch.compile(model, mode=cfg.get("compile_mode", "reduce-overhead"))
                     if cfg.get("compile_model", False) else model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"],
                                  betas=tuple(cfg.get("betas", (0.9, 0.95))))
    use_amp = str(cfg.get("precision", "fp16")) == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batch_size = int(cfg["global_batch_size"])
    steps = int(job["steps"])
    resume_path = job.get("resume_checkpoint_path")
    deadline_utc, deadline_source, deadline_source_sha256 = resolve_worker_deadline(
        job, seed, int(arch["max_context"]))
    deadline_timestamp = None
    if deadline_utc:
        parsed_deadline = datetime.fromisoformat(str(deadline_utc).replace("Z", "+00:00"))
        if parsed_deadline.tzinfo is None:
            raise ValueError("deadline_utc must include a timezone")
        deadline_timestamp = parsed_deadline.astimezone(timezone.utc).timestamp()
    start_step = 0
    permutation_seed = seed
    cursor = 0
    examples = 0
    optimizer_updates = 0
    skipped_updates = 0
    high_skip_checkpoints = 0
    if resume_path:
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (resume.get("checkpoint_kind") != "training_resume_candidate"
                or resume["config"]["arm"] != job["arm"]
                or int(resume["config"]["seed"]) != seed):
            raise RuntimeError("resume checkpoint does not match arm/seed")
        model.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume["grad_scaler_state_dict"])
        start_step = int(resume["step"])
        examples = int(resume["examples_seen"])
        optimizer_updates = int(resume["optimizer_updates"])
        skipped_updates = int(resume["skipped_updates"])
        high_skip_checkpoints = int(resume["high_skip_checkpoints"])
        permutation_seed = int(resume["sampler"]["permutation_seed"])
        cursor = int(resume["sampler"]["cursor"])
        initial_state_sha256 = resume["config"]["initial_state_sha256"]
        common_initial_state_sha256 = resume["config"]["common_initial_state_sha256"]
        rng = resume["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
    permutation = np.random.default_rng(permutation_seed).permutation(indices)
    start = time.monotonic()
    result_path = Path(job["result_path"])
    progress_path = Path(job["progress_path"])
    metrics_path = Path(job["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if job["mode"] == "final" and start_step == 0:
        model.eval()
        with torch.inference_mode():
            initial = eval_policy(model, valid_indices, arrays, batch_size=batch_size, device=device)
        with metrics_path.open("w") as stream:
            stream.write(json.dumps({"step": 0, "validation": initial,
                                    "elapsed_seconds": round(time.monotonic() - start, 3)}) + "\n")
        model.train()
    torch.cuda.reset_peak_memory_stats(device)
    train_started = time.monotonic()
    component_samples: list[dict] = []

    def save_resume(step: int) -> Path:
        checkpoint_path = progress_path.with_name("resume-latest.pt")
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save({
            "checkpoint_kind": "training_resume_candidate",
            "worker_source_sha256": sha256(Path(__file__)),
            "step": step,
            "examples_seen": examples,
            "optimizer_updates": optimizer_updates,
            "skipped_updates": skipped_updates,
            "high_skip_checkpoints": high_skip_checkpoints,
            "sampler": {"seed": seed, "permutation_seed": permutation_seed,
                        "cursor": cursor, "batch_size": batch_size},
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": scaler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda_all": torch.cuda.get_rng_state_all(),
            },
            "config": {"seed": seed, "arm": job["arm"], "recipe": cfg,
                       "initial_state_sha256": initial_state_sha256,
                       "common_initial_state_sha256": common_initial_state_sha256},
        }, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        return checkpoint_path

    def save_deadline_partial(step: int, loss: torch.Tensor | None) -> int:
        torch.cuda.synchronize(device)
        resume_checkpoint = save_resume(step)
        partial_path = progress_path.with_name(f"partial-model-step-{step}.pt")
        temporary_path = partial_path.with_suffix(partial_path.suffix + ".tmp")
        torch.save({"state_dict": model.state_dict(), "config": {
            **arch, "batch_size": batch_size, "learning_rate": cfg["learning_rate"],
            "weight_decay": cfg["weight_decay"], "steps": step, "seed": seed,
            "initialization_mode": "fresh_random", "initial_state_sha256": initial_state_sha256,
            "common_initial_state_sha256": common_initial_state_sha256,
            "mode": job["arm"], "model_structure": structure,
            "fresh_optimizer": "AdamW", "incomplete": True},
            "step": step, "primary_endpoint": "incomplete deadline checkpoint"}, temporary_path)
        os.replace(temporary_path, partial_path)
        partial = {"run_id": job["run_id"], "status": "incomplete_deadline_checkpointed",
                   "arm": job["arm"], "seed": seed, "step": step, "target_steps": steps,
                   "worker_source_sha256": sha256(Path(__file__)),
                   "examples_seen": examples, "optimizer_updates": optimizer_updates,
                   "skipped_updates": skipped_updates,
                   "grad_scaler_final_scale": float(scaler.get_scale()),
                   "last_train_loss": (float(loss.detach().float().cpu()) if loss is not None else None),
                   "resume_checkpoint_path": str(resume_checkpoint),
                   "resume_checkpoint_sha256": sha256(resume_checkpoint),
                   "partial_model_path": str(partial_path),
                   "partial_model_sha256": sha256(partial_path),
                   "deadline_utc": deadline_utc, "deadline_source": deadline_source,
                   "deadline_source_sha256": deadline_source_sha256}
        atomic_json(result_path, partial)
        atomic_json(progress_path, {"run_id": job["run_id"], "step": step,
            "examples_seen": examples, "optimizer_updates": optimizer_updates,
            "skipped_updates": skipped_updates, "status": partial["status"],
            "deadline_utc": deadline_utc})
        print(json.dumps({"training_incomplete": partial}), flush=True)
        return 75

    for step in range(start_step + 1, steps + 1):
        if (job["mode"] == "final" and deadline_timestamp is not None
                and time.time() >= deadline_timestamp and step < steps):
            return save_deadline_partial(step - 1, None)
        whole_step_started = time.perf_counter()
        if cursor + batch_size > len(permutation):
            permutation_seed = seed + step
            permutation = np.random.default_rng(permutation_seed).permutation(indices)
            cursor = 0
        batch_ids = permutation[cursor:cursor + batch_size]
        if len(batch_ids) != batch_size:
            raise RuntimeError("training sampler yielded a partial batch")
        cursor += batch_size
        if job["mode"] == "profile":
            torch.cuda.synchronize(device)
            gather_started = time.perf_counter()
        else:
            gather_started = None
        batch_cpu = make_batch(batch_ids, arrays, n_items=arch["n_items"],
                               max_context=arch["max_context"], device=torch.device("cpu"))
        if job["mode"] == "profile":
            gather_s = time.perf_counter() - gather_started
            torch.cuda.synchronize(device)
            mark = time.perf_counter()
        batch = {key: value.to(device) for key, value in batch_cpu.items()}
        if job["mode"] == "profile":
            torch.cuda.synchronize(device)
            h2d_s = time.perf_counter() - mark
        optimizer.zero_grad(set_to_none=True)
        warmup_steps = int(cfg.get("warmup_steps", 0))
        current_lr = cfg["learning_rate"] * min(1.0, step / max(1, warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        if job["mode"] == "profile":
            torch.cuda.synchronize(device)
            mark = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = forward_model(batch["tokens"], batch["padding_mask"])
            loss = F.cross_entropy(logits[batch["supported"]], batch["target"][batch["supported"]])
        if job["mode"] == "profile":
            torch.cuda.synchronize(device)
            forward_s = time.perf_counter() - mark
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if job["mode"] == "profile":
            torch.cuda.synchronize(device)
            mark = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
        if job["mode"] == "profile":
            torch.cuda.synchronize(device)
            backward_s = time.perf_counter() - mark
        scale_before = scaler.get_scale()
        if job["mode"] == "profile":
            torch.cuda.synchronize(device)
            mark = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        if job["mode"] == "profile":
            torch.cuda.synchronize(device)
            optimizer_s = time.perf_counter() - mark
            component_samples.append({"step": step, "cpu_batch_gather_seconds": gather_s,
                                      "host_to_device_seconds": h2d_s,
                                      "forward_and_loss_seconds": forward_s,
                                      "backward_unscale_clip_seconds": backward_s,
                                      "optimizer_step_update_seconds": optimizer_s,
                                      "total_seconds": time.perf_counter() - whole_step_started})
        if scaler.get_scale() >= scale_before:
            optimizer_updates += 1
        else:
            skipped_updates += 1
        examples += batch_size
        if job["mode"] == "final" and step >= 1000 and step % 1000 == 0:
            post_warmup_steps = step - 100
            skip_fraction = skipped_updates / post_warmup_steps
            high_skip_checkpoints = high_skip_checkpoints + 1 if skip_fraction > 0.01 else 0
            if high_skip_checkpoints >= 2:
                raise RuntimeError(
                    f"persistent GradScaler skipped-update fraction exceeded 1% after warmup: "
                    f"{skipped_updates}/{post_warmup_steps} at step {step}; fixed exposure retained")
        if ((job["mode"] == "profile" and step % 8 == 0)
                or (job["mode"] != "profile" and (step % 500 == 0 or step == steps))):
            snapshot = {"run_id": job["run_id"], "step": step,
                        "examples_seen": examples,
                        "optimizer_updates": optimizer_updates,
                        "skipped_updates": skipped_updates,
                        "grad_scaler_scale": float(scaler.get_scale()),
                        "elapsed_seconds": round(time.monotonic() - train_started, 3),
                        "loss": float(loss.detach().float().cpu()),
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            atomic_json(progress_path, snapshot)
            print(json.dumps({"progress": snapshot}), flush=True)
        evaluation_steps = set(int(x) for x in cfg["evaluations"].get("steps", []))
        if job["mode"] == "final" and step in evaluation_steps:
            torch.cuda.synchronize(device)
            model.eval()
            with torch.inference_mode():
                if step == steps:
                    scores, rows = eval_policy(model, valid_indices, arrays, batch_size=batch_size,
                                               device=device, return_rows=True)
                    np.savez_compressed(Path(job["validation_rows_path"]), **rows)
                else:
                    scores = eval_policy(model, valid_indices, arrays, batch_size=batch_size, device=device)
            record = {"step": step, "validation": scores,
                      "elapsed_seconds": round(time.monotonic() - start, 3)}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            model.train()
        checkpoint_interval = int(job.get("resume_checkpoint_interval_steps", 8000))
        if (job["mode"] == "final" and checkpoint_interval > 0
                and step % checkpoint_interval == 0 and step < steps):
            save_resume(step)
        if (job["mode"] == "final" and deadline_timestamp is not None
                and time.time() >= deadline_timestamp and step < steps):
            return save_deadline_partial(step, loss)
    torch.cuda.synchronize(device)
    train_elapsed = time.monotonic() - train_started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if job["mode"] == "final":
        final_path = Path(job["final_checkpoint_path"])
        torch.save({"state_dict": model.state_dict(), "config": {
            **arch, "batch_size": batch_size, "learning_rate": cfg["learning_rate"],
            "weight_decay": cfg["weight_decay"], "steps": steps, "seed": seed,
            "initialization_mode": "fresh_random", "initial_state_sha256": initial_state_sha256,
            "common_initial_state_sha256": common_initial_state_sha256,
            "mode": job["arm"], "model_structure": structure,
            "fresh_optimizer": "AdamW"},
            "step": steps, "primary_endpoint": "fixed final step; no best-checkpoint selection"}, final_path)
    train_ended = time.monotonic()
    result = {"run_id": job["run_id"], "mode": job["mode"], "arm": job["arm"],
              "worker_source_sha256": sha256(Path(__file__)),
              "deadline_utc": deadline_utc, "deadline_source": deadline_source,
              "deadline_source_sha256": deadline_source_sha256,
              "seed": seed, "device": str(device), "steps": steps,
              "examples_seen": examples, "optimizer_updates": optimizer_updates,
              "skipped_updates": skipped_updates,
              "grad_scaler_final_scale": float(scaler.get_scale()),
              "train_seconds": round(train_elapsed, 3),
              "train_started_monotonic": train_started,
              "train_ended_monotonic": train_ended,
              "examples_per_second": round(examples / max(train_elapsed, 1e-9), 3),
              "cuda_peak_allocated_bytes": peak_allocated,
              "cuda_peak_reserved_bytes": peak_reserved,
              "initialization_mode": "fresh_random", "initial_state_sha256": initial_state_sha256,
              "common_initial_state_sha256": common_initial_state_sha256,
              "model_structure": structure,
              "validation_event_indices_sha256_uint32": VALIDATION_EVENT_SHA}
    if job["mode"] == "profile":
        warm = int(job.get("profile_warm_steps", 16))
        timed = component_samples[warm:]
        if len(timed) < 32:
            raise RuntimeError(f"profile produced only {len(timed)} timed steps after warmup")
        keys = ("cpu_batch_gather_seconds", "host_to_device_seconds", "forward_and_loss_seconds",
                "backward_unscale_clip_seconds", "optimizer_step_update_seconds", "total_seconds")
        result["profile"] = {
            "total_steps": steps, "warm_steps": warm, "timed_steps": len(timed),
            "per_step_samples": component_samples,
            "p75_seconds_by_component": {
                key: float(np.percentile([x[key] for x in timed], 75)) for key in keys},
            "median_seconds_by_component": {
                key: float(np.median([x[key] for x in timed])) for key in keys},
            "optimizer_updates": optimizer_updates, "skipped_updates": skipped_updates,
            "cuda_name": torch.cuda.get_device_name(device),
        }
        atomic_json(Path(job["result_path"]), result)
        return 0
    if job["mode"] == "final":
        result.update({"final_checkpoint_path": job["final_checkpoint_path"],
                       "final_checkpoint_sha256": sha256(Path(job["final_checkpoint_path"])),
                       "metrics_path": str(metrics_path),
                       "validation_rows_path": job["validation_rows_path"],
                       "validation_rows_sha256": sha256(Path(job["validation_rows_path"]))})
    atomic_json(result_path, result)
    return 0


def paired_uid_bootstrap(subset_rows, full_rows, seed: int, replicates: int = 1000) -> dict:
    event_ids = np.asarray(subset_rows["event_indices"], dtype=np.uint32)
    if not np.array_equal(event_ids, np.asarray(full_rows["event_indices"], dtype=np.uint32)):
        raise RuntimeError("paired bootstrap event order mismatch")
    uid = np.asarray(subset_rows["uid"], dtype=np.int64)
    if not np.array_equal(uid, np.asarray(full_rows["uid"], dtype=np.int64)):
        raise RuntimeError("paired bootstrap user order mismatch")
    supported = np.asarray(subset_rows["supported"], dtype=bool)
    if not np.array_equal(supported, np.asarray(full_rows["supported"], dtype=bool)):
        raise RuntimeError("paired bootstrap support flags mismatch")
    rank_sub = np.asarray(subset_rows["rank"], dtype=np.int64)
    rank_full = np.asarray(full_rows["rank"], dtype=np.int64)
    ndcg_sub = np.where(supported & (rank_sub <= 100), 1.0 / np.log2(rank_sub + 1.0), 0.0)
    ndcg_full = np.where(supported & (rank_full <= 100), 1.0 / np.log2(rank_full + 1.0), 0.0)
    recall_sub = (supported & (rank_sub <= 100)).astype(np.float64)
    recall_full = (supported & (rank_full <= 100)).astype(np.float64)
    users, inverse = np.unique(uid, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(users)).astype(np.float64)
    sum_ndcg_sub = np.bincount(inverse, weights=ndcg_sub, minlength=len(users))
    sum_ndcg_full = np.bincount(inverse, weights=ndcg_full, minlength=len(users))
    sum_recall_sub = np.bincount(inverse, weights=recall_sub, minlength=len(users))
    sum_recall_full = np.bincount(inverse, weights=recall_full, minlength=len(users))
    rng = np.random.default_rng(seed)
    ndcg_delta = np.empty(replicates, dtype=np.float64)
    recall_delta = np.empty(replicates, dtype=np.float64)
    for rep in range(replicates):
        picks = rng.integers(0, len(users), size=len(users))
        denominator = counts[picks].sum()
        ndcg_delta[rep] = (sum_ndcg_full[picks].sum() - sum_ndcg_sub[picks].sum()) / denominator
        recall_delta[rep] = (sum_recall_full[picks].sum() - sum_recall_sub[picks].sum()) / denominator
    def interval(values):
        return [float(x) for x in np.percentile(values, [2.5, 97.5])]
    return {"scope": "all validation events; paired event-weighted user-cluster bootstrap",
            "replicates": replicates, "seed": seed, "user_count": int(len(users)),
            "event_count": int(len(uid)), "supported_event_count": int(supported.sum()),
            "primary_R4source_minus_R4self_ndcg100": float((ndcg_full - ndcg_sub).mean()),
            "primary_ndcg100_ci95": interval(ndcg_delta),
            "secondary_R4source_minus_R4self_recall100": float((recall_full - recall_sub).mean()),
            "secondary_recall100_ci95": interval(recall_delta),
            "conditional_scope_note": "Uncertainty is conditional on these two fitted arms for this initialization seed; it is not uncertainty over model initializations.",
            "validation_event_order_sha256": hashlib.sha256(event_ids.tobytes()).hexdigest()}


def proc_tree(root_pid: int) -> set[int]:
    parents: dict[int, list[int]] = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "stat").read_text()
            ppid = int(raw[raw.rfind(")") + 2:].split()[1])
            parents.setdefault(ppid, []).append(int(item.name))
        except (OSError, ValueError, IndexError):
            continue
    found: set[int] = set()
    todo = list(parents.get(root_pid, []))
    while todo:
        pid = todo.pop()
        if pid not in found:
            found.add(pid)
            todo.extend(parents.get(pid, []))
    return found


def stop_children(processes: list[subprocess.Popen], grace: float = 8.0) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace
    for proc in processes:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass


def gpu_sample() -> list[dict]:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"], text=True, timeout=2)
    samples = []
    for line in raw.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 5:
            raise RuntimeError(f"unexpected nvidia-smi telemetry row: {line}")
        samples.append({"index": int(parts[0]), "name": parts[1],
                        "util_percent": int(parts[2]), "memory_used_mib": int(parts[3]),
                        "memory_total_mib": int(parts[4])})
    if len(samples) != 2:
        raise RuntimeError(f"expected telemetry for two GPUs, got {len(samples)}")
    return sorted(samples, key=lambda item: item["index"])


def spawn_tee(command: list[str], env: dict, log_path: Path) -> tuple[subprocess.Popen, threading.Thread]:
    """Flush child output both to its immutable log and the live kernel stream."""
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, bufsize=0)
    assert proc.stdout is not None
    def copy_lines() -> None:
        with log_path.open("xb") as log:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                log.write(line)
                log.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
    thread = threading.Thread(target=copy_lines, daemon=True)
    thread.start()
    return proc, thread


def run_phase(phase: str, assignments: dict[int, list[tuple[str, int, str]]],
              common: dict, output: Path, deadline: float) -> dict:
    phase_root = output / "pilot" / phase
    phase_root.mkdir(parents=True, exist_ok=False)
    proc_meta: list[dict] = []
    procs: list[subprocess.Popen] = []
    tee_threads: list[threading.Thread] = []
    telemetry_path = phase_root / "gpu-telemetry.jsonl"
    started = time.monotonic()
    for gpu, jobs in assignments.items():
        for arm, seed, label in jobs:
            run_id = f"{phase}_{label}"
            run_dir = phase_root / run_id
            run_dir.mkdir()
            job = {**common, "mode": "profile", "arm": arm, "seed": seed,
                   "run_id": run_id, "steps": common["profile_steps"],
                   "profile_warm_steps": common["profile_warm_steps"],
                   "train_indices_path": str(common["indices"][arm]),
                   "result_path": str(run_dir / "pilot-result.json"),
                   "progress_path": str(run_dir / "progress.json"),
                   "metrics_path": str(run_dir / "metrics.jsonl"),
                   "final_checkpoint_path": str(run_dir / "NO_CHECKPOINT")}
            job_path = run_dir / "job.json"
            atomic_json(job_path, job)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)]
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"})
            launch_mono = time.monotonic()
            proc, tee_thread = spawn_tee(command, env, run_dir / "worker.log")
            procs.append(proc)
            tee_threads.append(tee_thread)
            proc_meta.append({"gpu": gpu, "arm": arm, "seed": seed, "label": label,
                              "run_id": run_id, "pid": proc.pid,
                              "started_monotonic": launch_mono,
                              "completed_monotonic": None,
                              "profile_child_timeout_seconds": common["profile_child_timeout_seconds"],
                              "result_path": str(run_dir / "pilot-result.json"), "log": str(run_dir / "worker.log")})
    phase_status = {"phase": phase, "status": "running", "started_elapsed_seconds": round(started - common["started_monotonic"], 3),
                    "worker_assignments": proc_meta}
    atomic_json(phase_root / "phase-status.json", phase_status)
    try:
        while any(proc.poll() is None for proc in procs):
            for item, proc in zip(proc_meta, procs):
                if proc.poll() is not None and item["completed_monotonic"] is None:
                    item["completed_monotonic"] = time.monotonic()
            if time.monotonic() >= deadline:
                raise TimeoutError("shared worker deadline reached during bounded profile")
            active_pids = {proc.pid for proc in procs if proc.poll() is None}
            timed_out = [x for x in proc_meta if x["pid"] in active_pids
                         and time.monotonic() - x["started_monotonic"] > common["profile_child_timeout_seconds"]]
            if timed_out:
                raise TimeoutError(f"bounded profile child exceeded 90s cap: {timed_out}")
            resource_gate(output)
            snapshots = gpu_sample()
            with telemetry_path.open("a") as stream:
                stream.write(json.dumps({"elapsed_seconds": round(time.monotonic() - common["started_monotonic"], 3),
                                         "gpus": snapshots}) + "\n")
            phase_status["last_gpu_telemetry"] = snapshots
            phase_status["active"] = [{"pid": p.pid, "alive": p.poll() is None} for p in procs]
            atomic_json(phase_root / "phase-status.json", phase_status)
            time.sleep(1.0)
        codes = [p.wait() for p in procs]
        for item, proc in zip(proc_meta, procs):
            if item["completed_monotonic"] is None:
                item["completed_monotonic"] = time.monotonic()
        completed_after_cap = [x for x in proc_meta if x["completed_monotonic"] - x["started_monotonic"]
                               > common["profile_child_timeout_seconds"]]
        if completed_after_cap:
            raise TimeoutError(f"bounded profile child exceeded 90s cap: {completed_after_cap}")
        for thread in tee_threads:
            thread.join(timeout=1)
        if any(code != 0 for code in codes):
            raise RuntimeError(f"pilot worker return codes: {codes}")
        end = time.monotonic()
        results = [json.loads(Path(x["result_path"]).read_text()) for x in proc_meta]
        throughput = {}
        train_windows = {}
        for gpu in assignments:
            selected = [(r, m) for r, m in zip(results, proc_meta) if m["gpu"] == gpu]
            interval_start = min(r["train_started_monotonic"] for r, _ in selected)
            interval_end = max(r["train_ended_monotonic"] for r, _ in selected)
            elapsed = interval_end - interval_start
            train_windows[str(gpu)] = {"start_monotonic": interval_start,
                                       "end_monotonic": interval_end,
                                       "elapsed_seconds": round(elapsed, 3)}
            throughput[str(gpu)] = round(sum(r["examples_seen"] for r, _ in selected) / max(elapsed, 1e-9), 2)
        telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines() if line.strip()]
        peak_used = {str(gpu): max(row["gpus"][gpu]["memory_used_mib"] for row in telemetry)
                     for gpu in range(2)}
        avg_util = {str(gpu): round(sum(row["gpus"][gpu]["util_percent"] for row in telemetry) / max(1, len(telemetry)), 1)
                    for gpu in range(2)}
        result = {"phase": phase, "status": "complete",
                  "phase_wall_seconds": round(end - started, 3),
                  "per_gpu_train_windows": train_windows,
                  "aggregate_examples_per_second_per_gpu": throughput,
                  "peak_memory_used_mib_per_gpu": peak_used,
                  "average_sampled_gpu_util_percent": avg_util,
                  "worker_results": results, "telemetry_samples": len(telemetry)}
        atomic_json(phase_root / "phase-result.json", result)
        return result
    except BaseException as exc:
        atomic_json(phase_root / "phase-failure.json", {
            "phase": phase, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        stop_children(procs)
        raise


def run_final_wave(name: str, jobs: list[tuple[int, str, int]], common: dict,
                   output: Path, deadline: float, monitor: LiveMonitor) -> dict:
    wave_root = output / "final" / name
    wave_root.mkdir(parents=True, exist_ok=False)
    procs = []
    tee_threads = []
    metadata = []
    for gpu, arm, seed in jobs:
        run_id = f"{arm}_seed{seed}"
        run_dir = output / "final" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        job = {**common, "mode": "final", "arm": arm, "seed": seed,
               "run_id": run_id, "steps": int(common["recipe"]["steps_per_arm"]),
               "train_indices_path": str(common["indices"][arm]),
               "result_path": str(run_dir / "run-result.json"),
               "progress_path": str(run_dir / "progress.json"),
               "metrics_path": str(run_dir / "metrics.jsonl"),
               "final_checkpoint_path": str(run_dir / f"final-{int(common['recipe']['steps_per_arm'])}.pt"),
               "validation_rows_path": str(run_dir / "validation-rows.npz")}
        job_path = run_dir / "job.json"
        atomic_json(job_path, job)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)]
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"})
        proc, tee_thread = spawn_tee(command, env, run_dir / "worker.log")
        procs.append(proc)
        tee_threads.append(tee_thread)
        metadata.append({"gpu": gpu, "arm": arm, "seed": seed, "run_id": run_id,
                         "pid": proc.pid, "run_dir": str(run_dir)})
    atomic_json(wave_root / "wave-status.json", {"status": "running", "jobs": metadata})
    start = time.monotonic()
    telemetry_path = wave_root / "gpu-telemetry.jsonl"
    last_logged_steps: dict[str, int] = {}
    try:
        while any(proc.poll() is None for proc in procs):
            if time.monotonic() >= deadline:
                raise TimeoutError("shared worker deadline reached during final training")
            resource_gate(output)
            snapshots = gpu_sample()
            with telemetry_path.open("a") as stream:
                stream.write(json.dumps({"elapsed_seconds": round(time.monotonic() - common["started_monotonic"], 3),
                                         "gpus": snapshots}) + "\n")
            progress = []
            for item in metadata:
                path = Path(item["run_dir"]) / "progress.json"
                if path.exists():
                    progress.append(json.loads(path.read_text()))
            status = {"status": "running", "jobs": metadata, "progress": progress,
                      "last_gpu_telemetry": snapshots,
                      "elapsed_seconds": round(time.monotonic() - common["started_monotonic"], 2)}
            atomic_json(wave_root / "wave-status.json", status)
            print(json.dumps({"wave_progress": status}), flush=True)
            for row in progress:
                run_id = str(row["run_id"])
                if int(row["step"]) <= last_logged_steps.get(run_id, -1):
                    continue
                last_logged_steps[run_id] = int(row["step"])
                monitor.log(stage=f"train_{row['run_id']}", step=int(row["step"]), metrics={
                    "loss": float(row["loss"]),
                    "amp_skipped_updates": int(row["skipped_updates"]),
                    "amp_scale": float(row["grad_scaler_scale"]),
                    "gpu_count": len(snapshots),
                    "gpu_memory_used_mib_max": max(x["memory_used_mib"] for x in snapshots)})
            time.sleep(10.0)
        codes = [proc.wait() for proc in procs]
        for thread in tee_threads:
            thread.join(timeout=1)
        if any(code != 0 for code in codes):
            raise RuntimeError(f"final worker return codes: {codes}")
        results = [json.loads((Path(x["run_dir"]) / "run-result.json").read_text()) for x in metadata]
        record = {"status": "complete", "elapsed_seconds": round(time.monotonic() - start, 3),
                  "jobs": results, "telemetry_samples": len(telemetry_path.read_text().splitlines())}
        atomic_json(wave_root / "wave-result.json", record)
        return record
    except BaseException:
        stop_children(procs)
        raise


def run_test_export_wave(name: str, arms: tuple[str, str], seed: int, common: dict,
                         output: Path, deadline: float, test_indices_path: Path,
                         init_sha: str, monitor: LiveMonitor) -> dict:
    root = output / "test-export" / name
    root.mkdir(parents=True, exist_ok=False)
    procs, threads, records = [], [], []
    for gpu, arm in enumerate(arms):
        run_dir = root / f"{arm}_seed{seed}"
        run_dir.mkdir()
        job = {"mode": "test_export", "arm": arm, "seed": seed,
               "evaluation_seed": 20261025, "recipe": common["recipe"],
               "feature_root": common["feature_root"], "batch_size": common["recipe"]["global_batch_size"],
               "final_checkpoint_path": str(output / "final" / f"{arm}_seed{seed}" /
                                             f"final-{common['recipe']['steps_per_arm']}.pt"),
               "test_indices_path": str(test_indices_path),
               "expected_common_initial_state_sha256": init_sha,
               "rows_path": str(run_dir / "test-ranks.npz"),
               "result_path": str(run_dir / "test-result.json")}
        job_path = run_dir / "job.json"
        atomic_json(job_path, job)
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"})
        proc, thread = spawn_tee([sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)],
                                 env, run_dir / "worker.log")
        procs.append(proc); threads.append(thread)
        records.append({"gpu": gpu, "arm": arm, "run_dir": str(run_dir), "pid": proc.pid})
    atomic_json(root / "wave-status.json", {"status": "running", "jobs": records})
    try:
        while any(proc.poll() is None for proc in procs):
            if time.monotonic() >= deadline:
                raise TimeoutError("worker deadline reached during frozen full-test export")
            resource_gate(output)
            telemetry = gpu_sample()
            atomic_json(root / "wave-status.json", {"status": "running", "jobs": records,
                "last_gpu_telemetry": telemetry, "elapsed_seconds": round(time.monotonic() - common["started_monotonic"], 2)})
            time.sleep(5)
        codes = [proc.wait() for proc in procs]
        for thread in threads:
            thread.join(timeout=1)
        if any(code != 0 for code in codes):
            raise RuntimeError(f"frozen test export child return codes: {codes}")
        results = [json.loads((Path(row["run_dir"]) / "test-result.json").read_text()) for row in records]
        if any(row.get("test_event_count") != 232_049 or row.get("test_indices_sha256") != TEST_INDICES_SHA
               for row in results):
            raise RuntimeError("frozen test result endpoint/count mismatch")
        record = {"status": "complete", "results": results}
        atomic_json(root / "wave-result.json", record)
        return record
    except BaseException:
        stop_children(procs)
        raise


def supervise(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    worker_deadline = float(args.deadline_monotonic)
    status_path = output.parent / "matched-supervisor-status.json"
    owns_output = False
    monitor = None
    try:
        if output.exists() or args.stage.resolve().exists():
            raise FileExistsError("refusing preexisting recurrent-r2 output or staged-input directory")
        output.mkdir(parents=True)
        owns_output = True
        seed = int(config["recipe"]["seed"])
        active_arms = list(config["recipe"]["active_arms"])
        if active_arms != ["R1source", "R4source", "R4self", "R4untiedsource"]:
            raise RuntimeError(f"unexpected registered arms: {active_arms}")
        monitor = LiveMonitor(output_dir=output, run_id=f"huginn-recurrent-r2-seed{seed}",
                              project="recsys-tt2", entity="vaarsenii-hse-university",
                              config={"job_id": f"huginn-recurrent-r2-seed{seed}",
                                      "dataset_manifest_sha256": FEATURE_MANIFEST_SHA,
                                      "worker_sha256": sha256(Path(__file__).resolve()),
                                      "config_sha256": sha256(config_path), "arms": active_arms,
                                      "seed": seed, "accelerator": "Kaggle T4x2"},
                              enable_wandb=True, allow_sdk_install=False)
        inputs = validate_inputs(config_path, config, args.feature.resolve(), args.index_root.resolve(),
                                 args.aux.resolve(), output, args.stage.resolve(),
                                 args.expected_config_sha256)
        source_pins = {
            "scripts/run_huginn_item.py": sha256(Path(__file__).resolve()),
            "src/recsys_tt2/huginn_item_backbone.py": sha256(ROOT / "src/recsys_tt2/huginn_item_backbone.py"),
            "scripts/run_experiments.py": sha256(ROOT / "scripts/run_experiments.py"),
            "src/recsys_tt2/model.py": sha256(ROOT / "src/recsys_tt2/model.py"),
            "src/recsys_tt2/batching.py": sha256(ROOT / "src/recsys_tt2/batching.py"),
            "src/recsys_tt2/catalog.py": sha256(ROOT / "src/recsys_tt2/catalog.py"),
            "src/recsys_tt2/live_monitor.py": sha256(ROOT / "src/recsys_tt2/live_monitor.py"),
            f"configs/{config_path.name}": sha256(config_path)}
        if (source_pins["scripts/run_huginn_item.py"] != config["source_sha256"]
                or source_pins["scripts/run_huginn_item.py"] != args.expected_worker_sha256):
            raise RuntimeError("worker source SHA does not match reviewed config pin")
        inputs.update(cuda_device_preflight())
        recipe = dict(config["recipe"])
        recipe["evaluations"] = dict(recipe["evaluations"])
        recipe["evaluations"]["steps"] = [0]
        common = {"feature_root": inputs["staged_feature_root"],
                  "valid_indices_path": str(Path(inputs["staged_index_root"]) / "valid_indices.npy"),
                  "indices": {arm: str(Path(inputs["staged_index_root"]) / "full_train_indices.npy")
                              for arm in active_arms},
                  "recipe": recipe, "started_monotonic": started,
                  "profile_steps": int(config["pilot"]["profile_steps_per_arm"]),
                  "profile_warm_steps": int(config["pilot"]["profile_warm_steps"]),
                  "profile_child_timeout_seconds": int(config["pilot"]["profile_max_wall_seconds_per_worker"])}
        summary = {"schema": config["schema"], "status": "profiling",
                   "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "config_sha256": sha256(config_path), "source_pins": source_pins,
                   "inputs": inputs, "scope": config["scope"], "pilot_rule": config["pilot"],
                   "test_event_index_values_loaded": False, "test_rows_scored": False}
        atomic_json(output / "study-record.json", summary)
        atomic_json(status_path, {"status": "preflight_pass", "output": str(output),
                                  "devices": inputs["devices"], "source_pins": source_pins})
        pairs = [("R1source", "R4source"), ("R4self", "R4untiedsource")]
        profile_rows: dict[str, dict] = {}
        candidate_records = []
        for batch_size in config["pilot"]["batch_candidates"]:
            common["recipe"]["global_batch_size"] = int(batch_size)
            candidate_start = time.monotonic()
            candidate_profiles = {}
            candidate_failed = None
            for pair_index, pair in enumerate(pairs):
                phase = f"profile-b{batch_size}-pair{pair_index+1}"
                try:
                    phase_result = run_phase(phase,
                        {0: [(pair[0], seed, f"{pair[0]}_b{batch_size}")],
                         1: [(pair[1], seed, f"{pair[1]}_b{batch_size}")]},
                        common, output, worker_deadline)
                except Exception as exc:
                    candidate_failed = f"{type(exc).__name__}: {exc}"
                    break
                for row in phase_result["worker_results"]:
                    candidate_profiles[row["arm"]] = row
            feasible = candidate_failed is None and len(candidate_profiles) == 4
            if feasible:
                mem = max(int(row["cuda_peak_reserved_bytes"]) for row in candidate_profiles.values())
                # Leave 1 GiB headroom below T4's 16 GiB to absorb allocator variance.
                skipped = sum(int(row["skipped_updates"]) for row in candidate_profiles.values())
                feasible = mem < 15 * 1024**3 and skipped == 0
                if skipped:
                    candidate_failed = f"profile precision skipped {skipped} optimizer updates"
                for arm, row in candidate_profiles.items():
                    p75 = row["profile"]["p75_seconds_by_component"]["total_seconds"]
                    profile_rows[f"{batch_size}:{arm}"] = {"arm": arm, "batch_size": int(batch_size),
                        "p75_seconds_per_step": p75, "examples_per_second": batch_size / p75,
                        "peak_reserved_bytes": int(row["cuda_peak_reserved_bytes"]),
                        "skipped_updates": int(row["skipped_updates"]),
                        "grad_scaler_final_scale": row["grad_scaler_final_scale"]}
            candidate_records.append({"batch_size": int(batch_size), "feasible": feasible,
                                      "failure": candidate_failed,
                                      "elapsed_seconds": round(time.monotonic() - candidate_start, 2),
                                      "arm_profiles": [profile_rows.get(f"{batch_size}:{arm}") for arm in active_arms]})
            atomic_json(output / "profile-candidates.json", {"candidates": candidate_records,
                "selection_uses_validation": False, "all_arm_profiles": profile_rows})
        feasible_batches = [row["batch_size"] for row in candidate_records if row["feasible"]]
        if not feasible_batches:
            raise RuntimeError("no common batch size completed all four mode profiles")
        locked_batch = config["pilot"].get("locked_batch_size")
        if locked_batch is not None:
            if int(locked_batch) not in feasible_batches:
                raise RuntimeError("seed29 locked profile batch is infeasible on this T4 package")
            selected_batch = int(locked_batch)
        else:
            selected_batch = max(feasible_batches, key=lambda bs: min(
                profile_rows[f"{bs}:{arm}"]["examples_per_second"] for arm in active_arms))
        chosen_rates = {arm: profile_rows[f"{selected_batch}:{arm}"]["examples_per_second"] for arm in active_arms}
        selected_dose = 32_768_000
        steps = selected_dose // selected_batch
        if selected_dose % selected_batch:
            raise RuntimeError("locked full dose is not divisible by common batch")
        reserves = {"setup_per_kernel_seconds": float(config["pilot"]["stage_and_setup_reserve_seconds"]),
                    "evaluation_per_arm_seconds": float(config["pilot"]["evaluation_reserve_seconds"]),
                    "save_per_arm_seconds": float(config["pilot"]["final_checkpoint_save_reserve_seconds"])}
        safety = float(config["pilot"]["projection_safety_factor"])
        per_arm_seconds = {arm: steps / chosen_rates[arm] * safety for arm in active_arms}
        pair_seconds = {"R1source+R4source": max(per_arm_seconds["R1source"], per_arm_seconds["R4source"]),
                        "R4self+R4untiedsource": max(per_arm_seconds["R4self"], per_arm_seconds["R4untiedsource"])}
        per_kernel_worker = sum(pair_seconds.values()) + 4 * (reserves["evaluation_per_arm_seconds"] +
                                                                 reserves["save_per_arm_seconds"]) + reserves["setup_per_kernel_seconds"]
        projected_device_seconds = 2 * per_kernel_worker
        fits = (projected_device_seconds <= int(config["resources"]["device_seconds_cap"])
                and per_kernel_worker < int(config["resources"]["worker_deadline_seconds"]))
        projections = {"steps_per_arm": steps, "per_arm_training_seconds_safety_scaled": per_arm_seconds,
                       "paired_wave_training_seconds_safety_scaled": pair_seconds,
                       "worker_seconds_per_kernel": per_kernel_worker,
                       "projected_device_seconds_per_kernel": projected_device_seconds,
                       "device_seconds_cap": int(config["resources"]["device_seconds_cap"]),
                       "worker_deadline_seconds": int(config["resources"]["worker_deadline_seconds"])}
        if not fits:
            decision = {"status": "stop_after_profile_budget", "profile_candidates": candidate_records,
                        "selected_batch_size": selected_batch, "selected_precision": "fp16",
                        "selected_examples_per_arm": selected_dose, "selected_steps_per_arm": steps,
                        "selected_arm_examples_per_second": chosen_rates, "projection": projections,
                        "reserves": reserves, "projection_safety_factor": safety,
                        "selection_uses_validation": False, "training_started": False}
            atomic_json(output / "pilot-decision.json", decision)
            summary.update({"status": "stopped_after_profile_budget", "pilot_decision": decision})
            atomic_json(output / "study-record.json", summary)
            monitor.finish(exit_code=0)
            return 0
        steps = selected_dose // selected_batch
        recipe["global_batch_size"] = selected_batch
        recipe["steps_per_arm"] = steps
        recipe["steps"] = steps
        recipe["examples_per_arm"] = selected_dose
        recipe["evaluations"]["steps"] = [0, steps // 2, steps]
        recipe["evaluations"]["interval_steps"] = steps // 2
        common["recipe"] = recipe
        decision = {"status": "timing_only_locked_before_training", "profile_candidates": candidate_records,
                    "selected_batch_size": selected_batch, "selected_examples_per_arm": selected_dose,
                    "selected_steps_per_arm": steps, "selected_arm_examples_per_second": chosen_rates,
                    "dose_projections": projections, "reserves": reserves,
                    "projection_safety_factor": safety, "selection_uses_validation": False,
                    "deadline_utc": config["resources"]["deadline_utc"]}
        atomic_json(output / "pilot-decision.json", decision)
        summary.update({"status": "training", "pilot_decision": decision,
                       "primary_endpoint": "fixed final CE; validation NDCG@100 R4source minus R4self",
                       "arm_comparisons": {"primary": "R4source-R4self",
                           "secondary": ["R4source-R1source", "R4source-R4untiedsource"]}})
        atomic_json(output / "study-record.json", summary)
        monitor.log(stage="dose_locked", metrics={"batch_size": selected_batch,
            "examples_per_arm": selected_dose, "steps_per_arm": steps,
            "projected_device_seconds": projections["projected_device_seconds_per_kernel"]})
        waves = []
        for pair_index, pair in enumerate(pairs):
            waves.append(run_final_wave(f"seed{seed}-pair{pair_index+1}",
                [(0, pair[0], seed), (1, pair[1], seed)], common, output, worker_deadline, monitor))
        final_results = []
        for arm in active_arms:
            run_dir = output / "final" / f"{arm}_seed{seed}"
            result = json.loads((run_dir / "run-result.json").read_text())
            metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            if result["steps"] != steps or result["examples_seen"] != selected_dose:
                raise RuntimeError(f"final arm did not reach locked dose: {result}")
            if [int(x["step"]) for x in metrics] != recipe["evaluations"]["steps"]:
                raise RuntimeError(f"final validation schedule incomplete for {arm}/seed{seed}")
            final_results.append({**result, "metrics_sha256": sha256(run_dir / "metrics.jsonl")})
        if len({row["common_initial_state_sha256"] for row in final_results}) != 1:
            raise RuntimeError("final modes do not share common initialization")
        # Test indices are first opened only after dose and all eight final checkpoints are fixed.
        test_indices_path = args.feature.resolve() / "test_indices.npy"
        if sha256(test_indices_path) != TEST_INDICES_SHA:
            raise RuntimeError("frozen canonical full-test indices SHA mismatch")
        init_sha = final_results[0]["common_initial_state_sha256"]
        test_waves = [
            run_test_export_wave(f"seed{seed}-test-pair1", ("R1source", "R4source"), seed,
                                 common, output, worker_deadline, test_indices_path, init_sha, monitor),
            run_test_export_wave(f"seed{seed}-test-pair2", ("R4self", "R4untiedsource"), seed,
                                 common, output, worker_deadline, test_indices_path, init_sha, monitor)]
        row_by_arm = {row["arm"]: np.load(row["validation_rows_path"]) for row in final_results}
        reference = row_by_arm["R4self"]
        bootstraps = {}
        for arm in ("R4source", "R1source", "R4untiedsource"):
            for key in ("event_indices", "uid", "supported", "organic"):
                if not np.array_equal(reference[key], row_by_arm[arm][key]):
                    raise RuntimeError(f"paired validation rows differ in {key}: R4self vs {arm}")
            bootstraps[f"{arm}_minus_R4self"] = paired_uid_bootstrap(reference, row_by_arm[arm],
                                                                        seed=20261025 + seed)
        for value in row_by_arm.values():
            value.close()
        atomic_json(output / "paired-uid-bootstrap.json", bootstraps)
        skip_fractions = {f"{row['arm']}_seed{seed}": row["skipped_updates"] / max(row["steps"], 1)
                          for row in final_results}
        update_gaps = {f"{a}_minus_{b}": abs(next(x["optimizer_updates"] for x in final_results if x["arm"] == a)
                                               - next(x["optimizer_updates"] for x in final_results if x["arm"] == b))
                       for a, b in [("R4source", "R4self"), ("R4source", "R1source"), ("R4source", "R4untiedsource")]}
        comparable = all(x <= 0.005 for x in skip_fractions.values()) and all(x <= 32 for x in update_gaps.values())
        summary.update({"status": "complete" if comparable else "complete_comparability_failure",
                        "elapsed_seconds": round(time.monotonic() - started, 2), "final_arms": final_results,
                        "frozen_full_test": {"test_indices_sha256": TEST_INDICES_SHA,
                                             "test_event_count": 232049,
                                             "fixed_final_checkpoints_only": True,
                                             "test_waves": test_waves},
                        "comparability": {"status": "within_registered_amp_tolerance" if comparable else "comparability_failure",
                            "arm_skipped_update_fractions": skip_fractions, "paired_applied_update_gaps": update_gaps,
                            "maximum_arm_skip_fraction": 0.005, "maximum_paired_gap_updates": 32},
                        "paired_uid_bootstraps": bootstraps,
                        "test_event_index_values_loaded": True, "test_rows_scored": True})
        atomic_json(output / "study-record.json", summary)
        atomic_json(status_path, {"status": summary["status"], "output": str(output),
                                  "elapsed_seconds": summary["elapsed_seconds"], "final_arms": final_results})
        monitor.finish(exit_code=0)
        return 0
    except BaseException as exc:
        failure = {"status": "failed", "elapsed_seconds": round(time.monotonic() - started, 2),
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                   "test_event_index_values_loaded": False, "test_rows_scored": False}
        if owns_output:
            atomic_json(output / "failure.json", failure)
            atomic_json(status_path, failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        if monitor is not None:
            monitor.finish(exit_code=1)
        return 1

def preflight_only(args: argparse.Namespace) -> int:
    """Run the production CLI/import/data-staging checks without CUDA or model work."""
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = args.output.resolve()
    stage = args.stage.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or stage.exists():
        raise FileExistsError("refusing preexisting bootstrap output or staged-input directory")
    output.mkdir(parents=True)
    inputs = validate_inputs(config_path, config, args.feature.resolve(),
                             args.index_root.resolve(), args.aux.resolve(), output,
                             stage, args.expected_config_sha256)
    source_pins = {
        "scripts/run_huginn_item.py": sha256(Path(__file__).resolve()),
        "src/recsys_tt2/huginn_item_backbone.py": sha256(ROOT / "src/recsys_tt2/huginn_item_backbone.py"),
        "scripts/run_experiments.py": sha256(ROOT / "scripts/run_experiments.py"),
        "src/recsys_tt2/model.py": sha256(ROOT / "src/recsys_tt2/model.py"),
        "src/recsys_tt2/batching.py": sha256(ROOT / "src/recsys_tt2/batching.py"),
        "src/recsys_tt2/catalog.py": sha256(ROOT / "src/recsys_tt2/catalog.py"),
        "src/recsys_tt2/live_monitor.py": sha256(ROOT / "src/recsys_tt2/live_monitor.py"),
        f"configs/{config_path.name}": sha256(config_path),
    }
    if (source_pins["scripts/run_huginn_item.py"] != config["source_sha256"]
            or source_pins["scripts/run_huginn_item.py"] != args.expected_worker_sha256):
        raise RuntimeError("worker source SHA does not match the reviewed config pin")
    model_preflight = cpu_model_preflight(
        Path(inputs["staged_feature_root"]),
        Path(inputs["staged_index_root"]) / "full_train_indices.npy",
        config["recipe"], int(config["recipe"]["seed"]))
    record = {"status": "preflight_complete", "gpu_work_started": False,
              "model_training_started": False, "cuda_device_preflight_called": False,
              "source_pins": source_pins, "inputs": inputs,
              "actual_model_cpu_preflight": model_preflight,
              "seed": int(config["recipe"]["seed"]),
              "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(output / "preflight-result.json", record)
    print("WORKER_PREFLIGHT_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--feature", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--aux", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--deadline-monotonic", type=float)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-worker-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker_train(argparse.Namespace(job_config=args.worker))
    required = (args.config, args.feature, args.index_root, args.aux, args.output, args.stage,
                args.deadline_monotonic, args.expected_config_sha256,
                args.expected_worker_sha256)
    if any(x is None for x in required):
        parser.error("supervisor mode requires config, feature, index-root, aux, output, and deadline")
    if args.preflight_only:
        return preflight_only(args)
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
