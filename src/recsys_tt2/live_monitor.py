"""Low-overhead aggregate progress logging for remote training workers.

This module is intended for one coordinator process per job. It writes safe,
aggregate JSONL records locally and to flushed stdout, and optionally mirrors
them to W&B when an API key file and the W&B SDK are available. It never logs
the credential or per-user/per-example data.
"""

from __future__ import annotations

import json
import importlib
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


DEFAULT_WANDB_KEY_FILE = Path(
    "/kaggle/input/datasets/arsen4ikvar/"
    "recsys-private-wandb-credential-20261020/wandb_api_key.txt"
)
_SAFE_METRICS = {
    "step",
    "steps_total",
    "epoch",
    "loss",
    "train_loss",
    "valid_loss",
    "valid_ndcg_at_100",
    "valid_recall_at_100",
    "valid_organic_ndcg_at_100",
    "valid_recommended_ndcg_at_100",
    "examples_per_second",
    "seconds_per_step",
    "gpu_utilization_pct_mean",
    "gpu_memory_used_mib_max",
    "gpu_power_w_mean",
    "gpu_count",
    "stage_elapsed_seconds",
    "rank",
    "host_cpu_count",
    "host_memory_available_mib",
    "process_cpu_seconds",
    "data_fetch_seconds",
    "h2d_seconds",
    "forward_seconds",
    "backward_seconds",
    "optimizer_seconds",
    "amp_skipped_updates",
    "amp_scale",
}
_SAFE_CONFIG = {
    "job_id",
    "stage",
    "seed",
    "arm",
    "dataset_id",
    "dataset_manifest_sha256",
    "source_sha256",
    "checkpoint_sha256",
    "parent_checkpoint_sha256",
    "config_sha256",
    "optimizer_steps",
    "batch_size",
    "anchor_count",
    "validation_sample_count",
    "accelerator",
    "visible_device_count",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_wandb_with_bounded_install(
    timeout_seconds: int = 90, *, allow_install: bool = True
) -> Any:
    try:
        return importlib.import_module("wandb")
    except ImportError:
        if not allow_install:
            raise
        # Kaggle images may omit the SDK. Keep package bootstrap bounded and
        # quiet; a failed optional install must not hold up training.
        installer_env = os.environ.copy()
        installer_env.pop("WANDB_API_KEY", None)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--retries",
                "0",
                "--timeout",
                "8",
                "wandb",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=installer_env,
            check=True,
            timeout=timeout_seconds,
        )
        importlib.invalidate_caches()
        return importlib.import_module("wandb")


def _safe_scalar(value: Any) -> int | float | str | bool | None:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _safe_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (config or {}).items():
        if key not in _SAFE_CONFIG:
            continue
        if key.endswith("sha256") and isinstance(value, str) and not _SHA256.fullmatch(value):
            continue
        safe = _safe_scalar(value)
        if safe is not None:
            result[key] = safe
    return result


class LiveMonitor:
    """Emit aggregate progress to stdout/JSONL and optionally W&B.

    The output file must be new. The caller should instantiate this once in
    the parent worker and report coarse, aggregate GPU and training metrics.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path,
        run_id: str,
        project: str,
        entity: str | None = None,
        config: Mapping[str, Any] | None = None,
        key_file: str | Path | None = None,
        enable_wandb: bool = True,
        allow_sdk_install: bool = True,
        sdk_install_timeout_seconds: int = 90,
    ) -> None:
        self.run_id = str(run_id)
        self.started = time.monotonic()
        self.path = Path(output_dir) / "live_progress.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", buffering=1)
        self._wandb = None
        self._wandb_run = None
        self._wandb_status = "disabled"
        self._emit({"event": "monitor_start", "run_id": self.run_id})
        self._emit({"event": "run_config", **_safe_config(config)})

        if not enable_wandb:
            self._emit({"event": "wandb_status", "status": "disabled_by_config"})
            return

        credential_path = Path(key_file) if key_file else DEFAULT_WANDB_KEY_FILE
        try:
            # Resolve/install the optional SDK before touching credentials so
            # the installer subprocess cannot inherit the W&B API key.
            wandb = _load_wandb_with_bounded_install(
                timeout_seconds=sdk_install_timeout_seconds,
                allow_install=allow_sdk_install,
            )
            key = credential_path.read_text(encoding="utf-8").strip()
            if not key:
                raise ValueError("empty credential file")
            previous_key = os.environ.get("WANDB_API_KEY")
            os.environ["WANDB_API_KEY"] = key

            settings = wandb.Settings(init_timeout=15, save_code=False)
            self._wandb = wandb
            self._wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=self.run_id,
                id=self.run_id,
                resume="never",
                config=_safe_config(config),
                settings=settings,
                mode="online",
                save_code=False,
            )
            self._wandb_status = "online"
            self._emit({"event": "wandb_status", "status": self._wandb_status})
        except Exception as exc:  # W&B is an optional mirror, never a training dependency.
            self._wandb = None
            self._wandb_run = None
            self._wandb_status = "unavailable"
            self._emit({
                "event": "wandb_status",
                "status": self._wandb_status,
                "reason_type": type(exc).__name__,
            })
        finally:
            # Remove the injected credential after SDK initialization.
            if "previous_key" in locals():
                if previous_key is None:
                    os.environ.pop("WANDB_API_KEY", None)
                else:
                    os.environ["WANDB_API_KEY"] = previous_key
            if "key" in locals():
                del key

    def _emit(self, record: Mapping[str, Any]) -> None:
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if key == "config":
                continue
            if key == "metrics" and isinstance(value, Mapping):
                for metric, raw in value.items():
                    if metric not in _SAFE_METRICS:
                        continue
                    safe_metric = _safe_scalar(raw)
                    if safe_metric is not None:
                        clean[metric] = safe_metric
                continue
            safe = _safe_scalar(value)
            if safe is not None:
                clean[str(key)] = safe
        clean.setdefault("elapsed_seconds", time.monotonic() - self.started)
        line = json.dumps(clean, separators=(",", ":"), allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def log(self, *, stage: str, metrics: Mapping[str, Any], step: int | None = None) -> None:
        safe_metrics = {
            key: value
            for key, raw in metrics.items()
            if key in _SAFE_METRICS and (value := _safe_scalar(raw)) is not None
        }
        record: dict[str, Any] = {
            "event": "progress",
            "run_id": self.run_id,
            "stage": stage,
            "metrics": safe_metrics,
        }
        if step is not None:
            record["step"] = int(step)
        self._emit(record)
        if self._wandb_run is not None:
            try:
                self._wandb_run.log({f"{stage}/{key}": val for key, val in safe_metrics.items()}, step=step)
            except Exception as exc:
                self._wandb_run = None
                self._wandb = None
                self._wandb_status = "unavailable"
                self._emit({
                    "event": "wandb_status",
                    "status": self._wandb_status,
                    "reason_type": type(exc).__name__,
                })

    def finish(self, *, exit_code: int = 0) -> None:
        self._emit({"event": "monitor_finish", "run_id": self.run_id, "exit_code": int(exit_code)})
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish(exit_code=int(exit_code))
            except Exception:
                self._emit({"event": "wandb_status", "status": "finish_failed"})
        self._stream.close()
