"""English FastSpeech 2 Live Monitoring, Persistent Logging & Checkpoint Verification.

Provides Google Colab and local training utilities for:
    - Part A.4: Dataset, manifest, feature path, NaN/Inf, and duration-mel consistency validation
    - Part A.5: Non-destructive model & loss smoke testing with gradient finiteness verification
    - Part C: Verified checkpoint saving (post-save load check before pruning, strict best-loss guard)
    - Part D.1-D.4: Live Colab HTML + 5-Graph Matplotlib dashboard and GPU/System telemetry
    - Part D.5: Persistent CSV, JSON status, and text logging on Google Drive with resume deduplication
    - Part E: Failure detection (NaN/Inf loss/gradients, OOM, missing files, interruption tracking)
    - Part F: End-of-training structured summary report and high-resolution plot export
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common.training.checkpoint import CheckpointManager
from common.training.engine import Trainer, TrainLosses
from common.utils.tools import to_device


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_seconds(seconds: float | None) -> str:
    """Format seconds into HH:MM:SS string."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ==============================================================================
# 1. GPU AND SYSTEM MONITORING (PART D.4)
# ==============================================================================


@dataclass
class SystemMetrics:
    """Container for GPU and host system telemetry."""

    cuda_available: bool = False
    device_type: str = "cpu"
    gpu_name: str = "N/A (CPU/MPS)"
    gpu_mem_allocated_gb: float = 0.0
    gpu_mem_reserved_gb: float = 0.0
    gpu_mem_total_gb: float = 0.0
    gpu_utilization_pct: float | None = None
    cpu_ram_used_gb: float = 0.0
    cpu_ram_total_gb: float = 0.0
    cpu_ram_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SystemMonitor:
    """Fault-tolerant GPU and system resource monitor.

    Never raises an exception; gracefully handles missing tools or non-CUDA environments.
    """

    @staticmethod
    def collect(device: torch.device | str | None = None) -> SystemMetrics:
        metrics = SystemMetrics()
        try:
            metrics.cuda_available = bool(torch.cuda.is_available())
            if device is not None:
                dev = torch.device(device)
                metrics.device_type = dev.type
            else:
                metrics.device_type = "cuda" if metrics.cuda_available else "cpu"

            if metrics.cuda_available:
                idx = 0
                if isinstance(device, torch.device) and device.index is not None:
                    idx = device.index
                metrics.gpu_name = str(torch.cuda.get_device_name(idx))
                metrics.gpu_mem_allocated_gb = float(
                    torch.cuda.memory_allocated(idx) / (1024**3)
                )
                metrics.gpu_mem_reserved_gb = float(
                    torch.cuda.memory_reserved(idx) / (1024**3)
                )
                props = torch.cuda.get_device_properties(idx)
                metrics.gpu_mem_total_gb = float(props.total_memory / (1024**3))

                # Query GPU utilization via nvidia-smi if available
                try:
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                            f"--id={idx}",
                        ],
                        timeout=1.5,
                        stderr=subprocess.DEVNULL,
                    )
                    val_str = out.decode("utf-8").strip().splitlines()[0].strip()
                    metrics.gpu_utilization_pct = float(val_str)
                except Exception:
                    metrics.gpu_utilization_pct = None
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if metrics.device_type == "mps":
                    metrics.gpu_name = "Apple Silicon MPS"
        except Exception:
            pass

        # Collect CPU RAM telemetry via psutil if available, with os/resource fallback
        try:
            import psutil

            vm = psutil.virtual_memory()
            metrics.cpu_ram_used_gb = float((vm.total - vm.available) / (1024**3))
            metrics.cpu_ram_total_gb = float(vm.total / (1024**3))
            metrics.cpu_ram_pct = float(vm.percent)
        except Exception:
            try:
                import resource
                import sys

                rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # On macOS ru_maxrss is bytes; on Linux it is kilobytes
                divisor = (1024**3) if sys.platform == "darwin" else (1024**2)
                metrics.cpu_ram_used_gb = float(rusage / divisor)
            except Exception:
                pass

        return metrics


# ==============================================================================
# 2. DATASET & MODEL SMOKE VALIDATION (PART A.4 & A.5)
# ==============================================================================


def validate_svarah_dataset(
    preprocessed_path: str | Path,
    model_config: Mapping[str, Any],
    train_loader: Any,
    val_loader: Any,
    expected_vocab_size: int = 360,
    max_file_checks: int | None = None,
    train_manifest: str | Path | None = None,
    val_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Validate manifests, feature files, vocabulary, speakers, and batch tensors.

    Checks:
        - train.txt and val.txt existence and non-empty line counts
        - speakers.json and stats.json structure and finiteness
        - Feature file existence (.npy) and sample consistency (mel_len == pitch_len == energy_len == sum(duration))
        - Real batch loading from train_loader and val_loader
        - Zero NaN/Inf in batch tensors and exact shape alignment
    """
    prep_dir = Path(preprocessed_path).resolve()
    t_total_start = time.perf_counter()

    # Stage 1: Metadata, JSONs, vocabulary, and manifest reading
    t_meta_start = time.perf_counter()
    if not prep_dir.exists():
        raise FileNotFoundError(f"Preprocessed data directory not found: {prep_dir}")

    train_txt = (
        Path(train_manifest).resolve()
        if train_manifest is not None
        else (prep_dir / "train.txt")
    )
    val_txt = (
        Path(val_manifest).resolve()
        if val_manifest is not None
        else (prep_dir / "val.txt")
    )
    speakers_json = prep_dir / "speakers.json"
    stats_json = prep_dir / "stats.json"

    for req_file in (train_txt, val_txt, speakers_json, stats_json):
        if not req_file.exists():
            raise FileNotFoundError(f"Required dataset file missing: {req_file}")

    with open(speakers_json, encoding="utf-8") as f:
        speakers = json.load(f)
    with open(stats_json, encoding="utf-8") as f:
        stats = json.load(f)

    if not speakers or not isinstance(speakers, Mapping):
        raise ValueError("speakers.json is empty or invalid.")

    for key in ("pitch", "energy"):
        if key not in stats or len(stats[key]) != 4:
            raise ValueError(f"stats.json missing valid 4-element '{key}' stats.")
        if not all(math.isfinite(float(x)) for x in stats[key]):
            raise ValueError(f"Non-finite values in stats.json['{key}']: {stats[key]}")

    # Check vocabulary size
    try:
        from English.text.symbols import symbols
    except ImportError:
        from text.symbols import symbols

    actual_vocab_size = len(symbols)
    cfg_vocab_size = int(model_config.get("n_src_vocab", expected_vocab_size))
    if actual_vocab_size != cfg_vocab_size:
        raise ValueError(
            f"Vocabulary size mismatch: len(symbols)={actual_vocab_size} vs model_config.n_src_vocab={cfg_vocab_size}"
        )

    manifest_lines: dict[str, list[str]] = {}
    manifest_counts: dict[str, int] = {}
    for split_name, txt_path in (("train", train_txt), ("val", val_txt)):
        with open(txt_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Manifest {txt_path} is empty.")
        manifest_lines[split_name] = lines
        manifest_counts[split_name] = len(lines)
    metadata_and_manifests_sec = time.perf_counter() - t_meta_start

    # Stage 2: Single directory scan per feature folder (avoids 15,488 individual stat() calls on Drive FUSE)
    t_dir_scan_start = time.perf_counter()
    mel_dir = prep_dir / "mel"
    pitch_dir = prep_dir / "pitch"
    energy_dir = prep_dir / "energy"
    dur_dir = prep_dir / "duration"

    mel_entries = set(os.listdir(mel_dir)) if mel_dir.is_dir() else set()
    pitch_entries = set(os.listdir(pitch_dir)) if pitch_dir.is_dir() else set()
    energy_entries = set(os.listdir(energy_dir)) if energy_dir.is_dir() else set()
    dur_entries = set(os.listdir(dur_dir)) if dur_dir.is_dir() else set()
    directory_index_scan_sec = time.perf_counter() - t_dir_scan_start

    # Stage 3 & 4: Feature file existence checks (all 4 features per entry) & deep array checks
    feature_existence_checks_sec = 0.0
    deep_array_checks_sec = 0.0
    checked_arrays = 0
    checked_feature_files = 0

    for split_name, txt_path in (("train", train_txt), ("val", val_txt)):
        lines = manifest_lines[split_name]
        check_subset = lines if max_file_checks is None else lines[:max_file_checks]
        for idx, line in enumerate(check_subset):
            t_exist_start = time.perf_counter()
            parts = line.split("|")
            if len(parts) < 4:
                raise ValueError(f"Malformed line {idx} in {txt_path}: {line}")
            basename, spk = parts[0], parts[1]
            if spk not in speakers:
                raise KeyError(f"Speaker '{spk}' in {txt_path} not found in speakers.json")

            mel_name = f"{spk}-mel-{basename}.npy"
            pitch_name = f"{spk}-pitch-{basename}.npy"
            energy_name = f"{spk}-energy-{basename}.npy"
            dur_name = f"{spk}-duration-{basename}.npy"

            for subdir_path, entry_set, fname in (
                (mel_dir, mel_entries, mel_name),
                (pitch_dir, pitch_entries, pitch_name),
                (energy_dir, energy_entries, energy_name),
                (dur_dir, dur_entries, dur_name),
            ):
                if fname not in entry_set:
                    fpath = subdir_path / fname
                    if not fpath.exists():
                        raise FileNotFoundError(f"Missing feature file: {fpath}")
                checked_feature_files += 1
            feature_existence_checks_sec += time.perf_counter() - t_exist_start

            # Deep array consistency check on first 15 items per split
            if idx < 15:
                t_deep_start = time.perf_counter()
                mel_f = mel_dir / mel_name
                pitch_f = pitch_dir / pitch_name
                energy_f = energy_dir / energy_name
                dur_f = dur_dir / dur_name

                mel_arr = np.load(mel_f)
                pitch_arr = np.load(pitch_f)
                energy_arr = np.load(energy_f)
                dur_arr = np.load(dur_f)

                if not (
                    np.isfinite(mel_arr).all()
                    and np.isfinite(pitch_arr).all()
                    and np.isfinite(energy_arr).all()
                    and np.isfinite(dur_arr).all()
                ):
                    raise ValueError(f"NaN/Inf detected in numpy arrays for {basename}")

                mel_frames = int(mel_arr.shape[0])
                dur_sum = int(dur_arr.sum())
                if mel_frames != dur_sum:
                    raise ValueError(
                        f"Duration sum ({dur_sum}) != mel frames ({mel_frames}) for {basename}"
                    )
                if pitch_arr.shape[0] != mel_frames or energy_arr.shape[0] != mel_frames:
                    raise ValueError(
                        f"Frame-level pitch/energy length mismatch with mel ({mel_frames}) for {basename}"
                    )
                checked_arrays += 1
                deep_array_checks_sec += time.perf_counter() - t_deep_start

    # Stage 5: Inspect one real training batch and one validation batch
    def _inspect_batch(loader: Any, label: str) -> tuple[dict[str, Any], float, float]:
        t_init_start = time.perf_counter()
        loader_iter = iter(loader)
        iter_init_sec = time.perf_counter() - t_init_start

        t_iter_start = time.perf_counter()
        item = next(loader_iter)
        batch = (
            item[0]
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], (tuple, list))
            else item
        )
        (
            ids,
            raw_texts,
            spks,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = batch

        for name, arr in (
            ("speakers", spks),
            ("texts", texts),
            ("src_lens", src_lens),
            ("mels", mels),
            ("mel_lens", mel_lens),
            ("pitches", pitches),
            ("energies", energies),
            ("durations", durations),
        ):
            t = torch.as_tensor(arr)
            if not torch.isfinite(t).all():
                raise ValueError(f"Non-finite values in {label} batch tensor '{name}'")

        b_size = int(texts.shape[0])
        if texts.shape != (b_size, int(max_src_len)):
            raise ValueError(f"Invalid {label} texts shape: {texts.shape}")
        if durations.shape != (b_size, int(max_src_len)):
            raise ValueError(f"Invalid {label} durations shape: {durations.shape}")
        if mels.shape != (b_size, int(max_mel_len), 80):
            raise ValueError(f"Invalid {label} mels shape: {mels.shape}")
        if pitches.shape != (b_size, int(max_mel_len)):
            raise ValueError(f"Invalid {label} pitches shape: {pitches.shape}")
        if energies.shape != (b_size, int(max_mel_len)):
            raise ValueError(f"Invalid {label} energies shape: {energies.shape}")

        # Check duration sum == mel_lens per sample in batch
        dur_sums = np.sum(durations, axis=1)
        if not np.array_equal(dur_sums, mel_lens):
            raise ValueError(
                f"{label} batch duration sums {dur_sums} do not match mel_lens {mel_lens}"
            )
        first_batch_sec = time.perf_counter() - t_iter_start

        return (
            {
                "batch_size": b_size,
                "max_src_len": int(max_src_len),
                "max_mel_len": int(max_mel_len),
                "texts_shape": list(texts.shape),
                "mels_shape": list(mels.shape),
            },
            iter_init_sec,
            first_batch_sec,
        )

    train_batch_info, train_iter_init_sec, train_first_batch_sec = _inspect_batch(
        train_loader, "train"
    )
    val_batch_info, val_iter_init_sec, val_first_batch_sec = _inspect_batch(
        val_loader, "val"
    )
    total_validation_sec = time.perf_counter() - t_total_start

    timings_sec = {
        "metadata_and_manifests_sec": round(metadata_and_manifests_sec, 4),
        "directory_index_scan_sec": round(directory_index_scan_sec, 4),
        "feature_existence_checks_sec": round(feature_existence_checks_sec, 4),
        "deep_array_checks_sec": round(deep_array_checks_sec, 4),
        "train_dataloader_iter_init_sec": round(train_iter_init_sec, 4),
        "train_dataloader_first_batch_sec": round(train_first_batch_sec, 4),
        "train_dataloader_total_sec": round(train_iter_init_sec + train_first_batch_sec, 4),
        "val_dataloader_iter_init_sec": round(val_iter_init_sec, 4),
        "val_dataloader_first_batch_sec": round(val_first_batch_sec, 4),
        "val_dataloader_total_sec": round(val_iter_init_sec + val_first_batch_sec, 4),
        "total_validation_sec": round(total_validation_sec, 4),
    }

    print(
        "[Dataset Validation Timings] "
        f"manifests={timings_sec['metadata_and_manifests_sec']:.3f}s | "
        f"dir_scan={timings_sec['directory_index_scan_sec']:.3f}s | "
        f"existence({checked_feature_files} files)={timings_sec['feature_existence_checks_sec']:.3f}s | "
        f"deep_arrays({checked_arrays} utts)={timings_sec['deep_array_checks_sec']:.3f}s | "
        f"train_loader={timings_sec['train_dataloader_total_sec']:.3f}s | "
        f"val_loader={timings_sec['val_dataloader_total_sec']:.3f}s | "
        f"total={timings_sec['total_validation_sec']:.3f}s"
    )

    return {
        "status": "PASSED",
        "train_utterances": manifest_counts["train"],
        "val_utterances": manifest_counts["val"],
        "num_speakers": len(speakers),
        "vocab_size": actual_vocab_size,
        "checked_feature_files": checked_feature_files,
        "deep_checked_arrays": checked_arrays,
        "train_batch": train_batch_info,
        "val_batch": val_batch_info,
        "timings_sec": timings_sec,
    }


def run_model_smoke_test(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: Any,
    val_loader: Any,
    device: torch.device | str,
) -> dict[str, Any]:
    """Run a non-destructive model and loss smoke test without stepping the optimizer.

    Verifies:
        - Forward pass on a real training batch on target device
        - Multi-task loss computation and finiteness of all 6 loss components
        - Backward pass gradient computation and finiteness across parameters
        - Gradients are cleared afterwards so model weights and optimizer remain untouched
        - Validation forward pass in eval mode
    """
    dev = torch.device(device)
    model = model.to(dev)
    loss_fn = loss_fn.to(dev)

    # 1. Training forward + backward test (no optimizer step)
    model.train()
    model.zero_grad(set_to_none=True)

    train_item = next(iter(train_loader))
    train_batch = (
        train_item[0]
        if isinstance(train_item, list)
        and len(train_item) > 0
        and isinstance(train_item[0], (tuple, list))
        else train_item
    )
    train_batch_dev = to_device(train_batch, dev)

    outputs = model(*(train_batch_dev[2:]))
    losses = loss_fn(train_batch_dev, outputs)
    loss_vals = [float(l.item()) for l in losses]

    if not all(math.isfinite(v) for v in loss_vals):
        raise FloatingPointError(f"Non-finite smoke test training loss: {loss_vals}")

    # Backward to verify gradients are finite, then clear gradients immediately
    losses[0].backward()
    grad_norms: list[float] = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                model.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradient detected in parameter '{name}' during smoke test."
                )
            grad_norms.append(float(param.grad.norm().item()))
    model.zero_grad(set_to_none=True)

    # 2. Validation forward test
    model.eval()
    val_item = next(iter(val_loader))
    val_batch = (
        val_item[0]
        if isinstance(val_item, list)
        and len(val_item) > 0
        and isinstance(val_item[0], (tuple, list))
        else val_item
    )
    val_batch_dev = to_device(val_batch, dev)
    with torch.no_grad():
        val_outputs = model(*(val_batch_dev[2:]))
        val_losses = loss_fn(val_batch_dev, val_outputs)
        val_loss_vals = [float(l.item()) for l in val_losses]

    if not all(math.isfinite(v) for v in val_loss_vals):
        raise FloatingPointError(f"Non-finite smoke test validation loss: {val_loss_vals}")

    return {
        "status": "PASSED",
        "device": str(dev),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_params_with_grad": len(grad_norms),
        "train_losses": {
            "total": loss_vals[0],
            "mel": loss_vals[1],
            "postnet_mel": loss_vals[2],
            "pitch": loss_vals[3],
            "energy": loss_vals[4],
            "duration": loss_vals[5],
        },
        "val_losses": {
            "total": val_loss_vals[0],
            "mel": val_loss_vals[1],
            "postnet_mel": val_loss_vals[2],
            "pitch": val_loss_vals[3],
            "energy": val_loss_vals[4],
            "duration": val_loss_vals[5],
        },
    }


# ==============================================================================
# 3. PERSISTENT LOGGING & HISTORY DEDUPLICATION (PART D.5 & PART E)
# ==============================================================================


class PersistentLogger:
    """Manages persistent logs on Google Drive / disk with safe resume deduplication.

    Directory structure under log_dir:
        - training_history.csv
        - validation_history.csv
        - training_status.json
        - training.log
        - plots/
    """

    TRAIN_CSV_FIELDS = [
        "timestamp",
        "epoch",
        "global_step",
        "total_loss",
        "mel_loss",
        "postnet_mel_loss",
        "pitch_loss",
        "energy_loss",
        "duration_loss",
        "learning_rate",
        "grad_norm",
        "steps_per_sec",
    ]

    VAL_CSV_FIELDS = [
        "timestamp",
        "epoch",
        "global_step",
        "total_loss",
        "mel_loss",
        "postnet_mel_loss",
        "pitch_loss",
        "energy_loss",
        "duration_loss",
        "best_val_loss",
        "improved",
        "checkpoint_path",
    ]

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.log_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.train_csv_path = self.log_dir / "training_history.csv"
        self.val_csv_path = self.log_dir / "validation_history.csv"
        self.status_json_path = self.log_dir / "training_status.json"
        self.text_log_path = self.log_dir / "training.log"

        self.train_records: list[dict[str, Any]] = []
        self.val_records: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.best_val_step: int = 0
        self.best_val_loss: float = float("inf")

        self._ensure_csv_headers()

    def _ensure_csv_headers(self) -> None:
        if not self.train_csv_path.exists():
            with open(self.train_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.TRAIN_CSV_FIELDS)
                writer.writeheader()
        if not self.val_csv_path.exists():
            with open(self.val_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.VAL_CSV_FIELDS)
                writer.writeheader()

    def log_message(self, message: str, level: str = "INFO") -> None:
        """Append a timestamped line to training.log."""
        ts = _utc_now_iso()
        line = f"[{ts}] [{level}] {message}\n"
        try:
            with open(self.text_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def record_warning(
        self,
        step: int,
        epoch: int,
        category: str,
        message: str,
    ) -> None:
        """Record a warning event to memory, training.log, and status JSON."""
        entry = {
            "timestamp": _utc_now_iso(),
            "step": int(step),
            "epoch": int(epoch),
            "category": category,
            "message": message,
        }
        self.warnings.append(entry)
        self.log_message(f"({category}) step={step}: {message}", level="WARNING")

    def restore_and_deduplicate(self, max_valid_step: int) -> None:
        """Load historical CSV records up to max_valid_step and deduplicate by global_step.

        Prevents duplicate global_step entries if resuming from an earlier checkpoint
        after an interrupted session.
        """
        # 1. Restore training_history.csv
        train_by_step: dict[int, dict[str, Any]] = {}
        if self.train_csv_path.exists():
            try:
                with open(self.train_csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        step = int(float(row["global_step"]))
                        if step <= max_valid_step:
                            train_by_step[step] = {
                                "timestamp": row.get("timestamp", ""),
                                "epoch": int(float(row.get("epoch", 1))),
                                "global_step": step,
                                "total_loss": float(row["total_loss"]),
                                "mel_loss": float(row["mel_loss"]),
                                "postnet_mel_loss": float(row["postnet_mel_loss"]),
                                "pitch_loss": float(row["pitch_loss"]),
                                "energy_loss": float(row["energy_loss"]),
                                "duration_loss": float(row["duration_loss"]),
                                "learning_rate": float(row.get("learning_rate") or 0.0),
                                "grad_norm": float(row["grad_norm"])
                                if row.get("grad_norm") not in (None, "", "None")
                                else 0.0,
                                "steps_per_sec": float(row.get("steps_per_sec") or 0.0),
                            }
            except Exception as e:
                self.log_message(f"Failed reading training_history.csv on restore: {e}", "WARNING")

        self.train_records = [train_by_step[s] for s in sorted(train_by_step.keys())]
        self._rewrite_csv_atomic(self.train_csv_path, self.TRAIN_CSV_FIELDS, self.train_records)

        # 2. Restore validation_history.csv
        val_by_step: dict[int, dict[str, Any]] = {}
        if self.val_csv_path.exists():
            try:
                with open(self.val_csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        step = int(float(row["global_step"]))
                        if step <= max_valid_step:
                            val_by_step[step] = {
                                "timestamp": row.get("timestamp", ""),
                                "epoch": int(float(row.get("epoch", 1))),
                                "global_step": step,
                                "total_loss": float(row["total_loss"]),
                                "mel_loss": float(row["mel_loss"]),
                                "postnet_mel_loss": float(row["postnet_mel_loss"]),
                                "pitch_loss": float(row["pitch_loss"]),
                                "energy_loss": float(row["energy_loss"]),
                                "duration_loss": float(row["duration_loss"]),
                                "best_val_loss": float(row["best_val_loss"]),
                                "improved": str(row.get("improved", "False")).lower() == "true",
                                "checkpoint_path": row.get("checkpoint_path", ""),
                            }
            except Exception as e:
                self.log_message(f"Failed reading validation_history.csv on restore: {e}", "WARNING")

        self.val_records = [val_by_step[s] for s in sorted(val_by_step.keys())]
        self._rewrite_csv_atomic(self.val_csv_path, self.VAL_CSV_FIELDS, self.val_records)

        # Recompute best_val_loss and best_val_step from restored records
        for rec in self.val_records:
            if rec["total_loss"] < self.best_val_loss:
                self.best_val_loss = rec["total_loss"]
                self.best_val_step = rec["global_step"]

        self.log_message(
            f"Restored and deduplicated history up to step {max_valid_step}: "
            f"{len(self.train_records)} train entries, {len(self.val_records)} val entries."
        )

    @staticmethod
    def _rewrite_csv_atomic(
        csv_path: Path,
        fieldnames: Sequence[str],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        tmp_path = csv_path.with_suffix(".csv.tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        os.replace(tmp_path, csv_path)

    def log_train_step(
        self,
        epoch: int,
        global_step: int,
        losses: TrainLosses,
        steps_per_sec: float,
    ) -> dict[str, Any]:
        """Record a training step to memory and append to training_history.csv."""
        row = {
            "timestamp": _utc_now_iso(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "total_loss": float(losses.total),
            "mel_loss": float(losses.mel),
            "postnet_mel_loss": float(losses.postnet_mel),
            "pitch_loss": float(losses.pitch),
            "energy_loss": float(losses.energy),
            "duration_loss": float(losses.duration),
            "learning_rate": float(losses.lr or 0.0),
            "grad_norm": float(losses.grad_norm or 0.0),
            "steps_per_sec": float(steps_per_sec),
        }

        # Deduplicate in case same step is logged twice
        if self.train_records and self.train_records[-1]["global_step"] == global_step:
            self.train_records[-1] = row
            self._rewrite_csv_atomic(
                self.train_csv_path, self.TRAIN_CSV_FIELDS, self.train_records
            )
        else:
            self.train_records.append(row)
            with open(self.train_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.TRAIN_CSV_FIELDS)
                writer.writerow(row)

        return row

    def log_validation_step(
        self,
        epoch: int,
        global_step: int,
        val_losses: TrainLosses,
        best_val_loss: float,
        improved: bool,
        checkpoint_path: str | Path = "",
    ) -> dict[str, Any]:
        """Record a validation step to memory and append to validation_history.csv."""
        if improved or val_losses.total <= self.best_val_loss:
            self.best_val_loss = float(best_val_loss)
            self.best_val_step = int(global_step)

        row = {
            "timestamp": _utc_now_iso(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "total_loss": float(val_losses.total),
            "mel_loss": float(val_losses.mel),
            "postnet_mel_loss": float(val_losses.postnet_mel),
            "pitch_loss": float(val_losses.pitch),
            "energy_loss": float(val_losses.energy),
            "duration_loss": float(val_losses.duration),
            "best_val_loss": float(best_val_loss),
            "improved": bool(improved),
            "checkpoint_path": str(checkpoint_path),
        }

        if self.val_records and self.val_records[-1]["global_step"] == global_step:
            self.val_records[-1] = row
            self._rewrite_csv_atomic(
                self.val_csv_path, self.VAL_CSV_FIELDS, self.val_records
            )
        else:
            self.val_records.append(row)
            with open(self.val_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.VAL_CSV_FIELDS)
                writer.writerow(row)

        self.log_message(
            f"Validation at step {global_step} (epoch {epoch}): "
            f"total={val_losses.total:.4f} (mel={val_losses.mel:.4f}, postnet={val_losses.postnet_mel:.4f}, "
            f"pitch={val_losses.pitch:.4f}, energy={val_losses.energy:.4f}, dur={val_losses.duration:.4f}) | "
            f"best={best_val_loss:.4f} | improved={improved}"
        )
        return row

    def update_status(
        self,
        status: str,
        epoch: int,
        global_step: int,
        total_step: int,
        best_val_loss: float,
        latest_checkpoint: str | Path | None = None,
        best_checkpoint: str | Path | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically write training_status.json."""
        payload: dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "status": status,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "total_step": int(total_step),
            "progress_pct": round(100.0 * global_step / max(total_step, 1), 2),
            "best_val_loss": float(best_val_loss) if math.isfinite(best_val_loss) else None,
            "best_val_step": int(self.best_val_step),
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
            "num_train_logs": len(self.train_records),
            "num_val_logs": len(self.val_records),
            "warnings_count": len(self.warnings),
            "recent_warnings": self.warnings[-5:],
        }
        if extra:
            payload.update(extra)

        tmp_file = self.status_json_path.with_suffix(".json.tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_file, self.status_json_path)
        except Exception:
            pass


# ==============================================================================
# 4. VERIFIED CHECKPOINT CONTROLLER (PART C)
# ==============================================================================


def verify_checkpoint_file(ckpt_path: str | Path) -> dict[str, Any]:
    """Verify that a saved checkpoint file exists, is non-empty, and loads cleanly.

    Args:
        ckpt_path: Path to the saved .pth.tar checkpoint file.

    Returns:
        Summary metadata of the verified checkpoint.

    Raises:
        FileNotFoundError: If file does not exist.
        RuntimeError: If file is empty, corrupted, or missing required state keys.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved checkpoint file does not exist: {path}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise RuntimeError(f"Saved checkpoint file is empty (0 bytes): {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = (
        "model",
        "optimizer",
        "epoch",
        "global_step",
        "best_val_loss",
        "history",
        "rng_state",
    )
    for k in required_keys:
        if k not in ckpt:
            raise RuntimeError(f"Verified checkpoint {path} is missing required key: '{k}'")

    if not isinstance(ckpt["optimizer"], Mapping) or "current_step" not in ckpt["optimizer"]:
        raise RuntimeError(f"Verified checkpoint {path} has incomplete optimizer state.")

    return {
        "path": str(path),
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "global_step": int(ckpt["global_step"]),
        "epoch": int(ckpt["epoch"]),
        "best_val_loss": float(ckpt["best_val_loss"]),
    }


def save_and_verify_checkpoint(
    trainer: Trainer,
    is_best: bool = False,
    max_to_keep: int = 5,
    extra_metadata: dict[str, Any] | None = None,
    logger: PersistentLogger | None = None,
) -> Path:
    """Atomically save checkpoint, verify load integrity, and then prune older checkpoints.

    Guarantees:
        1. New step checkpoint and latest checkpoint are written atomically (.tmp -> os.replace).
        2. Best checkpoint is ONLY updated when is_best is strictly True.
        3. Saved checkpoint is verified on disk before any older checkpoint is pruned.
        4. At most `max_to_keep` periodic step checkpoints are retained.
    """
    if trainer.checkpoint_manager is None:
        raise RuntimeError("Trainer has no CheckpointManager configured.")

    mgr = trainer.checkpoint_manager
    # Temporarily disable automatic pruning inside mgr.save so we can verify BEFORE pruning
    orig_max_to_keep = mgr.max_to_keep
    mgr.max_to_keep = None
    try:
        saved_step_path = trainer.save_checkpoint(
            is_best=is_best,
            extra_metadata=extra_metadata,
        )
        # Verify step checkpoint and latest checkpoint
        verify_checkpoint_file(saved_step_path)
        verify_checkpoint_file(mgr.latest_path)
        if is_best:
            verify_checkpoint_file(mgr.best_path)

        # Now safely prune older step checkpoints
        if max_to_keep is not None and max_to_keep > 0:
            step_ckpts = mgr.find_all_step_checkpoints()
            if len(step_ckpts) > max_to_keep:
                for _, old_path in step_ckpts[:-max_to_keep]:
                    try:
                        old_path.unlink()
                    except OSError:
                        pass

        if logger is not None:
            logger.log_message(
                f"Verified checkpoint save at step {trainer.global_step} "
                f"(is_best={is_best}): {saved_step_path.name}"
            )
        return saved_step_path
    except Exception as e:
        if logger is not None:
            logger.record_warning(
                step=trainer.global_step,
                epoch=trainer.epoch,
                category="CHECKPOINT_SAVE_ERROR",
                message=f"Checkpoint save or verification failed: {e}",
            )
        raise
    finally:
        mgr.max_to_keep = orig_max_to_keep


# ==============================================================================
# 5. LIVE DASHBOARD & 5-GRAPH GENERATOR (PART D.1 - D.4)
# ==============================================================================


class LiveDashboard:
    """Generates and updates the 5 required training/validation plots and live Colab view.

    Graphs generated:
        - Graph 1: Training total loss vs global step
        - Graph 2: Validation total loss vs global step
        - Graph 3: Individual training loss components (Mel, PostNet Mel, Pitch, Energy, Duration)
        - Graph 4: Individual validation loss components (Mel, PostNet Mel, Pitch, Energy, Duration)
        - Graph 5: Learning rate vs global step
    """

    def __init__(
        self,
        logger: PersistentLogger,
        total_steps: int,
        total_epochs: int,
        enable_notebook_display: bool = True,
    ) -> None:
        self.logger = logger
        self.total_steps = max(int(total_steps), 1)
        self.total_epochs = max(int(total_epochs), 1)
        self.enable_notebook_display = enable_notebook_display
        self.dashboard_png_path = self.logger.plots_dir / "loss_curves_dashboard.png"
        self.graph_paths = {
            "dashboard": self.dashboard_png_path,
            "graph1_train_total_loss": self.logger.plots_dir / "graph1_train_total_loss.png",
            "graph2_val_total_loss": self.logger.plots_dir / "graph2_val_total_loss.png",
            "graph3_train_components": self.logger.plots_dir / "graph3_train_loss_components.png",
            "graph4_val_components": self.logger.plots_dir / "graph4_val_loss_components.png",
            "graph5_learning_rate": self.logger.plots_dir / "graph5_learning_rate.png",
        }

    def render_plots(self) -> dict[str, Path]:
        """Render the 5 required graphs (both combined dashboard and individual files)."""
        train_recs = self.logger.train_records
        val_recs = self.logger.val_records

        t_steps = [r["global_step"] for r in train_recs]
        v_steps = [r["global_step"] for r in val_recs]

        # Combined 5-panel figure (3x2 grid with panel 6 showing summary info)
        fig, axes = plt.subplots(3, 2, figsize=(15, 13), dpi=110)
        fig.suptitle(
            "FastSpeech 2 (English Svarah) — Live Training & Validation Monitoring",
            fontsize=14,
            fontweight="bold",
        )

        # Graph 1: Training Total Loss vs Global Step
        ax1 = axes[0, 0]
        if t_steps:
            ax1.plot(
                t_steps,
                [r["total_loss"] for r in train_recs],
                color="#1f77b4",
                linewidth=1.5,
                label="Train Total Loss",
            )
            ax1.legend(loc="upper right")
        ax1.set_title("Graph 1: Training Total Loss vs. Global Step")
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Total Loss")
        ax1.grid(True, linestyle="--", alpha=0.4)

        # Graph 2: Validation Total Loss vs Global Step
        ax2 = axes[0, 1]
        if v_steps:
            v_totals = [r["total_loss"] for r in val_recs]
            ax2.plot(
                v_steps,
                v_totals,
                color="#d62728",
                marker="o",
                markersize=4,
                linewidth=1.6,
                label="Val Total Loss",
            )
            best_idx = int(np.argmin(v_totals))
            ax2.scatter(
                [v_steps[best_idx]],
                [v_totals[best_idx]],
                color="#2ca02c",
                s=90,
                zorder=5,
                label=f"Best ({v_totals[best_idx]:.4f} @ step {v_steps[best_idx]})",
            )
            ax2.legend(loc="upper right")
        ax2.set_title("Graph 2: Validation Total Loss vs. Global Step")
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Total Loss")
        ax2.grid(True, linestyle="--", alpha=0.4)

        # Graph 3: Individual Training Loss Components
        ax3 = axes[1, 0]
        if t_steps:
            ax3.plot(t_steps, [r["mel_loss"] for r in train_recs], label="Mel", linewidth=1.2)
            ax3.plot(
                t_steps,
                [r["postnet_mel_loss"] for r in train_recs],
                label="PostNet Mel",
                linewidth=1.2,
            )
            ax3.plot(t_steps, [r["pitch_loss"] for r in train_recs], label="Pitch", linewidth=1.2)
            ax3.plot(
                t_steps, [r["energy_loss"] for r in train_recs], label="Energy", linewidth=1.2
            )
            ax3.plot(
                t_steps, [r["duration_loss"] for r in train_recs], label="Duration", linewidth=1.2
            )
            ax3.legend(loc="upper right", fontsize=8, ncol=2)
        ax3.set_title("Graph 3: Individual Training Loss Components")
        ax3.set_xlabel("Global Step")
        ax3.set_ylabel("Component Loss")
        ax3.grid(True, linestyle="--", alpha=0.4)

        # Graph 4: Individual Validation Loss Components
        ax4 = axes[1, 1]
        if v_steps:
            ax4.plot(
                v_steps,
                [r["mel_loss"] for r in val_recs],
                marker=".",
                label="Val Mel",
                linewidth=1.2,
            )
            ax4.plot(
                v_steps,
                [r["postnet_mel_loss"] for r in val_recs],
                marker=".",
                label="Val PostNet Mel",
                linewidth=1.2,
            )
            ax4.plot(
                v_steps,
                [r["pitch_loss"] for r in val_recs],
                marker=".",
                label="Val Pitch",
                linewidth=1.2,
            )
            ax4.plot(
                v_steps,
                [r["energy_loss"] for r in val_recs],
                marker=".",
                label="Val Energy",
                linewidth=1.2,
            )
            ax4.plot(
                v_steps,
                [r["duration_loss"] for r in val_recs],
                marker=".",
                label="Val Duration",
                linewidth=1.2,
            )
            ax4.legend(loc="upper right", fontsize=8, ncol=2)
        ax4.set_title("Graph 4: Individual Validation Loss Components")
        ax4.set_xlabel("Global Step")
        ax4.set_ylabel("Component Loss")
        ax4.grid(True, linestyle="--", alpha=0.4)

        # Graph 5: Learning Rate vs Global Step
        ax5 = axes[2, 0]
        if t_steps:
            ax5.plot(
                t_steps,
                [r["learning_rate"] for r in train_recs],
                color="#9467bd",
                linewidth=1.5,
                label="Learning Rate (Noam)",
            )
            ax5.legend(loc="upper right")
        ax5.set_title("Graph 5: Learning Rate vs. Global Step")
        ax5.set_xlabel("Global Step")
        ax5.set_ylabel("Learning Rate")
        ax5.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax5.grid(True, linestyle="--", alpha=0.4)

        # Panel 6: Status & Telemetry Summary Box
        ax6 = axes[2, 1]
        ax6.axis("off")
        latest_t = train_recs[-1] if train_recs else None
        latest_v = val_recs[-1] if val_recs else None
        summary_lines = [
            f"Recorded Train Points : {len(train_recs)}",
            f"Recorded Val Points   : {len(val_recs)}",
            f"Latest Step           : {latest_t['global_step'] if latest_t else 0} / {self.total_steps}",
            f"Latest Train Loss     : {latest_t['total_loss']:.4f}" if latest_t else "Latest Train Loss     : N/A",
            f"Latest Val Loss       : {latest_v['total_loss']:.4f}" if latest_v else "Latest Val Loss       : N/A",
            f"Best Val Loss         : {self.logger.best_val_loss:.4f} (step {self.logger.best_val_step})"
            if math.isfinite(self.logger.best_val_loss)
            else "Best Val Loss         : N/A",
            f"Active Warnings       : {len(self.logger.warnings)}",
            f"Last Updated (UTC)    : {_utc_now_iso()}",
        ]
        ax6.text(
            0.05,
            0.5,
            "\n".join(summary_lines),
            fontsize=11,
            family="monospace",
            verticalalignment="center",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8f9fa", "edgecolor": "#ced4da"},
        )

        fig.tight_layout(rect=[0, 0.01, 1, 0.96])
        fig.savefig(self.dashboard_png_path)
        plt.close(fig)

        # Also save individual standalone plots for Graph 1..5
        self._save_single_graphs(train_recs, val_recs)
        return dict(self.graph_paths)

    def _save_single_graphs(
        self,
        train_recs: Sequence[Mapping[str, Any]],
        val_recs: Sequence[Mapping[str, Any]],
    ) -> None:
        t_steps = [r["global_step"] for r in train_recs]
        v_steps = [r["global_step"] for r in val_recs]

        # Graph 1
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        if t_steps:
            ax.plot(t_steps, [r["total_loss"] for r in train_recs], color="#1f77b4", lw=1.5)
        ax.set_title("Graph 1: Training Total Loss vs. Global Step")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Total Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(self.graph_paths["graph1_train_total_loss"])
        plt.close(fig)

        # Graph 2
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        if v_steps:
            ax.plot(
                v_steps,
                [r["total_loss"] for r in val_recs],
                color="#d62728",
                marker="o",
                lw=1.5,
            )
        ax.set_title("Graph 2: Validation Total Loss vs. Global Step")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Total Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(self.graph_paths["graph2_val_total_loss"])
        plt.close(fig)

        # Graph 3
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        if t_steps:
            for key, label in (
                ("mel_loss", "Mel"),
                ("postnet_mel_loss", "PostNet Mel"),
                ("pitch_loss", "Pitch"),
                ("energy_loss", "Energy"),
                ("duration_loss", "Duration"),
            ):
                ax.plot(t_steps, [r[key] for r in train_recs], label=label, lw=1.2)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Graph 3: Individual Training Loss Components")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(self.graph_paths["graph3_train_components"])
        plt.close(fig)

        # Graph 4
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        if v_steps:
            for key, label in (
                ("mel_loss", "Val Mel"),
                ("postnet_mel_loss", "Val PostNet Mel"),
                ("pitch_loss", "Val Pitch"),
                ("energy_loss", "Val Energy"),
                ("duration_loss", "Val Duration"),
            ):
                ax.plot(v_steps, [r[key] for r in val_recs], marker=".", label=label, lw=1.2)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Graph 4: Individual Validation Loss Components")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(self.graph_paths["graph4_val_components"])
        plt.close(fig)

        # Graph 5
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        if t_steps:
            ax.plot(t_steps, [r["learning_rate"] for r in train_recs], color="#9467bd", lw=1.5)
        ax.set_title("Graph 5: Learning Rate vs. Global Step")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(self.graph_paths["graph5_learning_rate"])
        plt.close(fig)

    def build_html_panel(
        self,
        epoch: int,
        global_step: int,
        train_losses: TrainLosses,
        elapsed_sec: float,
        steps_per_sec: float,
        sys_metrics: SystemMetrics,
        mode_banner: str = "TRAINING",
    ) -> str:
        """Build rich HTML status table for inline display in Google Colab."""
        pct = min(100.0, max(0.0, 100.0 * global_step / self.total_steps))
        rem_steps = max(0, self.total_steps - global_step)
        eta_sec = rem_steps / steps_per_sec if steps_per_sec > 1e-6 else None

        latest_val = self.logger.val_records[-1] if self.logger.val_records else None
        if latest_val is not None:
            val_badge = (
                "<span style='color:#155724;background:#d4edda;padding:2px 6px;border-radius:4px;font-weight:bold;'>IMPROVED (Best)</span>"
                if latest_val["improved"]
                else "<span style='color:#856404;background:#fff3cd;padding:2px 6px;border-radius:4px;'>No improvement</span>"
            )
            val_row_html = f"""
            <tr>
                <td><b>Validation (Step {latest_val['global_step']})</b></td>
                <td><b>{latest_val['total_loss']:.4f}</b></td>
                <td>{latest_val['mel_loss']:.4f}</td>
                <td>{latest_val['postnet_mel_loss']:.4f}</td>
                <td>{latest_val['pitch_loss']:.4f}</td>
                <td>{latest_val['energy_loss']:.4f}</td>
                <td>{latest_val['duration_loss']:.4f}</td>
                <td>Best: <b>{latest_val['best_val_loss']:.4f}</b> (step {self.logger.best_val_step}) {val_badge}</td>
            </tr>
            """
        else:
            val_row_html = """
            <tr>
                <td><b>Validation</b></td>
                <td colspan="7"><i>No validation run recorded yet (runs every val_step)</i></td>
            </tr>
            """

        gpu_util_str = (
            f"{sys_metrics.gpu_utilization_pct:.0f}%"
            if sys_metrics.gpu_utilization_pct is not None
            else "N/A"
        )
        warn_html = ""
        if self.logger.warnings:
            last_w = self.logger.warnings[-1]
            warn_html = (
                f"<div style='margin-top:8px;padding:6px 10px;background:#f8d7da;color:#721c24;"
                f"border:1px solid #f5c6cb;border-radius:4px;font-size:12px;'>"
                f"<b>Recent Warning [{last_w['category']}] (step {last_w['step']}):</b> {last_w['message']}</div>"
            )

        return f"""
        <div style="font-family:sans-serif;border:1px solid #dee2e6;border-radius:8px;padding:12px;background:#ffffff;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <h3 style="margin:0;color:#212529;">FastSpeech 2 Live Training Dashboard — {mode_banner}</h3>
                <span style="font-size:13px;color:#495057;">
                    Epoch: <b>{epoch} / {self.total_epochs}</b> &nbsp;|&nbsp;
                    Step: <b>{global_step:,} / {self.total_steps:,}</b> ({pct:.2f}%)
                </span>
            </div>
            <div style="width:100%;background:#e9ecef;border-radius:6px;height:16px;overflow:hidden;margin-bottom:10px;">
                <div style="width:{pct:.2f}%;background:#0d6efd;height:100%;"></div>
            </div>
            <div style="display:flex;gap:18px;font-size:13px;margin-bottom:10px;color:#343a40;flex-wrap:wrap;">
                <div><b>Learning Rate:</b> { (train_losses.lr or 0.0):.6e}</div>
                <div><b>Throughput:</b> {steps_per_sec:.2f} batches/s</div>
                <div><b>Elapsed:</b> {_format_seconds(elapsed_sec)}</div>
                <div><b>ETA:</b> {_format_seconds(eta_sec)}</div>
                <div><b>GPU:</b> {sys_metrics.gpu_name} (CUDA: {sys_metrics.cuda_available})</div>
                <div><b>VRAM:</b> {sys_metrics.gpu_mem_allocated_gb:.2f}GB alloc / {sys_metrics.gpu_mem_reserved_gb:.2f}GB res / {sys_metrics.gpu_mem_total_gb:.2f}GB ({gpu_util_str} util)</div>
                <div><b>System RAM:</b> {sys_metrics.cpu_ram_used_gb:.2f} / {sys_metrics.cpu_ram_total_gb:.2f} GB ({sys_metrics.cpu_ram_pct:.1f}%)</div>
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:13px;text-align:left;" border="1" bordercolor="#dee2e6" cellpadding="6">
                <thead style="background:#f1f3f5;">
                    <tr>
                        <th>Split / Stage</th>
                        <th>Total Loss</th>
                        <th>Mel Loss</th>
                        <th>PostNet Mel</th>
                        <th>Pitch Loss</th>
                        <th>Energy Loss</th>
                        <th>Duration Loss</th>
                        <th>Notes / Best Validation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><b>Training (Step {global_step})</b></td>
                        <td><b>{train_losses.total:.4f}</b></td>
                        <td>{train_losses.mel:.4f}</td>
                        <td>{train_losses.postnet_mel:.4f}</td>
                        <td>{train_losses.pitch:.4f}</td>
                        <td>{train_losses.energy:.4f}</td>
                        <td>{train_losses.duration:.4f}</td>
                        <td>Grad Norm: {(train_losses.grad_norm or 0.0):.4f}</td>
                    </tr>
                    {val_row_html}
                </tbody>
            </table>
            {warn_html}
        </div>
        """

    def update(
        self,
        epoch: int,
        global_step: int,
        train_losses: TrainLosses,
        elapsed_sec: float,
        steps_per_sec: float,
        device: torch.device | str | None = None,
        mode_banner: str = "TRAINING",
    ) -> None:
        """Render plots and update notebook display without interrupting training on error."""
        try:
            sys_metrics = SystemMonitor.collect(device)
            self.render_plots()
            html_str = self.build_html_panel(
                epoch=epoch,
                global_step=global_step,
                train_losses=train_losses,
                elapsed_sec=elapsed_sec,
                steps_per_sec=steps_per_sec,
                sys_metrics=sys_metrics,
                mode_banner=mode_banner,
            )

            if self.enable_notebook_display:
                try:
                    from IPython.display import HTML, Image, clear_output, display

                    clear_output(wait=True)
                    display(HTML(html_str))
                    if self.dashboard_png_path.exists():
                        display(Image(filename=str(self.dashboard_png_path)))
                except Exception:
                    pass
        except Exception as e:
            self.logger.log_message(f"Dashboard update non-fatal error: {e}", level="WARNING")


# ==============================================================================
# 6. TRAINING SUMMARY GENERATOR (PART F)
# ==============================================================================


def generate_training_summary(
    logger: PersistentLogger,
    checkpoint_manager: CheckpointManager,
    dashboard: LiveDashboard,
    total_duration_sec: float,
    completed_normally: bool,
    final_epoch: int,
    final_step: int,
    final_lr: float,
) -> dict[str, Any]:
    """Generate and save end-of-training summary JSON and Markdown report on Google Drive."""
    latest_train = logger.train_records[-1] if logger.train_records else None
    latest_val = logger.val_records[-1] if logger.val_records else None

    step_ckpts = [str(p) for _, p in checkpoint_manager.find_all_step_checkpoints()]
    summary: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "training_completed_normally": bool(completed_normally),
        "status": "COMPLETED" if completed_normally else "INTERRUPTED_OR_INCOMPLETE",
        "total_epochs": int(final_epoch),
        "total_optimizer_steps": int(final_step),
        "total_training_duration_sec": round(float(total_duration_sec), 2),
        "total_training_duration_hms": _format_seconds(total_duration_sec),
        "final_learning_rate": float(final_lr),
        "final_training_loss": {
            "total": latest_train["total_loss"],
            "mel": latest_train["mel_loss"],
            "postnet_mel": latest_train["postnet_mel_loss"],
            "pitch": latest_train["pitch_loss"],
            "energy": latest_train["energy_loss"],
            "duration": latest_train["duration_loss"],
        }
        if latest_train
        else None,
        "final_validation_loss": {
            "total": latest_val["total_loss"],
            "mel": latest_val["mel_loss"],
            "postnet_mel": latest_val["postnet_mel_loss"],
            "pitch": latest_val["pitch_loss"],
            "energy": latest_val["energy_loss"],
            "duration": latest_val["duration_loss"],
        }
        if latest_val
        else None,
        "best_validation_loss": float(logger.best_val_loss)
        if math.isfinite(logger.best_val_loss)
        else None,
        "best_validation_step": int(logger.best_val_step),
        "checkpoints": {
            "checkpoint_dir": str(checkpoint_manager.checkpoint_dir),
            "latest_checkpoint": str(checkpoint_manager.latest_path)
            if checkpoint_manager.latest_path.exists()
            else None,
            "best_checkpoint": str(checkpoint_manager.best_path)
            if checkpoint_manager.best_path.exists()
            else None,
            "retained_periodic_checkpoints": step_ckpts,
        },
        "loss_graphs": {k: str(v) for k, v in dashboard.graph_paths.items()},
        "warnings_count": len(logger.warnings),
        "warnings": logger.warnings,
    }

    summary_json_path = logger.log_dir / "training_summary.json"
    summary_md_path = logger.log_dir / "training_summary.md"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# English FastSpeech 2 (Svarah) — Training Run Summary",
        "",
        f"- **Status**: `{summary['status']}`",
        f"- **Generated (UTC)**: `{summary['generated_at_utc']}`",
        f"- **Total Epochs**: `{summary['total_epochs']}`",
        f"- **Total Optimizer Steps**: `{summary['total_optimizer_steps']:,}`",
        f"- **Total Training Duration**: `{summary['total_training_duration_hms']}` (`{summary['total_training_duration_sec']}s`)",
        f"- **Final Learning Rate**: `{summary['final_learning_rate']:.6e}`",
        f"- **Best Validation Loss**: `{summary['best_validation_loss']}` (at step `{summary['best_validation_step']}`)",
        "",
        "## Final Losses",
        f"- **Final Training Loss**: `{json.dumps(summary['final_training_loss'])}`",
        f"- **Final Validation Loss**: `{json.dumps(summary['final_validation_loss'])}`",
        "",
        "## Checkpoints Saved on Drive",
        f"- **Latest Checkpoint**: `{summary['checkpoints']['latest_checkpoint']}`",
        f"- **Best Checkpoint**: `{summary['checkpoints']['best_checkpoint']}`",
        f"- **Retained Periodic Checkpoints ({len(step_ckpts)})**:",
    ]
    for cp in step_ckpts:
        md_lines.append(f"  - `{cp}`")

    md_lines.append("")
    md_lines.append("## Saved Loss Graphs")
    for k, v in summary["loss_graphs"].items():
        md_lines.append(f"- **{k}**: `{v}`")

    md_lines.append("")
    md_lines.append(f"## Warnings / Anomalies ({len(logger.warnings)})")
    if logger.warnings:
        for w in logger.warnings:
            md_lines.append(
                f"- `[{w['timestamp']}]` Step {w['step']} (`{w['category']}`): {w['message']}"
            )
    else:
        md_lines.append("- None recorded.")

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return summary


__all__ = [
    "SystemMetrics",
    "SystemMonitor",
    "validate_svarah_dataset",
    "run_model_smoke_test",
    "PersistentLogger",
    "verify_checkpoint_file",
    "save_and_verify_checkpoint",
    "LiveDashboard",
    "generate_training_summary",
]
