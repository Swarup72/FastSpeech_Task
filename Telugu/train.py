"""Telugu FastSpeech 2 Training Runner for IndicTTS Dataset (Colab & CLI).

Integrates:
    - IndicTTS multi-speaker dataset loading (2 balanced speakers: spk_000, spk_001)
    - FastSpeech 2 architecture with verified Telugu grapheme vocabulary (VOCAB_SIZE=101)
    - ScheduledOptim with Transformer Noam learning rate schedule and linear warmup
    - CheckpointManager + atomic save_and_verify_checkpoint (max_to_keep=5)
    - PersistentLogger (training_history.csv, validation_history.csv, training_status.json, training.log)
    - LiveDashboard (HTML progress dashboard + 5 live Matplotlib loss/LR curves)
    - Failure detection (NaN/Inf loss/grad, CUDA OOM, interruption recovery)
    - Strict test-split isolation: test.txt and test features are NEVER accessed or read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add workspace root and original_fastspeech2 to sys.path
_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.config import FastSpeech2Config
from common.model.fastspeech2 import FastSpeech2
from common.model.loss import FastSpeech2Loss
from common.model.optimizer import ScheduledOptim
from common.paths import (
    PROJECT_ROOT,
    ensure_output_dir,
    get_config_path,
    resolve_project_path,
    setup_project_paths,
)

setup_project_paths()

from common.training.checkpoint import CheckpointManager
from common.training.engine import Trainer, _resolve_device
from dataset import Dataset
from English.monitoring import (
    LiveDashboard,
    PersistentLogger,
    SystemMonitor,
    generate_training_summary,
    run_model_smoke_test,
    save_and_verify_checkpoint,
    verify_checkpoint_file,
)


def _resolve_file_path(path_like: str | Path) -> Path:
    """Resolve an input file path, checking direct path first then project-root-relative."""
    p = Path(path_like).expanduser()
    if p.is_file() or p.is_absolute():
        return p
    proj_p = resolve_project_path(p)
    return proj_p if proj_p.is_file() else p


def validate_telugu_dataset(preprocessed_path: Path) -> dict[str, Any]:
    """Validate Telugu IndicTTS preprocessed dataset directory before training.

    Guarantees:
        1. Exactly 2,300 training manifest entries and 421 validation entries.
        2. Four-column manifest format (`uid|speaker|tokens|raw_text`).
        3. Exact 2-speaker mapping: `spk_000: 0` (female), `spk_001: 1` (male).
        4. Valid normalization statistics in `stats.json`.
        5. All 6 feature directories exist and contain all 16,326 required .npy arrays.
        6. Strictly NEVER opens, checks, or accesses `test.txt` or test features.

    Returns:
        Dictionary summarizing dataset validation metrics.
    """
    prep_dir = Path(preprocessed_path)
    if not prep_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed directory not found: {prep_dir}")

    train_txt = prep_dir / "train.txt"
    val_txt = prep_dir / "val.txt"
    speakers_json = prep_dir / "speakers.json"
    stats_json = prep_dir / "stats.json"

    for req_file in (train_txt, val_txt, speakers_json, stats_json):
        if not req_file.is_file():
            raise FileNotFoundError(f"Missing required metadata file: {req_file}")

    # 1. Manifest line counts and format
    with open(train_txt, encoding="utf-8") as f:
        train_lines = [line.strip() for line in f if line.strip()]
    with open(val_txt, encoding="utf-8") as f:
        val_lines = [line.strip() for line in f if line.strip()]

    if len(train_lines) != 2300:
        raise ValueError(
            f"Expected exactly 2,300 train manifest lines, got {len(train_lines)}"
        )
    if len(val_lines) != 421:
        raise ValueError(
            f"Expected exactly 421 val manifest lines, got {len(val_lines)}"
        )

    # 2. Speaker mapping
    with open(speakers_json, encoding="utf-8") as f:
        speakers = json.load(f)
    expected_speakers = {"spk_000": 0, "spk_001": 1}
    if speakers != expected_speakers:
        raise ValueError(
            f"Expected speaker mapping {expected_speakers}, got {speakers}"
        )

    # 3. Normalization statistics
    with open(stats_json, encoding="utf-8") as f:
        stats = json.load(f)
    for key in ("pitch", "energy"):
        if key not in stats or len(stats[key]) < 4:
            raise ValueError(f"Invalid or incomplete '{key}' statistics in stats.json")
        min_v, max_v, mean_v, std_v = stats[key][:4]
        if std_v <= 0.0 or math.isnan(std_v):
            raise ValueError(f"Invalid standard deviation ({std_v}) for '{key}' in stats.json")

    # 4. Feature file existence across all active utterances
    feature_dirs = [
        "mel",
        "pitch",
        "energy",
        "duration",
        "pitch_phoneme",
        "energy_phoneme",
    ]
    for d in feature_dirs:
        dp = prep_dir / d
        if not dp.is_dir():
            raise FileNotFoundError(f"Missing feature directory: {dp}")

    all_manifest_entries: list[tuple[str, str]] = []
    for line_idx, line in enumerate(train_lines + val_lines):
        parts = line.split("|")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid manifest line format (expected 4 pipe-delimited columns) at index {line_idx}: {line[:60]}..."
            )
        uid, spk, tokens, raw_text = parts
        all_manifest_entries.append((uid, spk))

    missing_count = 0
    missing_samples: list[str] = []
    for uid, spk in all_manifest_entries:
        for d in feature_dirs:
            feat_name = (
                "pitch"
                if d == "pitch_phoneme"
                else ("energy" if d == "energy_phoneme" else d)
            )
            fp = prep_dir / d / f"{spk}-{feat_name}-{uid}.npy"
            if not fp.exists():
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(str(fp))

    if missing_count > 0:
        raise FileNotFoundError(
            f"Detected {missing_count} missing feature files. First examples: {missing_samples}"
        )

    # 5. Spot-check 10 feature arrays for NaN/Inf values and dimensional consistency
    spot_check_uids = all_manifest_entries[:5] + all_manifest_entries[-5:]
    for uid, spk in spot_check_uids:
        mel = np.load(prep_dir / "mel" / f"{spk}-mel-{uid}.npy")
        pitch = np.load(prep_dir / "pitch" / f"{spk}-pitch-{uid}.npy")
        energy = np.load(prep_dir / "energy" / f"{spk}-energy-{uid}.npy")
        dur = np.load(prep_dir / "duration" / f"{spk}-duration-{uid}.npy")

        if not np.isfinite(mel).all() or not np.isfinite(pitch).all() or not np.isfinite(energy).all():
            raise FloatingPointError(f"Detected non-finite values in features for utterance {uid}")
        if int(np.sum(dur)) != mel.shape[0]:
            raise ValueError(
                f"Duration sum ({int(np.sum(dur))}) != mel frames ({mel.shape[0]}) for utterance {uid}"
            )

    return {
        "status": "VALID",
        "train_utterances": len(train_lines),
        "val_utterances": len(val_lines),
        "speakers": list(speakers.keys()),
        "total_features_verified": len(all_manifest_entries) * len(feature_dirs),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FastSpeech 2 on Telugu IndicTTS Dataset"
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "IndicTTS" / "preprocess.yaml"),
        help="Path to preprocess.yaml",
    )
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "IndicTTS" / "model.yaml"),
        help="Path to model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "IndicTTS" / "train.yaml"),
        help="Path to train.yaml",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default=None,
        help="Override preprocessed IndicTTS data directory (e.g. on Google Drive)",
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        default=None,
        help="Optional explicit path to train.txt manifest",
    )
    parser.add_argument(
        "--val_manifest",
        type=str,
        default=None,
        help="Optional explicit path to val.txt manifest",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint directory (e.g. output/ckpt/IndicTTS)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Override persistent log directory (e.g. output/log/IndicTTS)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Automatically resume from latest valid checkpoint in checkpoint_dir if available",
    )
    parser.add_argument(
        "--restore_step",
        type=int,
        default=None,
        help="Specific step number to resume from",
    )
    parser.add_argument(
        "--max_to_keep",
        type=int,
        default=5,
        help="Maximum periodic step checkpoints to retain (default: 5)",
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=1000,
        help="Periodic checkpoint save interval in steps (default: 1000)",
    )
    parser.add_argument(
        "--val_step",
        type=int,
        default=1000,
        help="Validation interval in steps (default: 1000)",
    )
    parser.add_argument(
        "--log_step",
        type=int,
        default=100,
        help="Logging and dashboard update interval in steps (default: 100)",
    )
    parser.add_argument(
        "--total_step",
        type=int,
        default=None,
        help="Override total training steps (default: 20000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'mps', 'cpu', or None for auto)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size per step",
    )
    parser.add_argument(
        "--no_notebook_display",
        action="store_true",
        help="Disable IPython clear_output/display updates (useful for pure terminal runs)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Execute dataset validation, smoke test, 1 train step, 1 val step, and verified save, then exit",
    )
    return parser.parse_args(argv)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Execute validated Telugu FastSpeech 2 training loop with live telemetry."""
    print("=" * 72)
    print("FASTSPEECH 2 TELUGU (INDICTTS) TRAINING PIPELINE")
    print("=" * 72)

    # 1. Load YAML configuration files
    with open(_resolve_file_path(args.preprocess_config), encoding="utf-8") as f:
        prep_dict = yaml.safe_load(f)
    with open(_resolve_file_path(args.model_config), encoding="utf-8") as f:
        model_dict = yaml.safe_load(f)
    with open(_resolve_file_path(args.train_config), encoding="utf-8") as f:
        train_dict = yaml.safe_load(f)

    # Apply directory overrides
    if args.preprocessed_dir is not None:
        p_override = Path(args.preprocessed_dir).expanduser()
        prep_dict["path"]["preprocessed_path"] = str(
            p_override.resolve() if (p_override.is_dir() or p_override.is_absolute()) else resolve_project_path(p_override)
        )
    else:
        prep_dict["path"]["preprocessed_path"] = str(
            resolve_project_path(prep_dict["path"]["preprocessed_path"])
        )

    preprocessed_path = Path(prep_dict["path"]["preprocessed_path"])
    stats_file = preprocessed_path / "stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Missing stats.json at {stats_file}")
    with open(stats_file, encoding="utf-8") as f:
        model_dict["stats"] = json.load(f)
    print(f"Loaded pitch & energy stats from: {stats_file}")

    # Validate dataset before initializing model or trainers
    ds_check = validate_telugu_dataset(preprocessed_path)
    print(
        f"Dataset Integrity Verified: {ds_check['status']} "
        f"({ds_check['train_utterances']} train, {ds_check['val_utterances']} val, "
        f"{len(ds_check['speakers'])} speakers, {ds_check['total_features_verified']:,} features)"
    )

    # Apply training interval & path overrides
    if args.batch_size is not None:
        train_dict["optimizer"]["batch_size"] = int(args.batch_size)
    if args.save_step is not None:
        train_dict["step"]["save_step"] = int(args.save_step)
    if args.val_step is not None:
        train_dict["step"]["val_step"] = int(args.val_step)
    if args.log_step is not None:
        train_dict["step"]["log_step"] = int(args.log_step)
    if args.total_step is not None:
        train_dict["step"]["total_step"] = int(args.total_step)
    if args.checkpoint_dir is not None:
        train_dict["path"]["ckpt_path"] = str(Path(args.checkpoint_dir).resolve())
    else:
        train_dict["path"]["ckpt_path"] = str(resolve_project_path(train_dict["path"]["ckpt_path"]))
    if args.log_dir is not None:
        train_dict["path"]["log_path"] = str(Path(args.log_dir).resolve())
    else:
        train_dict["path"]["log_path"] = str(resolve_project_path(train_dict["path"]["log_path"]))

    fs2_cfg = FastSpeech2Config.from_dicts(
        preprocess_dict=prep_dict,
        model_dict=model_dict,
        train_dict=train_dict,
    )
    prep_cfg = fs2_cfg.preprocess
    model_cfg = fs2_cfg.model
    train_cfg = fs2_cfg.train

    # 2. Setup Compute Device
    device = _resolve_device(args.device)
    sys_info = SystemMonitor.collect(device)
    print(f"Active Compute Device : {device} ({sys_info.gpu_name})")
    if device.type == "cuda":
        print(f"CUDA Total VRAM       : {sys_info.gpu_mem_total_gb:.2f} GB")

    # 3. Setup Checkpoint Manager & Persistent Logger
    ckpt_dir = ensure_output_dir(Path(train_cfg.path.ckpt_path).resolve())
    log_dir = ensure_output_dir(Path(train_cfg.path.log_path).resolve())

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        max_to_keep=args.max_to_keep,
    )
    logger = PersistentLogger(log_dir=log_dir)
    print(f"Checkpoint Directory  : {ckpt_dir} (max_to_keep={args.max_to_keep})")
    print(f"Persistent Log Dir    : {log_dir}")

    # 4. Resolve Manifest Paths & Initialize Datasets / DataLoaders
    def _resolve_manifest_path(
        explicit_arg: str | None,
        cfg_val: str | None,
        default_filename: str,
    ) -> str:
        raw = explicit_arg or cfg_val
        if raw is None:
            return str((preprocessed_path / default_filename).resolve())
        cand = Path(raw)
        if cand.is_absolute() and cand.exists():
            return str(cand.resolve())
        for base in (PROJECT_ROOT, preprocessed_path):
            resolved = (base / cand).resolve()
            if resolved.exists():
                return str(resolved)
        return str((PROJECT_ROOT / cand).resolve())

    path_dict = train_dict.get("path", {})
    train_manifest_str = _resolve_manifest_path(
        args.train_manifest, path_dict.get("train_manifest"), "train.txt"
    )
    val_manifest_str = _resolve_manifest_path(
        args.val_manifest, path_dict.get("val_manifest"), "val.txt"
    )
    print(f"Train Manifest        : {train_manifest_str}")
    print(f"Val Manifest          : {val_manifest_str}")

    batch_size = train_cfg.optimizer.batch_size
    group_size = 4
    train_dataset = Dataset(
        train_manifest_str, prep_dict, train_dict, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        val_manifest_str, prep_dict, train_dict, sort=False, drop_last=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0,
    )

    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    total_step = train_cfg.step.total_step
    estimated_total_epochs = max(1, math.ceil(total_step / steps_per_epoch))
    print(
        f"Dataset Loaded        : {len(train_dataset)} train, {len(val_dataset)} val "
        f"(~{steps_per_epoch} steps/epoch, ~{estimated_total_epochs} epochs for {total_step:,} steps)"
    )

    # 5. Build Model, Loss, Optimizer, Trainer
    model = FastSpeech2(prep_cfg, model_cfg)
    loss_fn = FastSpeech2Loss(prep_cfg, model_cfg)
    optimizer = ScheduledOptim(model, train_cfg, model_cfg, current_step=0)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FastSpeech 2 Params   : {num_params:,} total, {num_trainable:,} trainable")

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_config=train_cfg,
        device=device,
        checkpoint_manager=checkpoint_manager,
        grad_clip_thresh=train_cfg.optimizer.grad_clip_thresh,
        grad_acc_step=train_cfg.optimizer.grad_acc_step,
    )

    # 6. Resume Handling
    resumed = False
    mode_banner = "NEW RUN"
    if args.restore_step is not None:
        target_step = args.restore_step
        print(f"\n[RESUME] Restoring from explicit step {target_step}...")
        target_path = checkpoint_manager.resolve_checkpoint_path(target_step)
        trainer.resume_from_checkpoint(checkpoint_target=target_path)
        resumed = True
    elif args.resume:
        try:
            target_path = checkpoint_manager.resolve_checkpoint_path(None)
            print(f"\n[RESUME] Found existing checkpoint: {target_path.name}")
            trainer.resume_from_checkpoint(checkpoint_target=target_path)
            resumed = True
        except FileNotFoundError:
            print(
                "\n[FRESH START] No existing checkpoints found in checkpoint_dir. "
                "Initializing fresh training from step 0."
            )

    if resumed:
        mode_banner = f"RESUMED @ STEP {trainer.global_step}"
        logger.restore_and_deduplicate(max_valid_step=trainer.global_step)
        if math.isfinite(trainer.best_val_loss):
            logger.best_val_loss = trainer.best_val_loss
        if logger.best_val_step == 0 and checkpoint_manager.best_path.exists():
            try:
                best_ckpt_meta = verify_checkpoint_file(checkpoint_manager.best_path)
                logger.best_val_step = int(best_ckpt_meta.get("global_step", 0))
            except Exception:
                pass
        print(
            f"[RESUME OK] Restored Step={trainer.global_step}, Epoch={trainer.epoch}, "
            f"LR={optimizer.get_lr():.6e}, Best Val Loss={trainer.best_val_loss:.4f}"
        )
        logger.log_message(
            f"Resumed training at step={trainer.global_step}, epoch={trainer.epoch}, "
            f"lr={optimizer.get_lr():.6e}, best_val_loss={trainer.best_val_loss:.4f}"
        )
    else:
        logger.log_message("Starting fresh training run from step 0.")

    # 7. Initialize Live Dashboard
    dashboard = LiveDashboard(
        logger=logger,
        total_steps=total_step,
        total_epochs=estimated_total_epochs,
        enable_notebook_display=(not args.no_notebook_display) and (not args.dry_run),
    )

    if resumed and (logger.train_records or logger.val_records):
        dashboard.render_plots()

    # 8. Dry-Run Verification Mode
    if args.dry_run:
        print("\n--- Dry Run Mode: Validating Model Smoke Test, 1 Train & 1 Val Step ---")
        smoke_res = run_model_smoke_test(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        print(f"Model Smoke Test      : {smoke_res['status']}")

        train_iter = iter(train_loader)
        batch_group = next(train_iter)
        step_loss = trainer.train_step(batch_group[0])
        logger.log_train_step(
            epoch=trainer.epoch,
            global_step=trainer.global_step,
            losses=step_loss,
            steps_per_sec=1.0,
        )
        print(
            f"Dry Run Train Step OK : Step={trainer.global_step}, "
            f"Loss={step_loss.total:.4f} (Mel:{step_loss.mel:.4f}, PN:{step_loss.postnet_mel:.4f})"
        )

        val_iter = iter(val_loader)
        val_group = next(val_iter)
        val_step_loss, _ = trainer.validate_step(val_group[0])
        print(
            f"Dry Run Val Step OK   : Loss={val_step_loss.total:.4f} (Mel:{val_step_loss.mel:.4f})"
        )

        saved_path = save_and_verify_checkpoint(
            trainer=trainer,
            is_best=False,
            max_to_keep=args.max_to_keep,
            extra_metadata={"dry_run": True, "dataset": "IndicTTS"},
            logger=logger,
        )
        print(f"Dry Run Save & Verify : OK ({saved_path.name})")

        # Cleanup dry-run test checkpoint files
        saved_path.unlink(missing_ok=True)
        if checkpoint_manager.latest_path.exists():
            checkpoint_manager.latest_path.unlink(missing_ok=True)
        print("Dry Run Verification Complete. System is fully operational.")
        return {"status": "DRY_RUN_COMPLETED", "step": trainer.global_step}

    # 9. Main Training Loop
    save_step = train_cfg.step.save_step
    val_step = train_cfg.step.val_step
    log_step = train_cfg.step.log_step

    print("\n" + "=" * 72)
    print(f"STARTING TRAINING LOOP: {trainer.global_step:,} -> {total_step:,} steps")
    print(f"Save Interval: {save_step} | Val Interval: {val_step} | Log Interval: {log_step}")
    print("=" * 72 + "\n")

    start_time = time.time()
    steps_since_start = 0
    completed_normally = False

    try:
        while trainer.global_step < total_step:
            for batch_group in train_loader:
                if isinstance(batch_group, list) and len(batch_group) == 0:
                    continue
                batches = (
                    batch_group
                    if isinstance(batch_group, list)
                    and isinstance(batch_group[0], (tuple, list))
                    else [batch_group]
                )

                for batch in batches:
                    if trainer.global_step >= total_step:
                        break

                    prev_step = trainer.global_step
                    step_loss = trainer.train_step(batch)

                    if trainer.global_step > prev_step:
                        steps_since_start += 1
                        elapsed = max(time.time() - start_time, 1e-3)
                        steps_per_sec = steps_since_start / elapsed

                        if step_loss.grad_norm is not None and not math.isfinite(step_loss.grad_norm):
                            logger.record_warning(
                                step=trainer.global_step,
                                epoch=trainer.epoch,
                                category="NON_FINITE_GRAD_NORM",
                                message=f"Non-finite gradient norm detected at step {trainer.global_step}",
                            )

                        # Periodic Logging & Live Dashboard Update
                        if trainer.global_step % log_step == 0 or trainer.global_step == 1:
                            logger.log_train_step(
                                epoch=trainer.epoch,
                                global_step=trainer.global_step,
                                losses=step_loss,
                                steps_per_sec=steps_per_sec,
                            )
                            dashboard.update(
                                epoch=trainer.epoch,
                                global_step=trainer.global_step,
                                train_losses=step_loss,
                                elapsed_sec=elapsed,
                                steps_per_sec=steps_per_sec,
                                device=device,
                                mode_banner=mode_banner,
                            )
                            print(
                                f"Step {trainer.global_step:7d}/{total_step} | Epoch {trainer.epoch:4d} | "
                                f"Loss: {step_loss.total:7.4f} (Mel:{step_loss.mel:6.4f}, PN:{step_loss.postnet_mel:6.4f}, "
                                f"P:{step_loss.pitch:6.4f}, E:{step_loss.energy:6.4f}, D:{step_loss.duration:6.4f}) | "
                                f"LR: {step_loss.lr:.2e} | {steps_per_sec:4.1f} step/s"
                            )

                        # Validation Evaluation
                        if trainer.global_step % val_step == 0:
                            prev_best = trainer.best_val_loss
                            val_loss = trainer.validate(val_loader)
                            if not math.isfinite(val_loss.total):
                                logger.record_warning(
                                    step=trainer.global_step,
                                    epoch=trainer.epoch,
                                    category="INVALID_VALIDATION_LOSS",
                                    message=f"Validation produced non-finite loss: {val_loss.total}",
                                )
                            improved = val_loss.total < prev_best
                            saved_ckpt = save_and_verify_checkpoint(
                                trainer=trainer,
                                is_best=improved,
                                max_to_keep=args.max_to_keep,
                                extra_metadata={"dataset": "IndicTTS"},
                                logger=logger,
                            )
                            logger.log_validation_step(
                                epoch=trainer.epoch,
                                global_step=trainer.global_step,
                                val_losses=val_loss,
                                best_val_loss=trainer.best_val_loss,
                                improved=improved,
                                checkpoint_path=saved_ckpt,
                            )
                            dashboard.update(
                                epoch=trainer.epoch,
                                global_step=trainer.global_step,
                                train_losses=step_loss,
                                elapsed_sec=elapsed,
                                steps_per_sec=steps_per_sec,
                                device=device,
                                mode_banner=mode_banner,
                            )
                            status_str = "NEW BEST" if improved else "no improvement"
                            print(
                                f"\n[VALIDATION @ Step {trainer.global_step}] "
                                f"Val Loss: {val_loss.total:.4f} (Mel:{val_loss.mel:.4f}, PN:{val_loss.postnet_mel:.4f}, "
                                f"P:{val_loss.pitch:.4f}, E:{val_loss.energy:.4f}, D:{val_loss.duration:.4f}) | "
                                f"Best: {trainer.best_val_loss:.4f} ({status_str})\n"
                            )

                        # Periodic Checkpoint Saving
                        elif trainer.global_step % save_step == 0:
                            saved_ckpt = save_and_verify_checkpoint(
                                trainer=trainer,
                                is_best=False,
                                max_to_keep=args.max_to_keep,
                                extra_metadata={"dataset": "IndicTTS"},
                                logger=logger,
                            )
                            print(
                                f"[CHECKPOINT] Verified & Saved @ Step {trainer.global_step}: {saved_ckpt.name}"
                            )

                        # Update status JSON whenever a checkpoint or log step occurs
                        if (
                            trainer.global_step % log_step == 0
                            or trainer.global_step % save_step == 0
                            or trainer.global_step % val_step == 0
                        ):
                            logger.update_status(
                                status="TRAINING",
                                epoch=trainer.epoch,
                                global_step=trainer.global_step,
                                total_step=total_step,
                                best_val_loss=trainer.best_val_loss,
                                latest_checkpoint=checkpoint_manager.latest_path
                                if checkpoint_manager.latest_path.exists()
                                else None,
                                best_checkpoint=checkpoint_manager.best_path
                                if checkpoint_manager.best_path.exists()
                                else None,
                            )

                if trainer.global_step >= total_step:
                    break

            if trainer.global_step < total_step:
                trainer.epoch += 1

        completed_normally = trainer.global_step >= total_step

    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Training paused by user at step {trainer.global_step}.")
        logger.record_warning(
            step=trainer.global_step,
            epoch=trainer.epoch,
            category="KEYBOARD_INTERRUPT",
            message="Training interrupted by user",
        )
        try:
            emerg_path = save_and_verify_checkpoint(
                trainer=trainer,
                is_best=False,
                max_to_keep=args.max_to_keep,
                extra_metadata={"dataset": "IndicTTS", "emergency": True},
                logger=logger,
            )
            print(f"[RECOVERY] Emergency checkpoint verified at: {emerg_path}")
        except Exception as ckpt_err:
            print(f"[ERROR] Emergency checkpoint save failed: {ckpt_err}")

        logger.update_status(
            status="INTERRUPTED",
            epoch=trainer.epoch,
            global_step=trainer.global_step,
            total_step=total_step,
            best_val_loss=trainer.best_val_loss,
            latest_checkpoint=checkpoint_manager.latest_path
            if checkpoint_manager.latest_path.exists()
            else None,
            best_checkpoint=checkpoint_manager.best_path
            if checkpoint_manager.best_path.exists()
            else None,
        )
    except (FloatingPointError, RuntimeError) as e:
        is_oom = "out of memory" in str(e).lower()
        cat = "CUDA_OUT_OF_MEMORY" if is_oom else "RUNTIME_ERROR"
        print(f"\n[CRITICAL {cat}] at step {trainer.global_step}: {e}")
        if is_oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.record_warning(
            step=trainer.global_step,
            epoch=trainer.epoch,
            category=cat,
            message=str(e),
        )
        logger.update_status(
            status="FAILED_OOM" if is_oom else "FAILED_RUNTIME_ERROR",
            epoch=trainer.epoch,
            global_step=trainer.global_step,
            total_step=total_step,
            best_val_loss=trainer.best_val_loss,
            latest_checkpoint=checkpoint_manager.latest_path
            if checkpoint_manager.latest_path.exists()
            else None,
            best_checkpoint=checkpoint_manager.best_path
            if checkpoint_manager.best_path.exists()
            else None,
        )
        raise

    total_duration = time.time() - start_time
    if steps_since_start == 0 and len(logger.train_records) >= 2:
        try:
            from datetime import datetime

            t0_str = str(logger.train_records[0].get("timestamp", "")).replace("Z", "+00:00")
            t1_str = str(logger.train_records[-1].get("timestamp", "")).replace("Z", "+00:00")
            hist_duration = (
                datetime.fromisoformat(t1_str) - datetime.fromisoformat(t0_str)
            ).total_seconds()
            if hist_duration > total_duration:
                total_duration = hist_duration
        except Exception:
            pass

    if completed_normally:
        step_ckpt_path = checkpoint_manager.step_path(trainer.global_step)
        if steps_since_start > 0 and not step_ckpt_path.exists():
            final_path = save_and_verify_checkpoint(
                trainer=trainer,
                is_best=False,
                max_to_keep=args.max_to_keep,
                extra_metadata={"final": True, "dataset": "IndicTTS"},
                logger=logger,
            )
        elif step_ckpt_path.exists():
            final_path = step_ckpt_path
        elif checkpoint_manager.latest_path.exists():
            final_path = checkpoint_manager.latest_path
        else:
            final_path = save_and_verify_checkpoint(
                trainer=trainer,
                is_best=False,
                max_to_keep=args.max_to_keep,
                extra_metadata={"final": True, "dataset": "IndicTTS"},
                logger=logger,
            )
        logger.update_status(
            status="COMPLETED",
            epoch=trainer.epoch,
            global_step=trainer.global_step,
            total_step=total_step,
            best_val_loss=trainer.best_val_loss,
            latest_checkpoint=checkpoint_manager.latest_path
            if checkpoint_manager.latest_path.exists()
            else final_path,
            best_checkpoint=checkpoint_manager.best_path
            if checkpoint_manager.best_path.exists()
            else None,
        )
        print(f"\n[COMPLETED] Training reached {trainer.global_step} steps! Final checkpoint: {final_path}")
    else:
        final_path = checkpoint_manager.latest_path if checkpoint_manager.latest_path.exists() else None
        print(
            f"\n[INCOMPLETE] Training stopped at step {trainer.global_step}/{total_step}. "
            "Resume from latest checkpoint to continue."
        )

    dashboard.render_plots()
    summary = generate_training_summary(
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        dashboard=dashboard,
        total_duration_sec=total_duration,
        completed_normally=completed_normally,
        final_epoch=trainer.epoch,
        final_step=trainer.global_step,
        final_lr=optimizer.get_lr(),
    )
    return {
        **summary,
        "status": "COMPLETED" if completed_normally else "INTERRUPTED",
        "final_step": trainer.global_step,
        "final_epoch": trainer.epoch,
        "best_val_loss": trainer.best_val_loss,
        "latest_checkpoint": str(final_path) if final_path else None,
    }


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
