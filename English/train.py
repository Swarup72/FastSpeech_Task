"""English FastSpeech 2 Training Runner for Svarah Dataset (Colab & CLI).

Integrates:
    - Svarah multi-speaker dataset loading (115 demographic profiles)
    - FastSpeech 2 architecture with Svarah stats
    - ScheduledOptim with Transformer Noam learning rate schedule
    - CheckpointManager + save_and_verify_checkpoint (atomic writes, post-save load check, max_to_keep=5)
    - PersistentLogger (training_history.csv, validation_history.csv, training_status.json, training.log)
    - LiveDashboard (HTML progress table + 5 live Matplotlib loss/LR graphs)
    - Failure detection (NaN/Inf loss/grad, CUDA OOM, interruption recovery)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

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
    validate_svarah_dataset,
)


def _resolve_file_path(path_like: str | Path) -> Path:
    """Resolve an input file path, checking direct path first then project-root-relative."""
    p = Path(path_like).expanduser()
    if p.is_file() or p.is_absolute():
        return p
    proj_p = resolve_project_path(p)
    return proj_p if proj_p.is_file() else p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FastSpeech 2 on English Svarah Dataset"
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml"),
        help="Path to preprocess.yaml",
    )
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "Svarah" / "model.yaml"),
        help="Path to model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "Svarah" / "train.yaml"),
        help="Path to train.yaml",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default=None,
        help="Override preprocessed Svarah data directory (e.g. on Google Drive)",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Optional directory containing versioned train.txt and val.txt manifests",
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
        help="Override checkpoint directory (e.g. Google Drive checkpoints/Svarah)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Override persistent log directory (e.g. Google Drive logs/Svarah)",
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
        help="Periodic checkpoint save interval in steps (default: 1000 for Colab resilience)",
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
        help="Override total training steps",
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
    """Execute validated English FastSpeech 2 training loop with live monitoring."""
    print("=" * 72)
    print("FASTSPEECH 2 ENGLISH (SVARAH) TRAINING PIPELINE")
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

    # 4. Initialize Datasets & DataLoaders
    def _resolve_manifest_path(
        explicit_arg: str | None,
        cfg_val: str | None,
        split_dir_val: str | None,
        default_filename: str,
    ) -> tuple[str, Path | None]:
        raw = explicit_arg or cfg_val
        if raw is None and split_dir_val is not None:
            raw = str(Path(split_dir_val) / default_filename)
        if raw is None:
            return default_filename, None
        cand = Path(raw)
        if cand.is_absolute() and cand.exists():
            return str(cand.resolve()), cand.resolve()
        for base in (PROJECT_ROOT, preprocessed_path):
            resolved = (base / cand).resolve()
            if resolved.exists():
                return str(resolved), resolved
        resolved_fallback = (
            cand.resolve() if cand.is_absolute() else (PROJECT_ROOT / cand).resolve()
        )
        return str(resolved_fallback), resolved_fallback

    path_dict = train_dict.get("path", {})
    split_dir_cfg = getattr(args, "split_dir", None) or path_dict.get("split_dir")
    train_manifest_str, train_manifest_path = _resolve_manifest_path(
        getattr(args, "train_manifest", None),
        path_dict.get("train_manifest"),
        split_dir_cfg,
        "train.txt",
    )
    val_manifest_str, val_manifest_path = _resolve_manifest_path(
        getattr(args, "val_manifest", None),
        path_dict.get("val_manifest"),
        split_dir_cfg,
        "val.txt",
    )
    if train_manifest_path is not None or val_manifest_path is not None:
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
    print(f"FastSpeech 2 Params   : {num_params:,}")

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_config=train_cfg,
        device=device,
        checkpoint_manager=checkpoint_manager,
    )

    # 6. Check First-Run vs Resume State
    resumed = False
    mode_banner = "FRESH RUN"
    if args.restore_step is not None:
        print(f"\n[RESUME] Restoring from explicit step {args.restore_step}...")
        trainer.resume_from_checkpoint(checkpoint_target=args.restore_step)
        resumed = True
    elif args.resume:
        try:
            target_path = checkpoint_manager.resolve_checkpoint_path(None)
            print(f"\n[RESUME] Found existing checkpoint: {target_path}")
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

    dashboard = LiveDashboard(
        logger=logger,
        total_steps=total_step,
        total_epochs=estimated_total_epochs,
        enable_notebook_display=(not args.no_notebook_display) and (not args.dry_run),
    )

    # If resuming with historical data, reconstruct plots immediately
    if resumed and (logger.train_records or logger.val_records):
        dashboard.render_plots()

    # 7. Dry Run Mode
    if args.dry_run:
        print("\n--- Dry Run Mode: Validating Dataset, Model Smoke Test, 1 Train & 1 Val Step ---")
        ds_check = validate_svarah_dataset(
            preprocessed_path=preprocessed_path,
            model_config=model_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            max_file_checks=50,
            train_manifest=train_manifest_path,
            val_manifest=val_manifest_path,
        )
        print(f"Dataset Validation    : {ds_check['status']} ({ds_check['train_utterances']} train, {ds_check['val_utterances']} val)")

        smoke_res = run_model_smoke_test(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        print(
            f"Model Smoke Test      : {smoke_res['status']} "
            f"(Train Total={smoke_res['train_losses']['total']:.4f}, Val Total={smoke_res['val_losses']['total']:.4f})"
        )

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
            f"Total Loss={step_loss.total:.4f}, LR={step_loss.lr:.6e}"
        )

        val_iter = iter(val_loader)
        val_group = next(val_iter)
        val_step_loss, _ = trainer.validate_step(val_group[0])
        print(f"Dry Run Val Step OK   : Total Loss={val_step_loss.total:.4f}")

        saved_path = save_and_verify_checkpoint(
            trainer=trainer,
            is_best=False,
            max_to_keep=args.max_to_keep,
            logger=logger,
        )
        dashboard.render_plots()
        logger.update_status(
            status="DRY_RUN_COMPLETED",
            epoch=trainer.epoch,
            global_step=trainer.global_step,
            total_step=total_step,
            best_val_loss=trainer.best_val_loss,
            latest_checkpoint=checkpoint_manager.latest_path,
        )
        print(f"Verified Checkpoint   : {saved_path.name}")
        print("Dry run completed successfully! Ready for full Colab GPU training.")
        return {"status": "DRY_RUN_COMPLETED", "step": trainer.global_step}

    # 8. Main Training Loop
    save_step = train_cfg.step.save_step
    val_step = train_cfg.step.val_step
    log_step = train_cfg.step.log_step

    print("\nStarting Training Loop:")
    print(
        f"Total Steps: {total_step:,} | Log Step: {log_step} | "
        f"Val Step: {val_step} | Save Step: {save_step} | Max Checkpoints: {args.max_to_keep}"
    )

    logger.update_status(
        status="RUNNING",
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

                    # Only trigger step-based actions when optimizer actually stepped
                    if trainer.global_step > prev_step:
                        steps_since_start += 1
                        elapsed = max(time.time() - start_time, 1e-3)
                        steps_per_sec = steps_since_start / elapsed

                        # Check for suspicious gradient norm spikes
                        if step_loss.grad_norm is not None and not math.isfinite(
                            step_loss.grad_norm
                        ):
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
                                f"LR: {(step_loss.lr or 0.0):.4e} | {steps_per_sec:.2f} step/s"
                            )

                        # Periodic Validation
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
                            status_str = "NEW BEST!" if improved else "no improvement"
                            print(
                                f"\n[VALIDATION @ Step {trainer.global_step}] "
                                f"Total: {val_loss.total:.4f} (Mel: {val_loss.mel:.4f}, "
                                f"PN: {val_loss.postnet_mel:.4f}, Pitch: {val_loss.pitch:.4f}, "
                                f"Energy: {val_loss.energy:.4f}, Dur: {val_loss.duration:.4f}) | "
                                f"Best: {trainer.best_val_loss:.4f} ({status_str})\n"
                            )

                        # Periodic Checkpoint Save (if not already saved by validation at this exact step)
                        elif trainer.global_step % save_step == 0:
                            saved_ckpt = save_and_verify_checkpoint(
                                trainer=trainer,
                                is_best=False,
                                max_to_keep=args.max_to_keep,
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
                                status="RUNNING",
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
        print("\n[WARNING] Training interrupted by user / Colab signal! Saving emergency checkpoint...")
        logger.record_warning(
            step=trainer.global_step,
            epoch=trainer.epoch,
            category="INTERRUPTED",
            message="Training interrupted by KeyboardInterrupt / session signal.",
        )
        try:
            emerg_path = save_and_verify_checkpoint(
                trainer=trainer,
                is_best=False,
                max_to_keep=args.max_to_keep,
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
    except RuntimeError as e:
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
    except FloatingPointError as e:
        print(f"\n[CRITICAL NAN_OR_INF_LOSS] at step {trainer.global_step}: {e}")
        logger.record_warning(
            step=trainer.global_step,
            epoch=trainer.epoch,
            category="NAN_OR_INF_LOSS",
            message=str(e),
        )
        logger.update_status(
            status="FAILED_NON_FINITE_LOSS",
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
    if completed_normally:
        final_path = save_and_verify_checkpoint(
            trainer=trainer,
            is_best=False,
            max_to_keep=args.max_to_keep,
            logger=logger,
        )
        logger.update_status(
            status="COMPLETED",
            epoch=trainer.epoch,
            global_step=trainer.global_step,
            total_step=total_step,
            best_val_loss=trainer.best_val_loss,
            latest_checkpoint=checkpoint_manager.latest_path,
            best_checkpoint=checkpoint_manager.best_path
            if checkpoint_manager.best_path.exists()
            else None,
        )
        print(f"\n[COMPLETED] Training reached {trainer.global_step} steps! Final checkpoint: {final_path}")
    else:
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
    return summary


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
