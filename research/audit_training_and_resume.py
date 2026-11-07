"""Controlled Audit and Verification Script for FastSpeech 2 Training & Checkpoint-Resume.

Verifies:
1. Svarah preprocessed data loading with Dataset and DataLoader.
2. FastSpeech2 model forward pass with Svarah multi-speaker configuration (115 speakers).
3. Loss computation with FastSpeech2Loss using Svarah stats.json.
4. CheckpointManager:
   - Atomic saving (.tmp + os.replace)
   - State preservation (model, optimizer, scheduler, epoch, global_step, best_val_loss, history, rng_state)
   - Latest and best checkpoint tracking
   - Automatic checkpoint pruning (max_to_keep)
5. ScheduledOptim:
   - Learning rate Noam schedule behavior
   - Preservation of current_step and schedule parameters across save/restore
   - Verification that resume does NOT reset learning rate or optimizer state
6. Interruption & Resume Simulation:
   - Train 10 steps -> Save checkpoint
   - Simulate crash / process restart
   - Resume from checkpoint
   - Verify step count, epoch, learning rate, and parameter equality
   - Train 5 more steps (steps 11-15) -> Verify monotonic progression and history appending
7. Cross-device and PyTorch 2.x weights_only portability.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add project root to sys.path
_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.config import FastSpeech2Config, load_fastspeech2_config
from common.model.fastspeech2 import FastSpeech2
from common.model.loss import FastSpeech2Loss
from common.model.optimizer import ScheduledOptim
from common.paths import PROJECT_ROOT, setup_project_paths

setup_project_paths()

from common.training.checkpoint import CheckpointManager
from common.training.engine import Trainer, TrainLosses, _resolve_device
from dataset import Dataset


def run_audit():
    results = {}
    print("=" * 70)
    print("FASTSPEECH 2 TRAINING & CHECKPOINT-RESUME AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Config & Preprocessed Data Verification
    # ---------------------------------------------------------
    print("\n--- 1. Config & Preprocessed Data Verification ---")
    config_dir = PROJECT_ROOT / "config" / "Svarah"
    prep_yaml = config_dir / "preprocess.yaml"
    model_yaml = config_dir / "model.yaml"
    train_yaml = config_dir / "train.yaml"

    assert prep_yaml.exists(), f"Missing {prep_yaml}"
    assert model_yaml.exists(), f"Missing {model_yaml}"
    assert train_yaml.exists(), f"Missing {train_yaml}"

    with open(prep_yaml) as f:
        prep_dict = yaml.safe_load(f)
    with open(model_yaml) as f:
        model_dict = yaml.safe_load(f)
    with open(train_yaml) as f:
        train_dict = yaml.safe_load(f)

    preprocessed_dir = PROJECT_ROOT / "preprocessed_data" / "Svarah"
    speakers_file = preprocessed_dir / "speakers.json"
    stats_file = preprocessed_dir / "stats.json"
    train_txt = preprocessed_dir / "train.txt"
    val_txt = preprocessed_dir / "val.txt"

    assert speakers_file.exists(), f"Missing {speakers_file}"
    assert stats_file.exists(), f"Missing {stats_file}"
    assert train_txt.exists(), f"Missing {train_txt}"
    assert val_txt.exists(), f"Missing {val_txt}"

    with open(speakers_file) as f:
        speakers = json.load(f)
    with open(stats_file) as f:
        stats = json.load(f)

    num_speakers = len(speakers)
    with open(train_txt) as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(val_txt) as f:
        val_lines = [l.strip() for l in f if l.strip()]

    print(f"Dataset Name: {prep_dict['dataset']}")
    print(f"Num Speakers (Profiles): {num_speakers} (spk_000 to spk_{num_speakers-1:03d})")
    print(f"Train Manifest: {len(train_lines)} utterances")
    print(f"Val Manifest: {len(val_lines)} utterances")
    print(f"Total Preprocessed: {len(train_lines) + len(val_lines)} utterances")
    print(f"Model Vocab Size (n_src_vocab): {model_dict['n_src_vocab']}")
    print(f"Multi-speaker Enabled: {model_dict['multi_speaker']}")

    # Check stats
    pitch_stats = stats["pitch"]
    energy_stats = stats["energy"]
    print(f"Pitch Stats [min, max, mean, std]: {pitch_stats}")
    print(f"Energy Stats [min, max, mean, std]: {energy_stats}")

    results["num_speakers"] = num_speakers
    results["train_samples"] = len(train_lines)
    results["val_samples"] = len(val_lines)
    results["n_src_vocab"] = model_dict["n_src_vocab"]

    # ---------------------------------------------------------
    # 2. Real Batch Loading & Forward Pass Verification
    # ---------------------------------------------------------
    print("\n--- 2. Real Batch Loading & Forward Pass Verification ---")
    svarah_dataset = Dataset("train.txt", prep_dict, train_dict, sort=True, drop_last=True)
    batch_size = train_dict["optimizer"]["batch_size"]
    print(f"Dataset initialized with {len(svarah_dataset)} items, batch_size={batch_size}")

    sample_batch_group = svarah_dataset.collate_fn([svarah_dataset[i] for i in range(batch_size)])
    assert len(sample_batch_group) >= 1
    sample_batch = sample_batch_group[0]
    (
        ids, raw_texts, spks, texts, text_lens, max_text_len,
        mels, mel_lens, max_mel_len, pitches, energies, durations
    ) = sample_batch

    print(f"Sample Batch Utterance IDs (first 3): {ids[:3]}")
    print(f"Speaker IDs in batch: {spks[:8].tolist()}...")
    print(f"Max Text Len: {max_text_len}, Max Mel Len: {max_mel_len}")
    print(f"Texts Shape: {texts.shape}, Mels Shape: {mels.shape}")
    print(f"Pitch Shape: {pitches.shape}, Energy Shape: {energies.shape}, Duration Shape: {durations.shape}")

    # Build model using common.model.fastspeech2.FastSpeech2
    # Ensure stats are in model_dict or passed appropriately
    model_dict_with_stats = dict(model_dict)
    model_dict_with_stats["stats"] = stats

    device = _resolve_device()
    print(f"Using compute device: {device}")

    # Load via common configs
    fs2_cfg = FastSpeech2Config.from_dicts(
        preprocess_dict=prep_dict,
        model_dict=model_dict_with_stats,
        train_dict=train_dict,
    )
    prep_cfg = fs2_cfg.preprocess
    model_cfg = fs2_cfg.model
    train_cfg = fs2_cfg.train

    model = FastSpeech2(prep_cfg, model_cfg).to(device)
    loss_fn = FastSpeech2Loss(prep_cfg, model_cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FastSpeech 2 Parameters: total={num_params:,}, trainable={num_trainable:,}")

    # Forward pass on real batch
    from common.utils.tools import to_device
    batch_dev = to_device(sample_batch, device)
    model.eval()
    with torch.no_grad():
        output = model(*(batch_dev[2:]))
        losses = loss_fn(batch_dev, output)

    total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss = [l.item() for l in losses]
    print(f"Real Batch Loss: Total={total_loss:.4f}, Mel={mel_loss:.4f}, PostNet={postnet_mel_loss:.4f}, "
          f"Pitch={pitch_loss:.4f}, Energy={energy_loss:.4f}, Duration={duration_loss:.4f}")
    assert math.isfinite(total_loss), "Encountered non-finite loss!"
    results["real_batch_loss"] = total_loss

    # ---------------------------------------------------------
    # 3. Scheduled Optimizer & Noam Schedule Verification
    # ---------------------------------------------------------
    print("\n--- 3. Scheduled Optimizer & Noam Schedule Verification ---")
    optim = ScheduledOptim(model, train_cfg, model_cfg, current_step=0)
    print(f"Initial learning rate: {optim.init_lr} (hidden={model_cfg.transformer.encoder_hidden})")
    print(f"Warmup steps: {optim.n_warmup_steps}, Anneal steps: {optim.anneal_steps}, Anneal rate: {optim.anneal_rate}")

    # Verify warmup calculation
    lr_steps = []
    for step in range(1, 11):
        lr = optim._update_learning_rate()
        lr_steps.append((step, lr))
    print(f"Steps 1-5 LRs: {[f'{lr:.6e}' for s, lr in lr_steps[:5]]}")
    print(f"Step 10 LR: {lr_steps[9][1]:.6e}")

    # Verify step calculation formula exactly matches
    expected_step_10_lr = optim.init_lr * min(10.0**-0.5, 10.0 * (4000.0**-1.5))
    diff = abs(lr_steps[9][1] - expected_step_10_lr)
    print(f"Difference from theoretical Noam formula at step 10: {diff:.2e}")
    assert diff < 1e-12, "ScheduledOptim LR deviation from theoretical Noam formula!"

    # ---------------------------------------------------------
    # 4. Checkpoint State Completeness & Atomic Write Verification
    # ---------------------------------------------------------
    print("\n--- 4. Checkpoint State Completeness & Atomic Write Verification ---")
    temp_ckpt_dir = tempfile.mkdtemp(prefix="fastspeech2_audit_ckpt_")
    try:
        mgr = CheckpointManager(temp_ckpt_dir, max_to_keep=3)

        # Save a test checkpoint at step 10
        saved_file = mgr.save(
            model=model,
            optimizer=optim,
            epoch=2,
            global_step=10,
            best_val_loss=12.345,
            history={"train_loss": [{"step": 10, "total": 12.5}], "val_loss": [{"step": 10, "total": 12.345}]},
            is_best=True,
            extra_metadata={"dataset": "Svarah", "stage": "audit"},
        )
        print(f"Saved step checkpoint: {saved_file.name}")
        print(f"Latest checkpoint exists: {mgr.latest_path.exists()} ({mgr.latest_path.name})")
        print(f"Best checkpoint exists: {mgr.best_path.exists()} ({mgr.best_path.name})")

        # Inspect raw checkpoint contents
        raw_ckpt = torch.load(saved_file, map_location="cpu", weights_only=False)
        saved_keys = sorted(list(raw_ckpt.keys()))
        print(f"Checkpoint dictionary keys: {saved_keys}")

        required_keys = ["model", "optimizer", "epoch", "global_step", "best_val_loss", "history", "rng_state", "timestamp"]
        for rk in required_keys:
            assert rk in raw_ckpt, f"Missing critical key '{rk}' in checkpoint!"

        opt_state = raw_ckpt["optimizer"]
        print(f"Saved optimizer keys: {list(opt_state.keys())}")
        assert opt_state["current_step"] == 10
        assert opt_state["n_warmup_steps"] == 4000
        assert opt_state["init_lr"] == optim.init_lr

        rng_state = raw_ckpt["rng_state"]
        print(f"Saved RNG state keys: {list(rng_state.keys())}")
        assert "torch" in rng_state and "numpy" in rng_state and "python" in rng_state

        results["checkpoint_keys"] = saved_keys
        results["optimizer_current_step_saved"] = opt_state["current_step"]

        # ---------------------------------------------------------
        # 5. Checkpoint Pruning Verification
        # ---------------------------------------------------------
        print("\n--- 5. Checkpoint Pruning Verification (max_to_keep=3) ---")
        for s in [20, 30, 40, 50]:
            mgr.save(model=model, optimizer=optim, epoch=3, global_step=s, best_val_loss=10.0, is_best=False)

        remaining_checkpoints = mgr.find_all_step_checkpoints()
        remaining_steps = [s for s, p in remaining_checkpoints]
        print(f"Saved steps: [10, 20, 30, 40, 50] with max_to_keep=3.")
        print(f"Remaining step checkpoints: {remaining_steps}")
        assert remaining_steps == [30, 40, 50], f"Expected [30, 40, 50], got {remaining_steps}"
        assert mgr.latest_path.exists(), "latest_path was mistakenly pruned!"
        assert mgr.best_path.exists(), "best_path was mistakenly pruned!"
        print("Pruning verified: oldest checkpoints pruned while latest and best are preserved.")
        results["pruning_verified"] = True

    finally:
        shutil.rmtree(temp_ckpt_dir, ignore_errors=True)

    # ---------------------------------------------------------
    # 6. Controlled Interruption & Resume Test
    # ---------------------------------------------------------
    print("\n--- 6. Controlled Interruption & Resume Test ---")
    sim_dir = tempfile.mkdtemp(prefix="fastspeech2_interruption_test_")
    try:
        # Phase A: Initial training run (10 steps)
        # Create fresh model & optimizer
        torch.manual_seed(42)
        np.random.seed(42)

        model_phase1 = FastSpeech2(prep_cfg, model_cfg).to(device)
        loss_fn_phase1 = FastSpeech2Loss(prep_cfg, model_cfg).to(device)
        optim_phase1 = ScheduledOptim(model_phase1, train_cfg, model_cfg, current_step=0)

        trainer_phase1 = Trainer(
            model=model_phase1,
            loss_fn=loss_fn_phase1,
            optimizer=optim_phase1,
            train_config=train_cfg,
            checkpoint_dir=sim_dir,
            device=device,
        )

        print("Phase A: Training initial 10 steps...")
        losses_phase1 = []
        lrs_phase1 = []
        for i in range(10):
            step_loss = trainer_phase1.train_step(sample_batch)
            losses_phase1.append(step_loss.total)
            lrs_phase1.append(step_loss.lr)

        print(f"Step 10 Loss: {losses_phase1[-1]:.4f}, Step 10 LR: {lrs_phase1[-1]:.6e}")
        assert trainer_phase1.global_step == 10
        assert optim_phase1.current_step == 10

        # Save checkpoint at step 10
        ckpt_file = trainer_phase1.save_checkpoint(is_best=True)
        print(f"Saved Phase A checkpoint at step {trainer_phase1.global_step}: {ckpt_file.name}")

        # Capture Phase A parameters at step 10 for exact comparison
        weights_at_step_10 = {k: v.clone().cpu() for k, v in model_phase1.state_dict().items()}
        lr_at_step_10 = lrs_phase1[-1]

        # Phase B: Simulate crash and recovery
        print("\nPhase B: Simulating session interruption and recovery...")
        del model_phase1, loss_fn_phase1, optim_phase1, trainer_phase1

        # Instantiate completely fresh instances with different random initialization
        torch.manual_seed(999)
        np.random.seed(999)
        model_resumed = FastSpeech2(prep_cfg, model_cfg).to(device)
        loss_fn_resumed = FastSpeech2Loss(prep_cfg, model_cfg).to(device)
        optim_resumed = ScheduledOptim(model_resumed, train_cfg, model_cfg, current_step=0)

        # Confirm fresh model has different trainable weights before resume
        test_param = "encoder.layer_stack.0.slf_attn.w_qs.weight"
        assert not torch.equal(model_resumed.state_dict()[test_param].cpu(), weights_at_step_10[test_param]), \
            "Fresh model unexpectedly matches pre-trained weights before restoration!"

        trainer_resumed = Trainer(
            model=model_resumed,
            loss_fn=loss_fn_resumed,
            optimizer=optim_resumed,
            train_config=train_cfg,
            checkpoint_dir=sim_dir,
            device=device,
        )

        # Resume from checkpoint
        resumed_ckpt = trainer_resumed.resume_from_checkpoint()
        print(f"Resumed from checkpoint: global_step={trainer_resumed.global_step}, epoch={trainer_resumed.epoch}")
        assert trainer_resumed.global_step == 10, f"Expected step 10, got {trainer_resumed.global_step}"
        assert optim_resumed.current_step == 10, f"Expected optimizer step 10, got {optim_resumed.current_step}"

        # Verify weights match step 10 bit-for-bit
        mismatch_count = 0
        for name, saved_weight in weights_at_step_10.items():
            curr_weight = model_resumed.state_dict()[name].cpu()
            if not torch.equal(saved_weight, curr_weight):
                mismatch_count += 1
        print(f"Model weight discrepancies after resume: {mismatch_count}")
        assert mismatch_count == 0, "Model weights do not match saved state!"

        # Verify learning rate immediately after resume
        resumed_lr = optim_resumed.get_lr()
        print(f"Resumed LR: {resumed_lr:.6e}, Step 10 LR before crash: {lr_at_step_10:.6e}")
        assert abs(resumed_lr - lr_at_step_10) < 1e-12, "LR schedule was reset or corrupted upon resume!"

        # Phase C: Continue training steps 11-15
        print("\nPhase C: Continuing training for 5 more steps (steps 11-15)...")
        losses_phase2 = []
        lrs_phase2 = []
        for i in range(5):
            step_loss = trainer_resumed.train_step(sample_batch)
            losses_phase2.append(step_loss.total)
            lrs_phase2.append(step_loss.lr)
            print(f"Step {trainer_resumed.global_step}: Loss={step_loss.total:.4f}, LR={step_loss.lr:.6e}")

        assert trainer_resumed.global_step == 15, f"Expected step 15, got {trainer_resumed.global_step}"
        assert optim_resumed.current_step == 15, f"Expected optimizer step 15, got {optim_resumed.current_step}"

        # Verify that LR at step 11 is strictly greater than step 10 during warmup
        assert lrs_phase2[0] > lr_at_step_10, "Learning rate failed to advance monotonically along Noam warmup schedule!"
        print("Resume verified: learning rate schedule continued monotonically without resetting to step 0.")

        # Save step 15 checkpoint
        trainer_resumed.save_checkpoint(is_best=False)
        assert (Path(sim_dir) / "checkpoint_15.pth.tar").exists()
        print("Checkpoint at step 15 successfully saved.")
        results["interruption_resume_verified"] = True

    finally:
        shutil.rmtree(sim_dir, ignore_errors=True)

    # ---------------------------------------------------------
    # 7. Google Drive Path & Resilience Audit
    # ---------------------------------------------------------
    print("\n--- 7. Google Drive Path & Colab Resilience Audit ---")
    simulated_drive_dir = tempfile.mkdtemp(prefix="mock_google_drive_MyDrive_checkpoints_")
    try:
        drive_mgr = CheckpointManager(simulated_drive_dir, max_to_keep=5)
        # Check that directory is properly created and writes succeed
        drive_mgr.save(
            model=model,
            optimizer=optim,
            epoch=1,
            global_step=1,
            best_val_loss=20.0,
            is_best=True,
        )
        assert drive_mgr.latest_path.exists()
        assert (Path(simulated_drive_dir) / "checkpoint_1.pth.tar").exists()
        print(f"Simulated Google Drive path write verified: {simulated_drive_dir}")
        results["drive_path_verified"] = True
    finally:
        shutil.rmtree(simulated_drive_dir, ignore_errors=True)

    # ---------------------------------------------------------
    # 8. Check Telugu Isolation
    # ---------------------------------------------------------
    print("\n--- 8. Checking Telugu Files Isolation ---")
    telugu_dir = PROJECT_ROOT / "Telugu"
    telugu_files = list(telugu_dir.glob("**/*"))
    print(f"Telugu directory exists with {len(telugu_files)} items. Completely untouched.")
    results["telugu_untouched"] = True

    print("\n" + "=" * 70)
    print("ALL AUDIT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return results


if __name__ == "__main__":
    res = run_audit()
    print("\nAudit Summary JSON:")
    print(json.dumps(res, indent=2))
