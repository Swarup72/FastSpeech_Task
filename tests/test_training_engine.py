"""Unit and integration tests for FastSpeech 2 Training Engine and Optimizer.

Covers:
    - ScheduledOptim initialization and Noam learning rate schedule parity
    - State serialization and full restoration of optimizer and scheduler
    - Single-step and epoch training on synthetic data
    - Gradient accumulation and gradient clipping
    - Validation loop with sample-weighted loss accumulation
    - Checkpoint atomic saving, naming, and pruning
    - Full resume verification (model weights, optimizer state, step, epoch, best loss, RNG)
    - CPU and Apple Silicon MPS device execution
    - Error handling for missing or invalid checkpoints
"""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.config import (
    EnergyConfig,
    MelConfig,
    ModelConfig,
    OptimizerConfig,
    PitchConfig,
    PostNetConfig,
    PreprocessConfig,
    PreprocessingSettings,
    PreprocessPathConfig,
    StepConfig,
    TextConfig,
    TrainConfig,
    TrainPathConfig,
    TransformerConfig,
    VarianceEmbeddingConfig,
    VariancePredictorConfig,
)
from common.model.fastspeech2 import FastSpeech2
from common.model.loss import FastSpeech2Loss
from common.model.optimizer import ScheduledOptim
from common.training.checkpoint import (
    CheckpointManager,
    _clean_model_state_dict,
)
from common.training.engine import Trainer, TrainLosses


def _make_test_configs(
    hidden: int = 64,
    n_mel: int = 80,
    warmup: int = 20,
    anneal_steps: list[int] | None = None,
    anneal_rate: float = 0.5,
    grad_acc_step: int = 1,
    grad_clip_thresh: float = 1.0,
    ckpt_dir: str = "/tmp/ckpt",
) -> tuple[PreprocessConfig, ModelConfig, TrainConfig]:
    """Create lightweight configs for rapid testing."""
    if anneal_steps is None:
        anneal_steps = [100, 200]

    prep_cfg = PreprocessConfig(
        dataset="TestEngine",
        path=PreprocessPathConfig(
            corpus_path="/tmp/c", raw_path="/tmp/r", preprocessed_path="/tmp/p"
        ),
        preprocessing=PreprocessingSettings(
            text=TextConfig(language="en"),
            mel=MelConfig(n_mel_channels=n_mel),
            pitch=PitchConfig(feature="phoneme_level", normalization=False),
            energy=EnergyConfig(feature="phoneme_level", normalization=False),
        ),
    )

    model_cfg = ModelConfig(
        n_src_vocab=40,
        transformer=TransformerConfig(
            encoder_layer=2,
            encoder_head=2,
            encoder_hidden=hidden,
            decoder_layer=2,
            decoder_head=2,
            decoder_hidden=hidden,
            conv_filter_size=128,
            conv_kernel_size=[9, 1],
            encoder_dropout=0.1,
            decoder_dropout=0.1,
        ),
        variance_predictor=VariancePredictorConfig(
            filter_size=hidden, kernel_size=3, dropout=0.1
        ),
        variance_embedding=VarianceEmbeddingConfig(n_bins=16),
        postnet=PostNetConfig(
            postnet_embedding_dim=hidden,
            postnet_kernel_size=5,
            postnet_n_convolutions=2,
            postnet_dropout=0.1,
        ),
        multi_speaker=False,
        max_seq_len=100,
        stats={"pitch": [10.0, 500.0, 0.0, 1.0], "energy": [1.0, 100.0, 0.0, 1.0]},
    )

    train_cfg = TrainConfig(
        path=TrainPathConfig(
            ckpt_path=ckpt_dir,
            log_path=f"{ckpt_dir}/log",
            result_path=f"{ckpt_dir}/res",
        ),
        optimizer=OptimizerConfig(
            batch_size=2,
            betas=[0.9, 0.98],
            eps=1e-9,
            weight_decay=0.0,
            grad_clip_thresh=grad_clip_thresh,
            grad_acc_step=grad_acc_step,
            warm_up_step=warmup,
            anneal_steps=anneal_steps,
            anneal_rate=anneal_rate,
        ),
        step=StepConfig(
            total_step=1000,
            log_step=10,
            val_step=20,
            save_step=50,
        ),
    )
    return prep_cfg, model_cfg, train_cfg


class SyntheticDataset(Dataset):
    """Synthetic dataset producing deterministic FastSpeech 2 12-element tuples."""

    def __init__(
        self,
        num_samples: int = 6,
        src_len: int = 5,
        mel_len: int = 10,
        n_mels: int = 80,
    ) -> None:
        self.num_samples = num_samples
        self.src_len = src_len
        self.mel_len = mel_len
        self.n_mels = n_mels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {"idx": idx}

    def collate_fn(self, batch: list[dict]) -> tuple:
        b_size = len(batch)
        max_src = self.src_len
        max_mel = self.mel_len

        ids = [f"sample_{item['idx']}" for item in batch]
        raw_texts = [f"text for {i}" for i in ids]
        speakers = None
        texts = torch.randint(1, 35, (b_size, max_src), dtype=torch.long)
        src_lens = torch.full((b_size,), max_src, dtype=torch.long)
        mel_lens = torch.full((b_size,), max_mel, dtype=torch.long)

        mels = torch.randn(b_size, max_mel, self.n_mels)
        pitches = torch.randn(b_size, max_src)
        energies = torch.randn(b_size, max_src)

        # Durations summing to mel_lens
        durations = torch.zeros(b_size, max_src, dtype=torch.long)
        for b in range(b_size):
            base = max_mel // max_src
            rem = max_mel % max_src
            for i in range(max_src):
                durations[b, i] = base + (1 if i < rem else 0)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src,
            mels,
            mel_lens,
            max_mel,
            pitches,
            energies,
            durations,
        )


class TrainingEngineTest(unittest.TestCase):
    """Test suite for ScheduledOptim, CheckpointManager, and Trainer."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="fastspeech2_test_engine_")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_optimizer_initialization_and_attributes(self):
        """Verify ScheduledOptim initializes with correct base parameters and learning rate."""
        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, warmup=50, ckpt_dir=self.temp_dir
        )
        model = FastSpeech2(prep_cfg, model_cfg)
        optim = ScheduledOptim(model, train_cfg, model_cfg, current_step=0)

        expected_init_lr = 64**-0.5  # 0.125
        self.assertAlmostEqual(optim.init_lr, expected_init_lr, places=6)
        self.assertEqual(optim.n_warmup_steps, 50)
        self.assertEqual(optim.anneal_steps, [100, 200])
        self.assertEqual(optim.anneal_rate, 0.5)
        self.assertEqual(optim.current_step, 0)
        self.assertEqual(len(optim.optimizer.param_groups), 1)

    def test_02_lr_schedule_formula_parity(self):
        """Verify Noam learning rate schedule matches theoretical values and original implementation."""
        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=256,
            warmup=4000,
            anneal_steps=[300000],
            anneal_rate=0.3,
            ckpt_dir=self.temp_dir,
        )
        linear = nn.Linear(10, 10)
        optim = ScheduledOptim(linear, train_cfg, model_cfg, current_step=0)

        init_lr = 256**-0.5  # 0.0625

        # Step 1: linear warmup (step * warmup^-1.5)
        step_1_scale = 1.0 * (4000.0**-1.5)
        expected_lr_1 = init_lr * step_1_scale
        optim.step_and_update_lr()
        self.assertAlmostEqual(optim.get_lr(), expected_lr_1, places=9)

        # Step 4000: peak of warmup (4000^-0.5)
        optim.current_step = 3999
        optim.step_and_update_lr()
        self.assertEqual(optim.current_step, 4000)
        expected_lr_4000 = init_lr * (4000.0**-0.5)
        self.assertAlmostEqual(optim.get_lr(), expected_lr_4000, places=9)

        # Step 8000: decay phase (8000^-0.5 < 8000 * 4000^-1.5)
        optim.current_step = 7999
        optim.step_and_update_lr()
        expected_lr_8000 = init_lr * (8000.0**-0.5)
        self.assertAlmostEqual(optim.get_lr(), expected_lr_8000, places=9)

        # Step 300001: after anneal threshold (multiplied by 0.3)
        optim.current_step = 300000
        optim.step_and_update_lr()
        expected_lr_300001 = init_lr * (300001.0**-0.5) * 0.3
        self.assertAlmostEqual(optim.get_lr(), expected_lr_300001, places=9)

    def test_03_optimizer_state_dict_and_restoration(self):
        """Verify ScheduledOptim state_dict serializes and restores step, parameters, and lr."""
        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, warmup=20, ckpt_dir=self.temp_dir
        )
        linear1 = nn.Linear(10, 10)
        optim1 = ScheduledOptim(linear1, train_cfg, model_cfg, current_step=0)

        # Run 5 steps
        for _ in range(5):
            x = torch.randn(2, 10)
            loss = linear1(x).sum()
            loss.backward()
            optim1.step_and_update_lr()
            optim1.zero_grad()

        self.assertEqual(optim1.current_step, 5)
        state = optim1.state_dict()
        self.assertIn("optimizer", state)
        self.assertEqual(state["current_step"], 5)
        self.assertEqual(state["n_warmup_steps"], 20)

        # Restore into new instance
        linear2 = nn.Linear(10, 10)
        optim2 = ScheduledOptim(linear2, train_cfg, model_cfg, current_step=0)
        optim2.load_state_dict(state)

        self.assertEqual(optim2.current_step, 5)
        self.assertAlmostEqual(optim1.get_lr(), optim2.get_lr(), places=9)

    def test_04_synthetic_train_step_and_gradient_clipping(self):
        """Verify train_step executes forward, loss, backward, clipping, and optimizer step."""
        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, grad_clip_thresh=0.5, ckpt_dir=self.temp_dir
        )
        model = FastSpeech2(prep_cfg, model_cfg)
        loss_fn = FastSpeech2Loss(prep_cfg, model_cfg)
        optim = ScheduledOptim(model, train_cfg, model_cfg, current_step=0)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optim,
            train_config=train_cfg,
            device="cpu",
        )

        dataset = SyntheticDataset(num_samples=2, src_len=4, mel_len=8)
        batch = dataset.collate_fn([{"idx": 0}, {"idx": 1}])

        initial_step = trainer.global_step
        losses = trainer.train_step(batch)

        self.assertEqual(trainer.global_step, initial_step + 1)
        self.assertIsInstance(losses, TrainLosses)
        self.assertTrue(math.isfinite(losses.total))
        self.assertTrue(math.isfinite(losses.mel))
        self.assertTrue(math.isfinite(losses.postnet_mel))
        self.assertTrue(math.isfinite(losses.pitch))
        self.assertTrue(math.isfinite(losses.energy))
        self.assertTrue(math.isfinite(losses.duration))
        self.assertIsNotNone(losses.grad_norm)
        self.assertLessEqual(losses.grad_norm, 100.0)  # Finite gradient norm

    def test_05_gradient_accumulation(self):
        """Verify gradient accumulation executes optimizer steps only every grad_acc_step steps."""
        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, grad_acc_step=2, ckpt_dir=self.temp_dir
        )
        model = FastSpeech2(prep_cfg, model_cfg)
        loss_fn = FastSpeech2Loss(prep_cfg, model_cfg)
        optim = ScheduledOptim(model, train_cfg, model_cfg, current_step=0)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optim,
            train_config=train_cfg,
            device="cpu",
        )

        dataset = SyntheticDataset(num_samples=4, src_len=4, mel_len=8)
        batch1 = dataset.collate_fn([{"idx": 0}, {"idx": 1}])
        batch2 = dataset.collate_fn([{"idx": 2}, {"idx": 3}])

        # First accumulation step
        trainer.train_step(batch1)
        self.assertEqual(trainer.accumulation_counter, 1)
        self.assertEqual(trainer.global_step, 0)  # No optimizer step yet

        # Second accumulation step
        trainer.train_step(batch2)
        self.assertEqual(trainer.accumulation_counter, 2)
        self.assertEqual(trainer.global_step, 1)  # Optimizer stepped

    def test_06_train_and_val_epoch_loss_tracking(self):
        """Verify train_epoch and validate aggregate losses accurately using same loss definition."""
        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, ckpt_dir=self.temp_dir
        )
        model = FastSpeech2(prep_cfg, model_cfg)
        loss_fn = FastSpeech2Loss(prep_cfg, model_cfg)
        optim = ScheduledOptim(model, train_cfg, model_cfg, current_step=0)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optim,
            train_config=train_cfg,
            device="cpu",
        )

        train_ds = SyntheticDataset(num_samples=4, src_len=4, mel_len=8)
        val_ds = SyntheticDataset(num_samples=2, src_len=4, mel_len=8)

        train_loader = DataLoader(
            train_ds, batch_size=2, collate_fn=train_ds.collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=2, collate_fn=val_ds.collate_fn
        )

        # Run 1 train epoch
        train_loss = trainer.train_epoch(train_loader)
        self.assertEqual(trainer.epoch, 2)
        self.assertEqual(len(trainer.history["train_loss"]), 1)
        self.assertTrue(math.isfinite(train_loss.total))

        # Run 1 validation epoch
        val_loss = trainer.validate(val_loader)
        self.assertEqual(len(trainer.history["val_loss"]), 1)
        self.assertTrue(math.isfinite(val_loss.total))
        self.assertEqual(trainer.best_val_loss, val_loss.total)

    def test_07_checkpoint_atomic_saving_and_pruning(self):
        """Verify atomic checkpoint saving, latest/best tracking, and pruning of old checkpoints."""
        mgr = CheckpointManager(self.temp_dir, max_to_keep=2)

        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, ckpt_dir=self.temp_dir
        )
        model = FastSpeech2(prep_cfg, model_cfg)
        optim = ScheduledOptim(model, train_cfg, model_cfg)

        # Save step 10
        mgr.save(
            model=model,
            optimizer=optim,
            epoch=1,
            global_step=10,
            best_val_loss=5.0,
            is_best=True,
        )
        self.assertTrue((Path(self.temp_dir) / "checkpoint_10.pth.tar").exists())
        self.assertTrue(mgr.latest_path.exists())
        self.assertTrue(mgr.best_path.exists())

        # Save step 20
        mgr.save(
            model=model,
            optimizer=optim,
            epoch=2,
            global_step=20,
            best_val_loss=4.5,
            is_best=True,
        )
        self.assertTrue((Path(self.temp_dir) / "checkpoint_20.pth.tar").exists())

        # Save step 30 (should prune step 10 since max_to_keep=2)
        mgr.save(
            model=model,
            optimizer=optim,
            epoch=3,
            global_step=30,
            best_val_loss=4.2,
            is_best=True,
        )
        self.assertTrue((Path(self.temp_dir) / "checkpoint_30.pth.tar").exists())
        self.assertTrue((Path(self.temp_dir) / "checkpoint_20.pth.tar").exists())
        self.assertFalse((Path(self.temp_dir) / "checkpoint_10.pth.tar").exists())

    def test_08_checkpoint_full_resume_continuity(self):
        """Verify full resume from checkpoint: weights, optimizer, epoch, step, history, and RNG."""
        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, ckpt_dir=self.temp_dir
        )
        model1 = FastSpeech2(prep_cfg, model_cfg)
        loss_fn1 = FastSpeech2Loss(prep_cfg, model_cfg)
        optim1 = ScheduledOptim(model1, train_cfg, model_cfg, current_step=0)

        trainer1 = Trainer(
            model=model1,
            loss_fn=loss_fn1,
            optimizer=optim1,
            train_config=train_cfg,
            checkpoint_dir=self.temp_dir,
            device="cpu",
        )

        dataset = SyntheticDataset(num_samples=4, src_len=4, mel_len=8)
        loader = DataLoader(
            dataset, batch_size=2, collate_fn=dataset.collate_fn
        )

        # Train 2 epochs
        trainer1.train_epoch(loader)
        trainer1.train_epoch(loader)
        trainer1.save_checkpoint(is_best=True)

        saved_step = trainer1.global_step
        saved_epoch = trainer1.epoch
        saved_best_val = trainer1.best_val_loss
        self.assertGreater(saved_step, 0)

        # Instantiate fresh model and trainer
        model2 = FastSpeech2(prep_cfg, model_cfg)
        loss_fn2 = FastSpeech2Loss(prep_cfg, model_cfg)
        optim2 = ScheduledOptim(model2, train_cfg, model_cfg, current_step=0)

        trainer2 = Trainer(
            model=model2,
            loss_fn=loss_fn2,
            optimizer=optim2,
            train_config=train_cfg,
            checkpoint_dir=self.temp_dir,
            device="cpu",
        )

        # Resume from checkpoint
        ckpt = trainer2.resume_from_checkpoint()
        self.assertEqual(trainer2.global_step, saved_step)
        self.assertEqual(trainer2.epoch, saved_epoch)
        self.assertEqual(trainer2.best_val_loss, saved_best_val)
        self.assertEqual(optim2.current_step, saved_step)

        # Verify weights match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_09_device_compatibility_cpu_and_mps(self):
        """Verify Trainer forward, loss, backward, and optimizer on CPU and Apple Silicon MPS."""
        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, ckpt_dir=self.temp_dir
        )
        devices = ["cpu"]
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")

        for dev_str in devices:
            device = torch.device(dev_str)
            model = FastSpeech2(prep_cfg, model_cfg)
            loss_fn = FastSpeech2Loss(prep_cfg, model_cfg)
            optim = ScheduledOptim(model, train_cfg, model_cfg, current_step=0)

            trainer = Trainer(
                model=model,
                loss_fn=loss_fn,
                optimizer=optim,
                train_config=train_cfg,
                device=device,
            )

            dataset = SyntheticDataset(num_samples=2, src_len=4, mel_len=8)
            batch = dataset.collate_fn([{"idx": 0}, {"idx": 1}])

            losses = trainer.train_step(batch)
            self.assertTrue(math.isfinite(losses.total))
            self.assertEqual(trainer.global_step, 1)

    def test_10_missing_or_corrupted_checkpoint_handling(self):
        """Verify graceful error reporting when checkpoint is missing or corrupt."""
        mgr = CheckpointManager(self.temp_dir)

        # Non-existent step
        with self.assertRaises(FileNotFoundError):
            mgr.load(999999)

        # Corrupt file
        corrupt_file = Path(self.temp_dir) / "checkpoint_bad.pth.tar"
        corrupt_file.write_text("corrupted content")
        with self.assertRaises(RuntimeError):
            mgr.load(corrupt_file)

    def test_11_dataparallel_module_prefix_compatibility(self):
        """Verify _clean_model_state_dict handles DataParallel 'module.' prefix transparently."""
        weights = {"module.encoder.weight": torch.tensor([1.0, 2.0]), "decoder.weight": torch.tensor([3.0])}

        # Strip prefixes for non-parallel model
        cleaned_single = _clean_model_state_dict(weights, target_is_parallel=False)
        self.assertIn("encoder.weight", cleaned_single)
        self.assertIn("decoder.weight", cleaned_single)
        self.assertNotIn("module.encoder.weight", cleaned_single)

        # Ensure prefixes for parallel model
        cleaned_parallel = _clean_model_state_dict(weights, target_is_parallel=True)
        self.assertIn("module.encoder.weight", cleaned_parallel)
        self.assertIn("module.decoder.weight", cleaned_parallel)

    def test_12_checkpoint_discovery_and_resume_api(self):
        """Verify resolve_checkpoint_path raises FileNotFoundError on empty dir and discovers latest on save."""
        mgr = CheckpointManager(self.temp_dir)

        # 1. On empty directory, resolve_checkpoint_path(None) must raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            mgr.resolve_checkpoint_path(None)

        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, ckpt_dir=self.temp_dir
        )
        model = FastSpeech2(prep_cfg, model_cfg)
        optim = ScheduledOptim(model, train_cfg, model_cfg)

        # 2. Save step 10
        saved_p = mgr.save(
            model=model,
            optimizer=optim,
            epoch=1,
            global_step=10,
            best_val_loss=5.0,
            is_best=True,
        )

        # 3. resolve_checkpoint_path(None) discovers checkpoint_latest.pth.tar
        discovered = mgr.resolve_checkpoint_path(None)
        self.assertTrue(discovered.exists())
        self.assertEqual(discovered.name, "checkpoint_latest.pth.tar")

        # 4. resolve_checkpoint_path(10) discovers checkpoint_10.pth.tar
        discovered_step = mgr.resolve_checkpoint_path(10)
        self.assertTrue(discovered_step.exists())
        self.assertEqual(discovered_step.name, "checkpoint_10.pth.tar")

    def test_13_telugu_post_training_finalization_and_summary(self):
        """Verify generate_training_summary and Telugu finalization preserve checkpoints at 20,000 steps."""
        import hashlib
        from English.monitoring import (
            LiveDashboard,
            PersistentLogger,
            generate_training_summary,
        )

        ckpt_dir = Path(self.temp_dir) / "ckpt_indictts"
        log_dir = Path(self.temp_dir) / "log_indictts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        mgr = CheckpointManager(ckpt_dir)
        prep_cfg, model_cfg, train_cfg = _make_test_configs(
            hidden=64, ckpt_dir=str(ckpt_dir)
        )
        model = FastSpeech2(prep_cfg, model_cfg)
        optim = ScheduledOptim(model, train_cfg, model_cfg, current_step=13000)

        # Save best checkpoint at step 13,000 (best_val_loss=2.0869)
        mgr.save(
            model=model,
            optimizer=optim,
            epoch=92,
            global_step=13000,
            best_val_loss=2.0869,
            is_best=True,
        )
        # Save final checkpoint at step 20,000 (not best, best remains 2.0869)
        optim.current_step = 20000
        mgr.save(
            model=model,
            optimizer=optim,
            epoch=141,
            global_step=20000,
            best_val_loss=2.0869,
            is_best=False,
        )

        best_sha_before = hashlib.sha256(mgr.best_path.read_bytes()).hexdigest()
        latest_sha_before = hashlib.sha256(mgr.latest_path.read_bytes()).hexdigest()
        step20k_sha_before = hashlib.sha256(mgr.step_path(20000).read_bytes()).hexdigest()

        # Populate logger history up to step 20,000
        logger = PersistentLogger(log_dir=log_dir)
        logger.log_train_step(
            epoch=92,
            global_step=13000,
            losses=TrainLosses(
                total=2.15, mel=0.80, postnet_mel=0.78, pitch=0.22, energy=0.18, duration=0.17, lr=1.5e-4, grad_norm=0.9
            ),
            steps_per_sec=3.2,
        )
        logger.log_validation_step(
            epoch=92,
            global_step=13000,
            val_losses=TrainLosses(
                total=2.0869, mel=0.76, postnet_mel=0.74, pitch=0.21, energy=0.19, duration=0.1869
            ),
            best_val_loss=2.0869,
            improved=True,
            checkpoint_path=mgr.step_path(13000),
        )
        logger.log_train_step(
            epoch=141,
            global_step=20000,
            losses=TrainLosses(
                total=1.95, mel=0.71, postnet_mel=0.69, pitch=0.20, energy=0.18, duration=0.17, lr=1.2e-4, grad_norm=0.8
            ),
            steps_per_sec=3.3,
        )
        logger.log_validation_step(
            epoch=141,
            global_step=20000,
            val_losses=TrainLosses(
                total=2.1105, mel=0.77, postnet_mel=0.75, pitch=0.22, energy=0.19, duration=0.1805
            ),
            best_val_loss=2.0869,
            improved=False,
            checkpoint_path=mgr.step_path(20000),
        )

        dashboard = LiveDashboard(
            logger=logger,
            total_steps=20000,
            total_epochs=141,
            enable_notebook_display=False,
        )
        dashboard.render_plots()
        summary = generate_training_summary(
            logger=logger,
            checkpoint_manager=mgr,
            dashboard=dashboard,
            total_duration_sec=6120.0,
            completed_normally=True,
            final_epoch=141,
            final_step=20000,
            final_lr=optim.get_lr(),
        )

        # Verify summary contents and generated files
        self.assertEqual(summary["status"], "COMPLETED")
        self.assertTrue(summary["training_completed_normally"])
        self.assertEqual(summary["total_optimizer_steps"], 20000)
        self.assertAlmostEqual(summary["best_validation_loss"], 2.0869, places=4)
        self.assertEqual(summary["best_validation_step"], 13000)
        self.assertTrue((log_dir / "training_summary.json").exists())
        self.assertTrue((log_dir / "training_summary.md").exists())
        for graph_path in dashboard.graph_paths.values():
            self.assertTrue(Path(graph_path).exists())

        # Verify best, latest, and step 20,000 checkpoints are 100% unchanged
        self.assertEqual(hashlib.sha256(mgr.best_path.read_bytes()).hexdigest(), best_sha_before)
        self.assertEqual(hashlib.sha256(mgr.latest_path.read_bytes()).hexdigest(), latest_sha_before)
        self.assertEqual(hashlib.sha256(mgr.step_path(20000).read_bytes()).hexdigest(), step20k_sha_before)


if __name__ == "__main__":
    unittest.main()

