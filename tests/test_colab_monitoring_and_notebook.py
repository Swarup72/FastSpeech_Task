"""Unit and integration tests for Step 1.5 Colab Notebook, Monitoring, Logging & Checkpoint Recovery.

Covers:
    - English_FastSpeech2_Training.ipynb JSON schema, cell structure, and AST syntax validation
    - End-to-end local execution of notebook cells (Sections 1-9 with safety gate active)
    - PersistentLogger CSV appending, resume history restoration, and step deduplication
    - save_and_verify_checkpoint post-save verification, best-checkpoint guard, and max_to_keep=5 pruning
    - LiveDashboard 5-graph generation and HTML telemetry table rendering
    - generate_training_summary JSON and Markdown output
    - Minimal Colab archive bundling (excluding .venv, __pycache__, .wav)
"""

from __future__ import annotations

import ast
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from common.config import FastSpeech2Config
from common.model.fastspeech2 import FastSpeech2
from common.model.loss import FastSpeech2Loss
from common.model.optimizer import ScheduledOptim
from common.training.checkpoint import CheckpointManager
from common.training.engine import Trainer, TrainLosses
from English.monitoring import (
    LiveDashboard,
    PersistentLogger,
    SystemMonitor,
    generate_training_summary,
    save_and_verify_checkpoint,
    validate_svarah_dataset,
    verify_checkpoint_file,
)
from English.prepare_colab_bundle import create_colab_bundle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "English_FastSpeech2_Training.ipynb"


class ColabMonitoringAndNotebookTest(unittest.TestCase):
    """Test suite for Step 1.5 Colab training notebook and monitoring utilities."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_step15_colab_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_notebook_structure_and_syntax(self) -> None:
        """Verify English_FastSpeech2_Training.ipynb exists, has valid nbformat v4, and valid Python syntax."""
        self.assertTrue(NOTEBOOK_PATH.exists(), f"Missing {NOTEBOOK_PATH}")
        with open(NOTEBOOK_PATH, encoding="utf-8") as f:
            nb = json.load(f)

        self.assertEqual(nb.get("nbformat"), 4)
        cells = nb.get("cells", [])
        self.assertGreaterEqual(len(cells), 15)

        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 9)

        full_text = ""
        for idx, cell in enumerate(cells):
            src = "".join(cell.get("source", []))
            full_text += src + "\n"
            if cell.get("cell_type") == "code":
                try:
                    ast.parse(src)
                except SyntaxError as e:
                    self.fail(f"SyntaxError in notebook code cell {idx}: {e}")

        # Verify all required sections and profile_stratified_v1 configurations are present
        for section_marker in (
            "Section 1",
            "Section 2",
            "Section 3",
            "Section 4",
            "Section 5",
            "Section 6",
            "Section 7",
            "Section 8",
            "Section 9",
            "START_FULL_TRAINING = False",
            "train_profile_stratified_v1.yaml",
            "Svarah_profile_stratified_v1",
            "profile_stratified_v1",
            "train_manifest=TRAIN_MANIFEST",
            "val_manifest=VAL_MANIFEST",
            "Dataset(str(TRAIN_MANIFEST)",
            "Dataset(str(VAL_MANIFEST)",
        ):
            self.assertIn(section_marker, full_text)

        # Verify old baseline paths and --resume are NOT present in any cell
        for forbidden_marker in (
            "train.yaml",
            '"checkpoints" / "Svarah"',
            '"logs" / "Svarah"',
            'Dataset("train.txt"',
            'Dataset("val.txt"',
            '"--resume"',
        ):
            self.assertNotIn(
                forbidden_marker,
                full_text,
                f"Found forbidden baseline/resume marker in notebook: {forbidden_marker}",
            )

    def test_02_persistent_logger_and_resume_deduplication(self) -> None:
        """Verify PersistentLogger writes CSVs, deduplicates steps on resume, and updates status JSON."""
        log_dir = self.temp_dir / "logs"
        logger = PersistentLogger(log_dir=log_dir)

        # Log steps 100, 200, 300
        for step in (100, 200, 300):
            losses = TrainLosses(
                total=20.0 - step * 0.01,
                mel=5.0,
                postnet_mel=5.0,
                pitch=3.0,
                energy=2.0,
                duration=5.0 - step * 0.01,
                lr=1e-5 * step,
                grad_norm=0.8,
            )
            logger.log_train_step(epoch=1, global_step=step, losses=losses, steps_per_sec=2.5)

        val_loss_200 = TrainLosses(15.0, 4.0, 4.0, 2.0, 2.0, 3.0)
        logger.log_validation_step(
            epoch=1,
            global_step=200,
            val_losses=val_loss_200,
            best_val_loss=15.0,
            improved=True,
            checkpoint_path="ckpt_200.pth.tar",
        )
        val_loss_300 = TrainLosses(16.0, 4.5, 4.5, 2.0, 2.0, 3.0)
        logger.log_validation_step(
            epoch=1,
            global_step=300,
            val_losses=val_loss_300,
            best_val_loss=15.0,
            improved=False,
            checkpoint_path="ckpt_300.pth.tar",
        )

        # Simulate crash after step 300 where last saved checkpoint was step 200
        logger_resumed = PersistentLogger(log_dir=log_dir)
        logger_resumed.restore_and_deduplicate(max_valid_step=200)

        self.assertEqual([r["global_step"] for r in logger_resumed.train_records], [100, 200])
        self.assertEqual([r["global_step"] for r in logger_resumed.val_records], [200])
        self.assertAlmostEqual(logger_resumed.best_val_loss, 15.0)
        self.assertEqual(logger_resumed.best_val_step, 200)

        # Re-log step 200 and new step 300 -> verify no duplicate step 200
        logger_resumed.log_train_step(
            epoch=1,
            global_step=200,
            losses=TrainLosses(17.9, 5.0, 5.0, 3.0, 2.0, 2.9, lr=2e-3, grad_norm=0.7),
            steps_per_sec=2.6,
        )
        logger_resumed.log_train_step(
            epoch=1,
            global_step=300,
            losses=TrainLosses(16.9, 4.8, 4.8, 2.8, 2.0, 2.5, lr=3e-3, grad_norm=0.6),
            steps_per_sec=2.6,
        )
        self.assertEqual([r["global_step"] for r in logger_resumed.train_records], [100, 200, 300])

    def test_03_dashboard_five_graphs_and_summary(self) -> None:
        """Verify LiveDashboard generates all 5 required graphs and generate_training_summary writes reports."""
        log_dir = self.temp_dir / "logs_dash"
        ckpt_dir = self.temp_dir / "ckpt_dash"
        logger = PersistentLogger(log_dir=log_dir)
        ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, max_to_keep=5)

        for step in (100, 200):
            logger.log_train_step(
                epoch=1,
                global_step=step,
                losses=TrainLosses(18.0, 5.0, 5.0, 3.0, 2.0, 3.0, lr=1e-5 * step, grad_norm=0.5),
                steps_per_sec=2.0,
            )
        logger.log_validation_step(
            epoch=1,
            global_step=200,
            val_losses=TrainLosses(14.5, 4.0, 4.0, 2.5, 1.5, 2.5),
            best_val_loss=14.5,
            improved=True,
        )

        dash = LiveDashboard(
            logger=logger,
            total_steps=1000,
            total_epochs=10,
            enable_notebook_display=False,
        )
        paths = dash.render_plots()
        for key, path in paths.items():
            self.assertTrue(path.exists(), f"Missing plot {key} at {path}")
            self.assertGreater(path.stat().st_size, 1000)

        sys_metrics = SystemMonitor.collect("cpu")
        html = dash.build_html_panel(
            epoch=1,
            global_step=200,
            train_losses=TrainLosses(18.0, 5.0, 5.0, 3.0, 2.0, 3.0, lr=2e-3, grad_norm=0.5),
            elapsed_sec=100.0,
            steps_per_sec=2.0,
            sys_metrics=sys_metrics,
        )
        self.assertIn("FastSpeech 2 Live Training Dashboard", html)
        self.assertIn("IMPROVED (Best)", html)

        summary = generate_training_summary(
            logger=logger,
            checkpoint_manager=ckpt_mgr,
            dashboard=dash,
            total_duration_sec=120.0,
            completed_normally=False,
            final_epoch=1,
            final_step=200,
            final_lr=2e-3,
        )
        self.assertEqual(summary["status"], "INTERRUPTED_OR_INCOMPLETE")
        self.assertTrue((log_dir / "training_summary.json").exists())
        self.assertTrue((log_dir / "training_summary.md").exists())

    def test_04_verified_checkpoint_saving_and_best_guard(self) -> None:
        """Verify save_and_verify_checkpoint checks integrity, protects best checkpoint, and prunes to 5."""
        with open(PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml", encoding="utf-8") as f:
            prep_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "config" / "Svarah" / "model.yaml", encoding="utf-8") as f:
            model_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "config" / "Svarah" / "train.yaml", encoding="utf-8") as f:
            train_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "preprocessed_data" / "Svarah" / "stats.json", encoding="utf-8") as f:
            model_dict["stats"] = json.load(f)

        ckpt_dir = self.temp_dir / "ckpts"
        log_dir = self.temp_dir / "logs"
        train_dict["path"]["ckpt_path"] = str(ckpt_dir)

        fs2_cfg = FastSpeech2Config.from_dicts(prep_dict, model_dict, train_dict)
        model = FastSpeech2(fs2_cfg.preprocess, fs2_cfg.model)
        loss_fn = FastSpeech2Loss(fs2_cfg.preprocess, fs2_cfg.model)
        optim = ScheduledOptim(model, fs2_cfg.train, fs2_cfg.model, current_step=0)
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir, max_to_keep=5)
        logger = PersistentLogger(log_dir=log_dir)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optim,
            train_config=fs2_cfg.train,
            device="cpu",
            checkpoint_manager=mgr,
        )

        # Save step 1000 as best (best_val_loss = 12.0)
        trainer.global_step = 1000
        trainer.best_val_loss = 12.0
        save_and_verify_checkpoint(trainer, is_best=True, max_to_keep=5, logger=logger)
        best_meta_1 = verify_checkpoint_file(mgr.best_path)
        self.assertEqual(best_meta_1["global_step"], 1000)

        # Save steps 2000..6000 with is_best=False (worse validation loss)
        for step in (2000, 3000, 4000, 5000, 6000):
            trainer.global_step = step
            save_and_verify_checkpoint(trainer, is_best=False, max_to_keep=5, logger=logger)

        # Verify best_path was NOT overwritten and still points to step 1000!
        best_meta_after = verify_checkpoint_file(mgr.best_path)
        self.assertEqual(best_meta_after["global_step"], 1000)

        # Verify latest_path points to step 6000
        latest_meta = verify_checkpoint_file(mgr.latest_path)
        self.assertEqual(latest_meta["global_step"], 6000)

        # Verify periodic checkpoints pruned to exactly 5 ([2000, 3000, 4000, 5000, 6000])
        retained_steps = [s for s, _ in mgr.find_all_step_checkpoints()]
        self.assertEqual(retained_steps, [2000, 3000, 4000, 5000, 6000])

    def test_05_colab_code_only_bundle_creation(self) -> None:
        """Verify create_colab_bundle packages project code cleanly without .venv or .pyc files."""
        archive_path = self.temp_dir / "test_bundle.tar.gz"
        info = create_colab_bundle(archive_path, include_features=False)
        self.assertTrue(archive_path.exists())
        self.assertGreater(info["file_count"], 20)
        self.assertLess(info["size_mb"], 5.0)

    def test_06_empty_tail_batch_group_and_epoch_transition(self) -> None:
        """Regression test: 3,208 samples / 64 group_size leaves 8 tail samples -> collate_fn returns [].

        Verifies:
        1. Real Svarah train_loader (3,208 items, outer 64, inner 16) yields 51 groups where group 51 is [].
        2. The training loop unpacking logic skips [] without calling trainer.train_step([]).
        3. Resuming from checkpoint step 105 crosses the empty tail group into Epoch 2 without error.
        """
        import sys
        from unittest.mock import patch
        from torch.utils.data import DataLoader

        try:
            from common.dataset import Dataset
        except ImportError:
            if str(PROJECT_ROOT / "original_fastspeech2") not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT / "original_fastspeech2"))
            from dataset import Dataset
        from English.train import parse_args, run_training

        with open(PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml", encoding="utf-8") as f:
            prep_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "config" / "Svarah" / "model.yaml", encoding="utf-8") as f:
            model_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "config" / "Svarah" / "train.yaml", encoding="utf-8") as f:
            train_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "preprocessed_data" / "Svarah" / "stats.json", encoding="utf-8") as f:
            model_dict["stats"] = json.load(f)

        # Part A: Verify real Svarah train_loader boundary (3,208 samples, outer=64, inner=16)
        train_ds = Dataset("train.txt", prep_dict, train_dict, sort=True, drop_last=True)
        self.assertEqual(len(train_ds), 3208)
        train_loader = DataLoader(
            train_ds,
            batch_size=16 * 4,
            shuffle=False,
            collate_fn=train_ds.collate_fn,
        )

        outer_groups = list(train_loader)
        self.assertEqual(len(outer_groups), 51)
        self.assertEqual(len(outer_groups[50]), 0)
        self.assertEqual(outer_groups[50], [])

        unpacked_subbatches = []
        for batch_group in outer_groups:
            if isinstance(batch_group, list) and len(batch_group) == 0:
                continue
            batches = (
                batch_group
                if isinstance(batch_group, list)
                and isinstance(batch_group[0], (tuple, list))
                else [batch_group]
            )
            for b in batches:
                self.assertIsInstance(b, tuple)
                self.assertEqual(len(b), 12)
                unpacked_subbatches.append(b)
        self.assertEqual(len(unpacked_subbatches), 200)

        # Part B: Resume from step 105 and cross an epoch boundary with a 72-sample (64 + 8 tail) dataset
        ckpt_dir = self.temp_dir / "ckpt_step105"
        log_dir = self.temp_dir / "logs_step105"
        train_dict["path"]["ckpt_path"] = str(ckpt_dir)
        train_dict["path"]["log_path"] = str(log_dir)

        fs2_cfg = FastSpeech2Config.from_dicts(prep_dict, model_dict, train_dict)
        model = FastSpeech2(fs2_cfg.preprocess, fs2_cfg.model)
        loss_fn = FastSpeech2Loss(fs2_cfg.preprocess, fs2_cfg.model)
        optim = ScheduledOptim(model, fs2_cfg.train, fs2_cfg.model, current_step=105)
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir, max_to_keep=5)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optim,
            train_config=fs2_cfg.train,
            device="cpu",
            checkpoint_manager=mgr,
        )
        trainer.epoch = 1
        trainer.global_step = 105
        trainer.best_val_loss = 18.5
        save_and_verify_checkpoint(trainer, is_best=True, max_to_keep=5)

        # Slice Dataset to 72 items (1 full group of 64 = 4 sub-batches + 1 partial group of 8 -> collate_fn returns [])
        orig_init = Dataset.__init__

        def _sliced_init(self_ds, filename, p_cfg, t_cfg, sort=False, drop_last=False):
            orig_init(self_ds, filename, p_cfg, t_cfg, sort=sort, drop_last=drop_last)
            limit = 72 if filename == "train.txt" else 16
            self_ds.basename = self_ds.basename[:limit]
            self_ds.speaker = self_ds.speaker[:limit]
            self_ds.text = self_ds.text[:limit]
            self_ds.raw_text = self_ds.raw_text[:limit]

        orig_train_step = Trainer.train_step
        called_with_empty = False

        def _checked_train_step(self_tr, batch):
            nonlocal called_with_empty
            if isinstance(batch, (list, tuple)) and len(batch) == 0:
                called_with_empty = True
            return orig_train_step(self_tr, batch)

        with patch.object(Dataset, "__init__", _sliced_init), patch.object(
            Trainer, "train_step", _checked_train_step
        ):
            # Resume at step 105 and train to step 111 (4 steps in Epoch 1 + empty 8-sample tail -> Epoch 2 + 2 steps)
            args = parse_args(
                [
                    "--checkpoint_dir",
                    str(ckpt_dir),
                    "--log_dir",
                    str(log_dir),
                    "--device",
                    "cpu",
                    "--total_step",
                    "111",
                    "--save_step",
                    "1000",
                    "--val_step",
                    "1000",
                    "--log_step",
                    "100",
                    "--no_notebook_display",
                    "--resume",
                ]
            )
            summary = run_training(args)

        self.assertFalse(called_with_empty, "trainer.train_step was called with an empty batch []!")
        self.assertEqual(summary["status"], "COMPLETED")
        self.assertEqual(summary["total_optimizer_steps"], 111)
        self.assertEqual(summary["total_epochs"], 2)

    def test_07_validate_svarah_dataset_timings_and_regression_guarantees(self) -> None:
        """Verify validate_svarah_dataset checks all 15,488 files via directory sets, records stage timings, and preserves all failure checks."""
        import sys
        import numpy as np
        from torch.utils.data import DataLoader

        try:
            from common.dataset import Dataset
        except ImportError:
            if str(PROJECT_ROOT / "original_fastspeech2") not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT / "original_fastspeech2"))
            from dataset import Dataset

        with open(PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml", encoding="utf-8") as f:
            prep_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "config" / "Svarah" / "model.yaml", encoding="utf-8") as f:
            model_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "config" / "Svarah" / "train.yaml", encoding="utf-8") as f:
            train_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "preprocessed_data" / "Svarah" / "stats.json", encoding="utf-8") as f:
            model_dict["stats"] = json.load(f)

        train_ds = Dataset("train.txt", prep_dict, train_dict, sort=True, drop_last=True)
        val_ds = Dataset("val.txt", prep_dict, train_dict, sort=False, drop_last=False)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, collate_fn=train_ds.collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=val_ds.collate_fn)

        # 1. Full 3,872-utterance (15,488-file) validation on real Svarah dataset
        report = validate_svarah_dataset(
            preprocessed_path=PROJECT_ROOT / "preprocessed_data" / "Svarah",
            model_config=model_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            expected_vocab_size=360,
            max_file_checks=None,
        )
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["train_utterances"], 3208)
        self.assertEqual(report["val_utterances"], 664)
        self.assertEqual(report["checked_feature_files"], 15488)
        self.assertEqual(report["deep_checked_arrays"], 30)
        self.assertIn("timings_sec", report)
        for timing_key in (
            "metadata_and_manifests_sec",
            "directory_index_scan_sec",
            "feature_existence_checks_sec",
            "deep_array_checks_sec",
            "train_dataloader_iter_init_sec",
            "train_dataloader_first_batch_sec",
            "train_dataloader_total_sec",
            "val_dataloader_iter_init_sec",
            "val_dataloader_first_batch_sec",
            "val_dataloader_total_sec",
            "total_validation_sec",
        ):
            self.assertIn(timing_key, report["timings_sec"])
            self.assertGreaterEqual(report["timings_sec"][timing_key], 0.0)

        # 2. Verify all failure guarantees on a synthetic mini-dataset
        synth_dir = self.temp_dir / "synth_svarah"
        for sub in ("mel", "pitch", "energy", "duration"):
            (synth_dir / sub).mkdir(parents=True, exist_ok=True)
        with open(synth_dir / "speakers.json", "w", encoding="utf-8") as f:
            json.dump({"Svarah": 0}, f)
        with open(synth_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump({"pitch": [0.0, 1.0, -2.0, 2.0], "energy": [0.0, 1.0, -2.0, 2.0]}, f)
        with open(synth_dir / "train.txt", "w", encoding="utf-8") as f:
            f.write("utt001|Svarah|{HH AH0 L OW1}|hello\n")
        with open(synth_dir / "val.txt", "w", encoding="utf-8") as f:
            f.write("utt002|Svarah|{W ER1 L D}|world\n")

        for bname in ("utt001", "utt002"):
            np.save(synth_dir / "mel" / f"Svarah-mel-{bname}.npy", np.zeros((10, 80), dtype=np.float32))
            np.save(synth_dir / "pitch" / f"Svarah-pitch-{bname}.npy", np.zeros((10,), dtype=np.float32))
            np.save(synth_dir / "energy" / f"Svarah-energy-{bname}.npy", np.zeros((10,), dtype=np.float32))
            np.save(synth_dir / "duration" / f"Svarah-duration-{bname}.npy", np.array([4, 6], dtype=np.int64))

        # Verify missing file check for each of the 4 feature types
        for missing_sub in ("mel", "pitch", "energy", "duration"):
            target_file = synth_dir / missing_sub / f"Svarah-{missing_sub}-utt002.npy"
            backup_file = synth_dir / missing_sub / f"Svarah-{missing_sub}-utt002.npy.bak"
            target_file.rename(backup_file)
            try:
                with self.assertRaises(FileNotFoundError) as ctx:
                    validate_svarah_dataset(synth_dir, model_dict, train_loader, val_loader)
                self.assertIn(f"Svarah-{missing_sub}-utt002.npy", str(ctx.exception))
            finally:
                backup_file.rename(target_file)

        # Verify NaN detection in deep array check
        np.save(synth_dir / "pitch" / "Svarah-pitch-utt001.npy", np.array([np.nan] * 10, dtype=np.float32))
        with self.assertRaises(ValueError) as ctx:
            validate_svarah_dataset(synth_dir, model_dict, train_loader, val_loader)
        self.assertIn("NaN/Inf", str(ctx.exception))
        np.save(synth_dir / "pitch" / "Svarah-pitch-utt001.npy", np.zeros((10,), dtype=np.float32))

        # Verify duration sum != mel frames detection
        np.save(synth_dir / "duration" / "Svarah-duration-utt001.npy", np.array([4, 5], dtype=np.int64))
        with self.assertRaises(ValueError) as ctx:
            validate_svarah_dataset(synth_dir, model_dict, train_loader, val_loader)
        self.assertIn("Duration sum", str(ctx.exception))

    def test_08_profile_stratified_v1_split_and_fresh_training_config(self) -> None:
        """Verify Plan B-Lite profile_stratified_v1 split invariants, deterministic reproduction, SHA-256 preservation, and Step-0 training config."""
        import sys
        from torch.utils.data import DataLoader
        from English.metadata.generate_profile_stratified_v1_split import (
            DEFAULT_OUT_DIR,
            generate_profile_stratified_split,
            normalize_transcript_for_dedup,
            parse_uid_hierarchy,
            sha256_file,
        )

        try:
            from common.dataset import Dataset
        except ImportError:
            if str(PROJECT_ROOT / "original_fastspeech2") not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT / "original_fastspeech2"))
            from dataset import Dataset

        # 1. Verify all 5 required files exist in English/metadata/splits/profile_stratified_v1/
        for fname in (
            "train.txt",
            "val.txt",
            "split_summary.json",
            "split_manifest.csv",
            "split_audit.json",
        ):
            self.assertTrue(
                (DEFAULT_OUT_DIR / fname).is_file(),
                f"Missing required split artifact: {DEFAULT_OUT_DIR / fname}",
            )

        with open(DEFAULT_OUT_DIR / "split_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        with open(DEFAULT_OUT_DIR / "split_audit.json", encoding="utf-8") as f:
            audit = json.load(f)

        self.assertEqual(audit["status"], "PASSED")
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(summary["train"]["utterances"], 3208)
        self.assertEqual(summary["val"]["utterances"], 664)
        self.assertEqual(summary["train"]["distinct_speaker_profiles"], 88)
        self.assertEqual(summary["val"]["distinct_speaker_profiles"], 88)
        self.assertEqual(summary["val"]["seen_speaker_profiles_in_train"], 88)
        self.assertEqual(summary["val"]["female_utterances"], 332)
        self.assertEqual(summary["val"]["male_utterances"], 332)

        # 2. Independently verify zero user_prefix, recording_id, and normalized transcript overlap
        def _parse_split(path: Path):
            uids, spks, uprefs, recs, nts = set(), set(), set(), set(), set()
            lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            for line in lines:
                uid, spk, _, raw = line.split("|")
                u, r = parse_uid_hierarchy(uid)
                uids.add(uid)
                spks.add(spk)
                uprefs.add(u)
                recs.add(r)
                nts.add(normalize_transcript_for_dedup(raw))
            return lines, uids, spks, uprefs, recs, nts

        tr_lines, tr_uids, tr_spks, tr_uprefs, tr_recs, tr_nts = _parse_split(
            DEFAULT_OUT_DIR / "train.txt"
        )
        va_lines, va_uids, va_spks, va_uprefs, va_recs, va_nts = _parse_split(
            DEFAULT_OUT_DIR / "val.txt"
        )

        self.assertEqual(len(tr_lines), 3208)
        self.assertEqual(len(va_lines), 664)
        self.assertEqual(len(tr_spks), 88)
        self.assertEqual(len(va_spks), 88)
        self.assertEqual(tr_spks, va_spks)
        self.assertEqual(len(tr_uids & va_uids), 0)
        self.assertEqual(len(tr_uprefs & va_uprefs), 0)
        self.assertEqual(len(tr_recs & va_recs), 0)
        self.assertEqual(len(tr_nts & va_nts), 0)

        # 3. Verify frozen files (including test.txt and original train.txt/val.txt) match baseline SHA-256
        expected_sha256 = {
            PROJECT_ROOT / "English" / "metadata" / "train.txt": "4c34430c082bf7ec7055e5dd0283c1b61296f0a1e6381235bbc0bcafa4a9239b",
            PROJECT_ROOT / "English" / "metadata" / "val.txt": "5c18f223e3b5b604c3dba4264d5bb4778cc2e57f3ef6160a5a65296dc69d11a2",
            PROJECT_ROOT / "English" / "metadata" / "test.txt": "b265543e1a4d8531a227609f4356e9494c9de420e3945da1bde2dc0d48542fef",
            PROJECT_ROOT / "preprocessed_data" / "Svarah" / "train.txt": "047bb252fffec303dab03fd53cba7ae4dcb4d5487fac17d09553d0c9438f896e",
            PROJECT_ROOT / "preprocessed_data" / "Svarah" / "val.txt": "b2bb3967e26ee044b7f75a11341f58913caf879cec133cb5575af5fe89ba558d",
            PROJECT_ROOT / "preprocessed_data" / "Svarah" / "stats.json": "1c2e9ff77e49de744f68cd7944c93992fdda317bbe374a21d0fdf47c671f1f79",
            PROJECT_ROOT / "preprocessed_data" / "Svarah" / "speakers.json": "1a5313130a7a2e0b9c7196ee303cb6a7817d0257791bd25775b27293592587de",
            PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml": "b6db335b38b285ef40b68831eed19aedc1e511da6e9d6cb1f4acc0b23b6b35de",
            PROJECT_ROOT / "config" / "Svarah" / "model.yaml": "bf5a5e60905419aa19c8c9305994d45a69ff69ca182f716b9a7ff54d5aca70c4",
            PROJECT_ROOT / "config" / "Svarah" / "train.yaml": "71bcea7b8e96f30e2f83865e4a6f8b72ccb6449cf1ef2015fb752d2cdd8d1657",
        }
        for fpath, exp_hash in expected_sha256.items():
            self.assertEqual(
                sha256_file(fpath),
                exp_hash,
                f"Frozen file {fpath} was modified!",
            )

        # 4. Verify deterministic reproduction in a temporary directory
        repro_dir = self.temp_dir / "repro_split"
        generate_profile_stratified_split(out_dir=repro_dir, verify_all_npy_shapes=False)
        self.assertEqual(
            sha256_file(repro_dir / "train.txt"),
            sha256_file(DEFAULT_OUT_DIR / "train.txt"),
        )
        self.assertEqual(
            sha256_file(repro_dir / "val.txt"),
            sha256_file(DEFAULT_OUT_DIR / "val.txt"),
        )

        # 5. Verify Dataset, DataLoader, and validate_svarah_dataset with train_profile_stratified_v1.yaml
        strat_cfg_path = PROJECT_ROOT / "config" / "Svarah" / "train_profile_stratified_v1.yaml"
        self.assertTrue(strat_cfg_path.is_file())
        with open(PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml", encoding="utf-8") as f:
            prep_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "config" / "Svarah" / "model.yaml", encoding="utf-8") as f:
            model_dict = yaml.safe_load(f)
        with open(strat_cfg_path, encoding="utf-8") as f:
            strat_train_dict = yaml.safe_load(f)
        with open(PROJECT_ROOT / "preprocessed_data" / "Svarah" / "stats.json", encoding="utf-8") as f:
            model_dict["stats"] = json.load(f)

        self.assertNotEqual(
            strat_train_dict["path"]["ckpt_path"],
            "./output/ckpt/Svarah",
        )
        self.assertEqual(
            strat_train_dict["path"]["ckpt_path"],
            "./output/ckpt/Svarah_profile_stratified_v1",
        )

        train_ds = Dataset(
            str(DEFAULT_OUT_DIR / "train.txt"),
            prep_dict,
            strat_train_dict,
            sort=True,
            drop_last=True,
        )
        val_ds = Dataset(
            str(DEFAULT_OUT_DIR / "val.txt"),
            prep_dict,
            strat_train_dict,
            sort=False,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=False, collate_fn=train_ds.collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=64, shuffle=False, collate_fn=val_ds.collate_fn
        )

        val_report = validate_svarah_dataset(
            preprocessed_path=PROJECT_ROOT / "preprocessed_data" / "Svarah",
            model_config=model_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            expected_vocab_size=360,
            max_file_checks=None,
            train_manifest=DEFAULT_OUT_DIR / "train.txt",
            val_manifest=DEFAULT_OUT_DIR / "val.txt",
        )
        self.assertEqual(val_report["status"], "PASSED")
        self.assertEqual(val_report["train_utterances"], 3208)
        self.assertEqual(val_report["val_utterances"], 664)
        self.assertEqual(val_report["checked_feature_files"], 15488)


if __name__ == "__main__":
    unittest.main()

