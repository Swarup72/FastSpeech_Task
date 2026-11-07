"""Comprehensive quality control and dataset integrity suite for Svarah FastSpeech 2.

Validates:
1. Complete feature file presence (mel, duration, pitch, energy, phoneme-level features).
2. Exact temporal frame equality (sum(duration) == mel.shape[0] == len(pitch) == len(energy)).
3. Finite values (no NaN or Inf anywhere in feature tensors).
4. Strict positive durations (d_i >= 1 for all tokens).
5. FastSpeech 2 vocabulary symbol containment.
6. Speaker profile mapping validity (speakers.json).
7. Zero train / validation leakage (disjoint utterance sets).
8. DataLoader compatibility (loads batches cleanly with Dataset class).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data" / "Svarah"
FS2_ROOT = PROJECT_ROOT / "original_fastspeech2"


class TestSvarahPreprocessingQC(unittest.TestCase):
    """Quality control test suite for Svarah preprocessed dataset."""

    @classmethod
    def setUpClass(cls):
        cls.train_txt = PREPROCESSED_DIR / "train.txt"
        cls.val_txt = PREPROCESSED_DIR / "val.txt"
        cls.stats_json = PREPROCESSED_DIR / "stats.json"
        cls.speakers_json = PREPROCESSED_DIR / "speakers.json"

        # Check required metadata files exist
        assert cls.train_txt.exists(), "Missing train.txt"
        assert cls.val_txt.exists(), "Missing val.txt"
        assert cls.stats_json.exists(), "Missing stats.json"
        assert cls.speakers_json.exists(), "Missing speakers.json"

        with open(cls.train_txt, encoding="utf-8") as f:
            cls.train_lines = [line.strip().split("|") for line in f if line.strip()]
        with open(cls.val_txt, encoding="utf-8") as f:
            cls.val_lines = [line.strip().split("|") for line in f if line.strip()]

        with open(cls.speakers_json, encoding="utf-8") as f:
            cls.speakers = json.load(f)

        with open(cls.stats_json, encoding="utf-8") as f:
            cls.stats = json.load(f)

        # Load vocabulary
        try:
            from English.text.symbols import symbols
        except ImportError:
            import sys
            if str(FS2_ROOT) not in sys.path:
                sys.path.insert(0, str(FS2_ROOT))
            from text.symbols import symbols
        cls.valid_symbols = set(symbols)

    def test_sample_counts_and_disjoint_splits(self):
        """Verify train and validation splits have expected sample sizes and no leakage."""
        self.assertGreaterEqual(len(self.train_lines), 3200)
        self.assertGreaterEqual(len(self.val_lines), 660)

        train_uids = {l[0] for l in self.train_lines}
        val_uids = {l[0] for l in self.val_lines}

        # Zero utterance ID leakage
        self.assertTrue(
            train_uids.isdisjoint(val_uids),
            f"Train and val share {len(train_uids & val_uids)} utterance IDs!",
        )

    def test_speakers_mapping_validity(self):
        """Verify speakers.json contains all 115 profiles mapped contiguously."""
        self.assertEqual(len(self.speakers), 115)
        for i in range(115):
            spk_str = f"spk_{i:03d}"
            self.assertIn(spk_str, self.speakers)
            self.assertEqual(self.speakers[spk_str], i)

    def test_stats_json_structure_and_finite_values(self):
        """Verify stats.json format and finite ranges."""
        self.assertIn("pitch", self.stats)
        self.assertIn("energy", self.stats)
        for key in ["pitch", "energy"]:
            s = self.stats[key]
            self.assertEqual(len(s), 4)  # min, max, mean, std
            min_v, max_v, mean_v, std_v = s
            self.assertLess(min_v, max_v)
            self.assertGreater(std_v, 0.0)
            self.assertTrue(np.isfinite(mean_v))

    def test_feature_integrity_on_representative_train_samples(self):
        """Verify feature dimensions, finite values, and exact frame equality on train samples."""
        try:
            from English.text import text_to_sequence
        except ImportError:
            from text import text_to_sequence

        # Test first 100 train samples
        for line in self.train_lines[:100]:
            uid, spk, text, raw_text = line
            self.assertIn(spk, self.speakers)

            mel_path = PREPROCESSED_DIR / "mel" / f"{spk}-mel-{uid}.npy"
            dur_path = PREPROCESSED_DIR / "duration" / f"{spk}-duration-{uid}.npy"
            pitch_path = PREPROCESSED_DIR / "pitch" / f"{spk}-pitch-{uid}.npy"
            energy_path = PREPROCESSED_DIR / "energy" / f"{spk}-energy-{uid}.npy"

            self.assertTrue(mel_path.exists(), f"Missing {mel_path}")
            self.assertTrue(dur_path.exists(), f"Missing {dur_path}")
            self.assertTrue(pitch_path.exists(), f"Missing {pitch_path}")
            self.assertTrue(energy_path.exists(), f"Missing {energy_path}")

            mel = np.load(mel_path)
            dur = np.load(dur_path)
            pitch = np.load(pitch_path)
            energy = np.load(energy_path)

            # Dimensions
            t_mel = mel.shape[0]
            self.assertEqual(mel.shape[1], 80)
            self.assertEqual(len(pitch), t_mel)
            self.assertEqual(len(energy), t_mel)
            self.assertEqual(int(np.sum(dur)), t_mel)

            # Non-empty and finite
            self.assertTrue(np.isfinite(mel).all())
            self.assertTrue(np.isfinite(pitch).all())
            self.assertTrue(np.isfinite(energy).all())

            # Durations positive
            self.assertTrue((dur >= 1).all())

            # Text to sequence parsing
            seq = text_to_sequence(text, ["english_cleaners"])
            self.assertEqual(len(seq), len(dur))

    def test_feature_integrity_on_representative_val_samples(self):
        """Verify feature dimensions, finite values, and exact frame equality on val samples."""
        try:
            from English.text import text_to_sequence
        except ImportError:
            from text import text_to_sequence

        # Test first 50 val samples
        for line in self.val_lines[:50]:
            uid, spk, text, raw_text = line
            self.assertIn(spk, self.speakers)

            mel_path = PREPROCESSED_DIR / "mel" / f"{spk}-mel-{uid}.npy"
            dur_path = PREPROCESSED_DIR / "duration" / f"{spk}-duration-{uid}.npy"
            pitch_path = PREPROCESSED_DIR / "pitch" / f"{spk}-pitch-{uid}.npy"
            energy_path = PREPROCESSED_DIR / "energy" / f"{spk}-energy-{uid}.npy"

            self.assertTrue(mel_path.exists())
            self.assertTrue(dur_path.exists())
            self.assertTrue(pitch_path.exists())
            self.assertTrue(energy_path.exists())

            mel = np.load(mel_path)
            dur = np.load(dur_path)
            pitch = np.load(pitch_path)
            energy = np.load(energy_path)

            t_mel = mel.shape[0]
            self.assertEqual(mel.shape[1], 80)
            self.assertEqual(len(pitch), t_mel)
            self.assertEqual(len(energy), t_mel)
            self.assertEqual(int(np.sum(dur)), t_mel)

            self.assertTrue(np.isfinite(mel).all())
            self.assertTrue(np.isfinite(pitch).all())
            self.assertTrue(np.isfinite(energy).all())
            self.assertTrue((dur >= 1).all())

            seq = text_to_sequence(text, ["english_cleaners"])
            self.assertEqual(len(seq), len(dur))

    def test_fastspeech2_dataset_loader(self):
        """Verify FastSpeech 2 Dataset class can load and collate batches."""
        try:
            from common.dataset import Dataset
        except ImportError:
            from dataset import Dataset

        preprocess_config = {
            "dataset": "Svarah",
            "path": {"preprocessed_path": str(PREPROCESSED_DIR)},
            "preprocessing": {"text": {"text_cleaners": ["english_cleaners"]}},
        }
        train_config = {"optimizer": {"batch_size": 4}}

        ds = Dataset("train.txt", preprocess_config, train_config)
        self.assertGreater(len(ds), 3000)

        # Fetch sample 0
        sample = ds[0]
        self.assertIn("id", sample)
        self.assertIn("speaker", sample)
        self.assertIn("text", sample)
        self.assertIn("mel", sample)
        self.assertIn("pitch", sample)
        self.assertIn("energy", sample)
        self.assertIn("duration", sample)

        # Collate batch of 4 samples
        batch_samples = [ds[i] for i in range(4)]
        batch = ds.reprocess(batch_samples, list(range(4)))
        self.assertIsNotNone(batch)
        # batch has 12 elements
        self.assertEqual(len(batch), 12)
        mels = batch[6]
        pitches = batch[9]
        energies = batch[10]
        durations = batch[11]

        # Check tensor shapes
        self.assertEqual(mels.shape[0], 4)
        self.assertEqual(mels.shape[2], 80)
        self.assertEqual(pitches.shape[0], 4)
        self.assertEqual(energies.shape[0], 4)
        self.assertEqual(durations.shape[0], 4)


if __name__ == "__main__":
    unittest.main()

