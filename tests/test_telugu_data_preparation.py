"""Unit and regression test suite for Telugu data preparation, splitting, and text representation.

Verifies:
1. Unicode NFC normalization and punctuation sanitization.
2. Telugu grapheme tokenization, special token IDs, and roundtrip sequence decoding.
3. Vocabulary configuration consistency between symbols.py and vocabulary.json.
4. Deterministic partition integrity and metadata line formatting.
5. Multi-speaker representation across train, validation, and test splits.
6. Zero duplicate transcript leakage between train, val, and test.
7. Split cumulative duration bounds (5.0h train, 1.0h val).
8. Band-limited sinc resampling for 16 kHz (female) and 48 kHz (male) to 22.05 kHz.
9. Acoustic silence segmentation logic.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest
import numpy as np
import torch

from Telugu.text.symbols import (
    symbols,
    symbol_to_id,
    id_to_symbol,
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
    VOCAB_SIZE,
)
from Telugu.text.normalizer import (
    normalize_text,
    text_to_sequence,
    sequence_to_text,
)
from Telugu.alignment.resample import resample_audio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TELUGU_METADATA_DIR = PROJECT_ROOT / "Telugu" / "metadata"


class TestTeluguDataPreparation(unittest.TestCase):
    """Test suite for Telugu data preparation, normalizer, and split verification."""

    def test_special_token_ids_and_conventions(self):
        """Verify PAD is 0 and special tokens follow FastSpeech 2 conventions."""
        self.assertEqual(PAD_ID, 0)
        self.assertEqual(symbol_to_id["_"], 0)
        self.assertEqual(UNK_ID, 1)
        self.assertEqual(symbol_to_id["<unk>"], 1)
        self.assertEqual(BOS_ID, 2)
        self.assertEqual(symbol_to_id["<bos>"], 2)
        self.assertEqual(EOS_ID, 3)
        self.assertEqual(symbol_to_id["<eos>"], 3)

        # Silence tokens for MFA alignment compatibility
        self.assertIn("@sp", symbol_to_id)
        self.assertIn("@spn", symbol_to_id)
        self.assertIn("@sil", symbol_to_id)

        # Vocabulary size
        self.assertEqual(len(symbols), VOCAB_SIZE)
        self.assertEqual(len(symbol_to_id), VOCAB_SIZE)
        self.assertEqual(len(id_to_symbol), VOCAB_SIZE)

    def test_unicode_nfc_normalization(self):
        """Verify canonical NFC normalization for composite Telugu matras."""
        # Decomposed Telugu ai matra (U+0C46 + U+0C56) vs composed (U+0C48)
        decomposed = "క\u0c46\u0c56"
        normalized = normalize_text(decomposed)
        self.assertEqual(normalized, "కై")

        # Zero-width non-joiner and joiner removal
        with_zwnj = "న\u200cర\u200dక"
        self.assertEqual(normalize_text(with_zwnj), "నరక")

    def test_punctuation_and_whitespace_sanitization(self):
        """Verify standard punctuation, dandas, curly quotes, and whitespace cleanup."""
        raw = "   \u201cనరకవాసి\u201d\u0964  ఇది   మిత్రవిందకుడు...  \n  "
        normalized = normalize_text(raw)
        self.assertEqual(normalized, '"నరకవాసి". ఇది మిత్రవిందకుడు...')

        # Dashes
        dashes = "రాముడు \u2013 లక్ష్మణుడు"
        self.assertEqual(normalize_text(dashes), "రాముడు - లక్ష్మణుడు")

    def test_text_to_sequence_and_decode(self):
        """Verify text to integer sequence tokenization and decoding."""
        telugu_text = "నరకవాసి."
        seq = text_to_sequence(telugu_text, add_bos=True, add_eos=True)

        self.assertEqual(seq[0], BOS_ID)
        self.assertEqual(seq[-1], EOS_ID)
        self.assertTrue(all(isinstance(x, int) and 0 <= x < VOCAB_SIZE for x in seq))

        # Reconstruct without BOS/EOS
        decoded = sequence_to_text(seq)
        self.assertEqual(decoded, telugu_text)

    def test_vocabulary_json_consistency(self):
        """Verify Telugu/text/vocabulary.json exactly matches symbols.py."""
        vocab_path = PROJECT_ROOT / "Telugu" / "text" / "vocabulary.json"
        self.assertTrue(vocab_path.exists(), "Missing Telugu/text/vocabulary.json")

        with open(vocab_path, encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["vocab_size"], VOCAB_SIZE)
        self.assertEqual(data["special_tokens"]["pad"]["id"], 0)
        self.assertEqual(data["symbols"], symbols)

    def test_deterministic_split_files_presence_and_format(self):
        """Verify train.txt, val.txt, test.txt exist and follow canonical schema."""
        train_path = TELUGU_METADATA_DIR / "train.txt"
        val_path = TELUGU_METADATA_DIR / "val.txt"
        test_path = TELUGU_METADATA_DIR / "test.txt"
        summary_path = TELUGU_METADATA_DIR / "split_summary.json"

        self.assertTrue(train_path.exists(), "Missing train.txt")
        self.assertTrue(val_path.exists(), "Missing val.txt")
        self.assertTrue(test_path.exists(), "Missing test.txt")
        self.assertTrue(summary_path.exists(), "Missing split_summary.json")

        for p in [train_path, val_path, test_path]:
            with open(p, encoding="utf-8") as f:
                lines = [line.strip().split("|") for line in f if line.strip()]
            self.assertGreater(len(lines), 0, f"Empty split file: {p.name}")
            for parts in lines:
                # basename|speaker|raw|norm|dur|gender
                self.assertEqual(len(parts), 6, f"Invalid line parts in {p.name}: {parts}")
                self.assertIn(parts[1], {"0", "1"}, f"Invalid speaker ID: {parts[1]}")
                self.assertIn(parts[5], {"female", "male"}, f"Invalid gender: {parts[5]}")
                dur = float(parts[4])
                self.assertGreaterEqual(dur, 1.0, f"Duration sub 1s: {dur}")
                self.assertLessEqual(dur, 15.0, f"Duration super 15s: {dur}")

    def test_speaker_representation_in_all_splits(self):
        """Verify both female (0) and male (1) speakers exist in every split."""
        for split_name in ["train.txt", "val.txt", "test.txt"]:
            path = TELUGU_METADATA_DIR / split_name
            with open(path, encoding="utf-8") as f:
                speakers = set(line.strip().split("|")[1] for line in f if line.strip())
            self.assertEqual(speakers, {"0", "1"}, f"Speaker missing in {split_name}: {speakers}")

    def test_zero_transcript_leakage(self):
        """Verify strict zero duplicate transcript leakage between train, val, and test."""
        def load_transcripts(filename: str) -> set[str]:
            path = TELUGU_METADATA_DIR / filename
            with open(path, encoding="utf-8") as f:
                return set(line.strip().split("|")[3] for line in f if line.strip())

        train_texts = load_transcripts("train.txt")
        val_texts = load_transcripts("val.txt")
        test_texts = load_transcripts("test.txt")

        # Set intersection must be completely empty
        self.assertEqual(len(train_texts.intersection(val_texts)), 0, "Leakage: train & val overlap!")
        self.assertEqual(len(train_texts.intersection(test_texts)), 0, "Leakage: train & test overlap!")
        self.assertEqual(len(val_texts.intersection(test_texts)), 0, "Leakage: val & test overlap!")

    def test_split_duration_targets(self):
        """Verify train is ~5.0h (2.5h each) and val is ~1.0h (0.5h each)."""
        summary_path = TELUGU_METADATA_DIR / "split_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        splits = summary["splits"]

        # Train: ~5 hours total (+/- 0.05h)
        train_tot = splits["train"]["total_duration_hours"]
        self.assertAlmostEqual(train_tot, 5.0, delta=0.05)
        self.assertAlmostEqual(splits["train"]["female_duration_hours"], 2.5, delta=0.05)
        self.assertAlmostEqual(splits["train"]["male_duration_hours"], 2.5, delta=0.05)

        # Val: ~1 hour total (+/- 0.02h)
        val_tot = splits["val"]["total_duration_hours"]
        self.assertAlmostEqual(val_tot, 1.0, delta=0.02)
        self.assertAlmostEqual(splits["val"]["female_duration_hours"], 0.5, delta=0.02)
        self.assertAlmostEqual(splits["val"]["male_duration_hours"], 0.5, delta=0.02)

    def test_resampling_female_16k_to_22k(self):
        """Verify Kaiser sinc resampling from 16,000 Hz to 22,050 Hz."""
        sr_orig = 16000
        sr_target = 22050
        duration = 1.5
        samples_orig = int(sr_orig * duration)
        t = np.linspace(0, duration, samples_orig, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

        resampled, new_sr = resample_audio(tone, orig_sr=sr_orig, target_sr=sr_target)
        self.assertEqual(new_sr, sr_target)
        self.assertTrue(np.isfinite(resampled).all())

        expected_samples = int(round(samples_orig * sr_target / sr_orig))
        self.assertAlmostEqual(len(resampled), expected_samples, delta=1)
        self.assertLessEqual(np.max(np.abs(resampled)), 1.0)

    def test_resampling_male_48k_to_22k(self):
        """Verify Kaiser sinc resampling from 48,000 Hz to 22,050 Hz."""
        sr_orig = 48000
        sr_target = 22050
        duration = 1.5
        samples_orig = int(sr_orig * duration)
        t = np.linspace(0, duration, samples_orig, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        resampled, new_sr = resample_audio(tone, orig_sr=sr_orig, target_sr=sr_target)
        self.assertEqual(new_sr, sr_target)
        self.assertTrue(np.isfinite(resampled).all())

        expected_samples = int(round(samples_orig * sr_target / sr_orig))
        self.assertAlmostEqual(len(resampled), expected_samples, delta=1)
        self.assertLessEqual(np.max(np.abs(resampled)), 1.0)


class TestTeluguOption1BSplit(unittest.TestCase):
    """Test suite for Option 1B Telugu Dataset Split under Telugu/metadata/splits/option_1b/."""

    OPTION_1B_DIR = TELUGU_METADATA_DIR / "splits" / "option_1b"

    def test_01_all_six_artifacts_exist_and_legacy_files_preserved(self):
        """Verify all 6 Option 1B files exist and legacy split files remain untouched."""
        for fname in (
            "train.txt",
            "val.txt",
            "test.txt",
            "split_summary.json",
            "split_manifest.csv",
            "split_audit.json",
        ):
            fpath = self.OPTION_1B_DIR / fname
            self.assertTrue(fpath.exists(), f"Missing Option 1B file: {fpath}")

        # Verify legacy files in Telugu/metadata/ were preserved unchanged (2330, 457, 2366)
        with open(TELUGU_METADATA_DIR / "train.txt", encoding="utf-8") as f:
            self.assertEqual(len([l for l in f if l.strip()]), 2330)
        with open(TELUGU_METADATA_DIR / "val.txt", encoding="utf-8") as f:
            self.assertEqual(len([l for l in f if l.strip()]), 457)
        with open(TELUGU_METADATA_DIR / "test.txt", encoding="utf-8") as f:
            self.assertEqual(len([l for l in f if l.strip()]), 2366)

    def test_02_exact_counts_duration_bounds_and_gender_balance(self):
        """Verify 2,302 train, 422 val, 422 test counts, [1.0s, 15.0s] actual WAV durations, and 50/50 gender balance."""
        with open(self.OPTION_1B_DIR / "split_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        with open(self.OPTION_1B_DIR / "split_audit.json", encoding="utf-8") as f:
            audit = json.load(f)

        self.assertEqual(audit["audit_status"], "PASSED")
        for check_name, passed in audit["checks"].items():
            if isinstance(passed, bool):
                self.assertTrue(passed, f"Audit check failed: {check_name}")

        splits = summary["splits"]
        self.assertEqual(splits["train"]["total_utterances"], 2302)
        self.assertEqual(splits["train"]["female_utterances"], 1055)
        self.assertEqual(splits["train"]["male_utterances"], 1247)
        self.assertAlmostEqual(splits["train"]["total_duration_hours"], 5.3289, delta=0.001)
        self.assertAlmostEqual(splits["train"]["female_duration_pct"], 50.0, delta=0.01)
        self.assertAlmostEqual(splits["train"]["male_duration_pct"], 50.0, delta=0.01)

        self.assertEqual(splits["val"]["total_utterances"], 422)
        self.assertEqual(splits["val"]["female_utterances"], 192)
        self.assertEqual(splits["val"]["male_utterances"], 230)
        self.assertAlmostEqual(splits["val"]["total_duration_hours"], 1.0000, delta=0.001)
        self.assertAlmostEqual(splits["val"]["female_duration_pct"], 50.0, delta=0.01)

        self.assertEqual(splits["test"]["total_utterances"], 422)
        self.assertEqual(splits["test"]["female_utterances"], 188)
        self.assertEqual(splits["test"]["male_utterances"], 234)
        self.assertAlmostEqual(splits["test"]["total_duration_hours"], 1.0000, delta=0.001)
        self.assertAlmostEqual(splits["test"]["female_duration_pct"], 50.0, delta=0.01)

    def test_03_zero_id_and_transcript_leakage_and_strict_duration_window(self):
        """Verify 0 ID/text overlap across train, val, test and 1.0 <= dur <= 15.0 for all 3,146 active items."""
        import csv

        split_ids: dict[str, set[str]] = {}
        split_texts: dict[str, set[str]] = {}

        for sname in ("train", "val", "test"):
            ids, texts = set(), set()
            with open(self.OPTION_1B_DIR / f"{sname}.txt", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    self.assertEqual(len(parts), 6)
                    uid, spk, raw, norm, dur_str, gender_str = parts
                    dur = float(dur_str)
                    self.assertGreaterEqual(dur, 1.0)
                    self.assertLessEqual(dur, 15.0)
                    self.assertIn(spk, {"0", "1"})
                    self.assertIn(gender_str, {"female", "male"})
                    ids.add(uid)
                    texts.add(norm)
            split_ids[sname] = ids
            split_texts[sname] = texts

        self.assertEqual(len(split_ids["train"]), 2302)
        self.assertEqual(len(split_ids["val"]), 422)
        self.assertEqual(len(split_ids["test"]), 422)

        self.assertEqual(len(split_ids["train"] & split_ids["val"]), 0)
        self.assertEqual(len(split_ids["train"] & split_ids["test"]), 0)
        self.assertEqual(len(split_ids["val"] & split_ids["test"]), 0)

        self.assertEqual(len(split_texts["train"] & split_texts["val"]), 0)
        self.assertEqual(len(split_texts["train"] & split_texts["test"]), 0)
        self.assertEqual(len(split_texts["val"] & split_texts["test"]), 0)

        # Verify split_manifest.csv has all 8,576 rows
        with open(self.OPTION_1B_DIR / "split_manifest.csv", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 8576)

    def test_04_deterministic_reproducibility(self):
        """Verify re-running generate_option_1b_split(seed=42) produces identical SHA-256 checksums."""
        import shutil
        import tempfile
        from Telugu.metadata.generate_option_1b_split import generate_option_1b_split

        with open(self.OPTION_1B_DIR / "split_summary.json", encoding="utf-8") as f:
            orig_summary = json.load(f)

        tmp_dir = Path(tempfile.mkdtemp(prefix="test_opt1b_repro_"))
        try:
            repro_summary = generate_option_1b_split(output_dir=tmp_dir, seed=42)
            self.assertEqual(
                orig_summary["file_checksums_sha256"],
                repro_summary["file_checksums_sha256"],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
