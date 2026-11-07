"""Validation test suite for English Svarah metadata manifests,

speaker profile mapping, and train/val/test splits.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "English" / "metadata"


class TestSvarahManifests(unittest.TestCase):
    """Automated integrity checks for Svarah manifests and profile mapping."""

    def setUp(self) -> None:
        self.profiles_path = METADATA_DIR / "speaker_profiles.json"
        self.manifest_path = METADATA_DIR / "full_manifest.json"
        self.rejected_path = METADATA_DIR / "rejected_utterances.json"
        self.train_path = METADATA_DIR / "train.txt"
        self.val_path = METADATA_DIR / "val.txt"
        self.test_path = METADATA_DIR / "test.txt"

        self.assertTrue(self.profiles_path.exists(), "speaker_profiles.json missing")
        self.assertTrue(self.manifest_path.exists(), "full_manifest.json missing")
        self.assertTrue(self.rejected_path.exists(), "rejected_utterances.json missing")
        self.assertTrue(self.train_path.exists(), "train.txt missing")
        self.assertTrue(self.val_path.exists(), "val.txt missing")
        self.assertTrue(self.test_path.exists(), "test.txt missing")

    def test_speaker_profiles_integrity(self) -> None:
        """Verify profile IDs are contiguous 0..114 and unique."""
        with open(self.profiles_path, encoding="utf-8") as f:
            data = json.load(f)

        meta = data["metadata"]
        self.assertEqual(meta["n_profiles"], 115)
        self.assertEqual(meta["id_range"], [0, 114])
        self.assertIn("limitation_statement", meta)

        profiles = data["profiles"]
        self.assertEqual(len(profiles), 115)

        ids = [p["speaker_profile_id"] for p in profiles]
        self.assertEqual(ids, list(range(115)), "Profile IDs must be contiguous 0..114")

        keys = [p["profile_key"] for p in profiles]
        self.assertEqual(len(set(keys)), 115, "Profile keys must be unique")
        # Ensure alphabetical sort for stability
        self.assertEqual(keys, sorted(keys), "Profile keys must be alphabetically sorted")

    def test_full_manifest_integrity(self) -> None:
        """Verify full manifest contains 6,656 utterances without duplicates."""
        with open(self.manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(len(manifest), 6656, "Full manifest must have exactly 6,656 rows")

        utterance_ids = [entry["utterance_id"] for entry in manifest]
        self.assertEqual(len(set(utterance_ids)), 6656, "Utterance IDs must be unique")

        # Verify fields and normalized text non-empty
        for entry in manifest:
            self.assertTrue(entry["utterance_id"])
            self.assertTrue(entry["audio_path"].endswith(".wav"))
            self.assertTrue(entry["raw_text"])
            self.assertTrue(entry["normalized_text"].strip())
            self.assertGreater(entry["duration"], 0.0)
            self.assertEqual(entry["sample_rate"], 16000)
            self.assertIn(entry["status"], ("valid", "rejected"))
            self.assertIn(entry["split"], ("train", "val", "test"))
            self.assertTrue(0 <= entry["speaker_profile_id"] < 115)

    def test_duration_filtering_limits(self) -> None:
        """Verify valid utterances strictly satisfy 1.0s <= duration <= 15.0s."""
        with open(self.manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        valid_count = 0
        rejected_count = 0

        for entry in manifest:
            d = entry["duration"]
            if entry["status"] == "valid":
                valid_count += 1
                self.assertGreaterEqual(d, 1.0, f"Valid utterance {entry['utterance_id']} has duration {d} < 1.0s")
                self.assertLessEqual(d, 15.0, f"Valid utterance {entry['utterance_id']} has duration {d} > 15.0s")
                self.assertIsNone(entry["rejection_reason"])
            else:
                rejected_count += 1
                self.assertTrue(d < 1.0 or d > 15.0)
                self.assertIn(entry["rejection_reason"], ("duration_under_1s", "duration_over_15s"))

        self.assertEqual(valid_count, 5180)
        self.assertEqual(rejected_count, 1476)

    def test_rejected_manifest_consistency(self) -> None:
        """Verify rejected_utterances.json matches rejected records in full_manifest."""
        with open(self.rejected_path, encoding="utf-8") as f:
            rejected = json.load(f)

        self.assertEqual(len(rejected), 1476)
        under_1s = sum(1 for r in rejected if r["rejection_reason"] == "duration_under_1s")
        over_15s = sum(1 for r in rejected if r["rejection_reason"] == "duration_over_15s")
        self.assertEqual(under_1s, 1215)
        self.assertEqual(over_15s, 261)

    def test_profile_leakage_and_split_durations(self) -> None:
        """Verify zero profile leakage across train, val, and test splits."""

        def read_split_lines(path: Path) -> list[list[str]]:
            with open(path, encoding="utf-8") as f:
                return [line.strip().split("|") for line in f if line.strip()]

        train_records = read_split_lines(self.train_path)
        val_records = read_split_lines(self.val_path)
        test_records = read_split_lines(self.test_path)

        train_utts = {r[0] for r in train_records}
        val_utts = {r[0] for r in val_records}
        test_utts = {r[0] for r in test_records}

        # Utterance disjointness
        self.assertEqual(len(train_utts & val_utts), 0, "Utterance overlap between train and val")
        self.assertEqual(len(train_utts & test_utts), 0, "Utterance overlap between train and test")
        self.assertEqual(len(val_utts & test_utts), 0, "Utterance overlap between val and test")

        # Profile disjointness (zero leakage)
        train_profiles = {int(r[1]) for r in train_records}
        val_profiles = {int(r[1]) for r in val_records}
        test_profiles = {int(r[1]) for r in test_records}

        self.assertEqual(len(train_profiles & val_profiles), 0, "PROFILE LEAKAGE: Profile in both train and val!")
        self.assertEqual(len(train_profiles & test_profiles), 0, "PROFILE LEAKAGE: Profile in both train and test!")
        self.assertEqual(len(val_profiles & test_profiles), 0, "PROFILE LEAKAGE: Profile in both val and test!")

        all_split_profiles = train_profiles | val_profiles | test_profiles
        self.assertEqual(len(all_split_profiles), 115, "All 115 profiles must be allocated")

        # Duration targets:
        # Train ~ 5 hours (within 1%)
        train_sec = sum(float(r[4]) for r in train_records)
        train_hrs = train_sec / 3600.0
        self.assertAlmostEqual(train_hrs, 5.0, delta=0.05, msg="Train duration must be approx 5 hours")

        # Val ~ 1 hour (within 1%)
        val_sec = sum(float(r[4]) for r in val_records)
        val_hrs = val_sec / 3600.0
        self.assertAlmostEqual(val_hrs, 1.0, delta=0.05, msg="Val duration must be approx 1 hour")

        # Test duration ~ 1.97 hours
        test_sec = sum(float(r[4]) for r in test_records)
        test_hrs = test_sec / 3600.0
        self.assertAlmostEqual(test_hrs, 1.97, delta=0.1)


if __name__ == "__main__":
    unittest.main()

