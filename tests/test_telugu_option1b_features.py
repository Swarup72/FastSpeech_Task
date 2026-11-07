"""Comprehensive unit and integration tests for Telugu Option 1B acoustic feature extraction.

Covers:
1. Mel feature shape ([T_mel, 80], float32, finite) across all 2,721 utterances.
2. Pitch and energy validity (frame-level [T_mel] and phoneme-level [L_phone], finite, non-constant, unvoiced interpolation).
3. Duration consistency (sum(duration) == T_mel == len(pitch) == len(energy), min >= 1, len(duration) == len(phoneme_sequence)).
4. NaN/Inf detection and dimension mismatch rejection in validate_utterance_feature_files.
5. Manifest consistency (2,300 train + 421 val, disjoint IDs, Latin-o normalization preservation, FastSpeech 2 Dataset compatibility).
6. Normalization statistics (stats.json computed strictly on training data, zero validation leakage).
7. Resume behavior (idempotent skipping of valid files without overwriting mtime, selective recovery of deleted file).
8. Exclusion of failed alignments (train_telugufemale_00930, train_telugufemale_00928, train_telugumale_02390).
9. Strict exclusion of held-out test data (test.txt never accessed or processed).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

try:
    from common.dataset import Dataset, TextDataset
except ImportError:
    from original_fastspeech2.dataset import Dataset, TextDataset
from Telugu.alignment.preprocess_features import (
    DEFAULT_OUT_DIR,
    EXCLUDED_FAILED_IDS,
    HOP_LENGTH,
    N_MEL,
    OPTION_1B_SPLIT_DIR,
    SAMPLING_RATE,
    SPEAKERS_MAP,
    TEXTGRID_ROOT_DIR,
    TeluguFeatureExtractor,
    _load_option_1b_split_manifest,
    _load_parquet_audio_for_ids,
    extract_telugu_durations_from_textgrid,
    remove_outliers,
    run_telugu_option_1b_preprocessing,
    validate_utterance_feature_files,
)
from Telugu.text.normalizer import text_to_sequence
from Telugu.text.symbols import UNK_ID, VOCAB_SIZE


class TestTeluguOption1BFeatureExtraction(unittest.TestCase):
    """Verify the Telugu Option 1B preprocessed acoustic dataset and extraction pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.out_dir = DEFAULT_OUT_DIR
        cls.split_dir = OPTION_1B_SPLIT_DIR
        cls.tg_root = TEXTGRID_ROOT_DIR

        with open(cls.out_dir / "extraction_summary.json", encoding="utf-8") as f:
            cls.summary = json.load(f)
        with open(cls.out_dir / "stats.json", encoding="utf-8") as f:
            cls.stats = json.load(f)
        with open(cls.out_dir / "speakers.json", encoding="utf-8") as f:
            cls.speakers = json.load(f)
        with open(cls.out_dir / "extraction_failures.json", encoding="utf-8") as f:
            cls.failures = json.load(f)

        with open(cls.out_dir / "train.txt", encoding="utf-8") as f:
            cls.train_lines = [line.strip() for line in f if line.strip()]
        with open(cls.out_dir / "val.txt", encoding="utf-8") as f:
            cls.val_lines = [line.strip() for line in f if line.strip()]

    def test_01_mel_feature_shapes_and_dtypes_all_2721_utterances(self) -> None:
        """Every mel file must have shape [T_mel, 80], dtype float32, T_mel > 0, and finite values."""
        mel_dir = self.out_dir / "mel"
        mel_files = sorted(mel_dir.glob("*.npy"))
        self.assertEqual(len(mel_files), 2721)

        for line in self.train_lines + self.val_lines:
            uid, spk_str, _, _ = line.split("|")
            mel_path = mel_dir / f"{spk_str}-mel-{uid}.npy"
            self.assertTrue(mel_path.is_file(), f"Missing mel file: {mel_path}")
            mel = np.load(mel_path)
            self.assertEqual(mel.ndim, 2, f"Mel ndim != 2 for {uid}")
            self.assertEqual(mel.shape[1], N_MEL, f"Mel bins != 80 for {uid}")
            self.assertGreater(mel.shape[0], 0, f"Empty mel frames for {uid}")
            self.assertEqual(mel.dtype, np.float32, f"Mel dtype != float32 for {uid}")
            self.assertTrue(np.isfinite(mel).all(), f"NaN/Inf in mel for {uid}")

    def test_02_pitch_and_energy_validity_and_unvoiced_interpolation(self) -> None:
        """Frame and phoneme pitch/energy arrays must be finite, aligned, and interpolate unvoiced frames."""
        for sub in ("pitch", "energy", "pitch_phoneme", "energy_phoneme"):
            files = sorted((self.out_dir / sub).glob("*.npy"))
            self.assertEqual(len(files), 2721, f"Expected 2721 files in {sub}/, found {len(files)}")

        pitch_stds = []
        energy_stds = []
        for line in (self.train_lines[:50] + self.val_lines[:50]):
            uid, spk_str, fs2_text, _ = line.split("|")
            mel = np.load(self.out_dir / "mel" / f"{spk_str}-mel-{uid}.npy")
            pitch = np.load(self.out_dir / "pitch" / f"{spk_str}-pitch-{uid}.npy")
            energy = np.load(self.out_dir / "energy" / f"{spk_str}-energy-{uid}.npy")
            dur = np.load(self.out_dir / "duration" / f"{spk_str}-duration-{uid}.npy")
            p_ph = np.load(self.out_dir / "pitch_phoneme" / f"{spk_str}-pitch-{uid}.npy")
            e_ph = np.load(self.out_dir / "energy_phoneme" / f"{spk_str}-energy-{uid}.npy")

            t_mel = mel.shape[0]
            l_phone = len(dur)
            self.assertEqual(pitch.shape, (t_mel,))
            self.assertEqual(energy.shape, (t_mel,))
            self.assertEqual(p_ph.shape, (l_phone,))
            self.assertEqual(e_ph.shape, (l_phone,))
            self.assertEqual(pitch.dtype, np.float32)
            self.assertEqual(energy.dtype, np.float32)
            self.assertTrue(np.isfinite(pitch).all())
            self.assertTrue(np.isfinite(energy).all())
            self.assertTrue(np.isfinite(p_ph).all())
            self.assertTrue(np.isfinite(e_ph).all())
            pitch_stds.append(float(np.std(pitch)))
            energy_stds.append(float(np.std(energy)))

        self.assertGreater(float(np.mean(pitch_stds)), 0.1)
        self.assertGreater(float(np.mean(energy_stds)), 0.1)

        # Direct unit test of unvoiced frame interpolation on real utterance with unvoiced segments
        sample_uid = "train_telugufemale_01230"
        tg_path = self.tg_root / "train" / f"{sample_uid}.TextGrid"
        audio_map = _load_parquet_audio_for_ids({sample_uid})
        import io
        import soundfile as sf
        from Telugu.alignment.resample import resample_audio

        wav, orig_sr = sf.read(io.BytesIO(audio_map[sample_uid]), dtype="float32")
        wav_22k, _ = resample_audio(wav, orig_sr, SAMPLING_RATE)
        extractor = TeluguFeatureExtractor()
        raw_res = extractor.extract_from_waveform_and_textgrid(wav_22k, tg_path)
        # Raw F0 has unvoiced (0.0) frames, whereas interpolated pitch_frame is strictly positive everywhere
        self.assertTrue((raw_res["raw_f0"] == 0.0).any())
        self.assertTrue((raw_res["pitch_frame"] > 0.0).all())
        self.assertTrue(np.isfinite(raw_res["pitch_frame"]).all())

    def test_03_duration_consistency_across_all_2721_utterances(self) -> None:
        """Verify sum(duration) == mel_frames == len(pitch) == len(energy) and len(duration) == len(phoneme_seq) for all 2,721 utterances."""
        dur_files = sorted((self.out_dir / "duration").glob("*.npy"))
        self.assertEqual(len(dur_files), 2721)

        for line in self.train_lines + self.val_lines:
            uid, spk_str, fs2_text, _ = line.split("|")
            ok, err, meta = validate_utterance_feature_files(
                self.out_dir, uid, spk_str, fs2_text
            )
            self.assertTrue(ok, f"Validation failed for {uid}: {err}")
            self.assertIsNotNone(meta)

        # Also test extract_telugu_durations_from_textgrid frame reconciliation behavior
        sample_tg = self.tg_root / "train" / "train_telugufemale_01230.TextGrid"
        raw_dur_res = extract_telugu_durations_from_textgrid(sample_tg, target_mel_length=None)
        raw_total = raw_dur_res.total_frames

        rec_down = extract_telugu_durations_from_textgrid(sample_tg, target_mel_length=raw_total - 2)
        self.assertTrue(rec_down.reconciliation_applied)
        self.assertEqual(rec_down.total_frames, raw_total - 2)
        self.assertTrue(all(d >= 1 for d in rec_down.durations))

        rec_up = extract_telugu_durations_from_textgrid(sample_tg, target_mel_length=raw_total + 3)
        self.assertTrue(rec_up.reconciliation_applied)
        self.assertEqual(rec_up.total_frames, raw_total + 3)
        self.assertTrue(all(d >= 1 for d in rec_up.durations))

    def test_04_nan_inf_and_corruption_detection(self) -> None:
        """Verify validate_utterance_feature_files rejects NaN, Inf, missing files, and shape mismatches."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for sub in ("mel", "pitch", "energy", "duration", "pitch_phoneme", "energy_phoneme"):
                (tmp_dir / sub).mkdir(parents=True)

            uid = "test_dummy_0001"
            spk = "spk_000"
            fs2_text = "{న ర క వ ా స ి}"
            seq = text_to_sequence(fs2_text)
            n_phones = len(seq)
            dur = np.full(n_phones, 4, dtype=np.int32)
            t_mel = int(np.sum(dur))

            def write_valid() -> None:
                np.save(tmp_dir / "mel" / f"{spk}-mel-{uid}.npy", np.zeros((t_mel, 80), dtype=np.float32))
                np.save(tmp_dir / "pitch" / f"{spk}-pitch-{uid}.npy", np.zeros(t_mel, dtype=np.float32))
                np.save(tmp_dir / "energy" / f"{spk}-energy-{uid}.npy", np.zeros(t_mel, dtype=np.float32))
                np.save(tmp_dir / "duration" / f"{spk}-duration-{uid}.npy", dur)
                np.save(tmp_dir / "pitch_phoneme" / f"{spk}-pitch-{uid}.npy", np.zeros(n_phones, dtype=np.float32))
                np.save(tmp_dir / "energy_phoneme" / f"{spk}-energy-{uid}.npy", np.zeros(n_phones, dtype=np.float32))

            write_valid()
            ok, err, _ = validate_utterance_feature_files(tmp_dir, uid, spk, fs2_text)
            self.assertTrue(ok, f"Expected valid files to pass, got {err}")

            # Inject NaN into mel
            bad_mel = np.zeros((t_mel, 80), dtype=np.float32)
            bad_mel[2, 10] = np.nan
            np.save(tmp_dir / "mel" / f"{spk}-mel-{uid}.npy", bad_mel)
            ok, err, _ = validate_utterance_feature_files(tmp_dir, uid, spk, fs2_text)
            self.assertFalse(ok)
            self.assertEqual(err, "nan_or_inf_in_mel")

            # Inject Inf into pitch
            write_valid()
            bad_pitch = np.zeros(t_mel, dtype=np.float32)
            bad_pitch[0] = np.inf
            np.save(tmp_dir / "pitch" / f"{spk}-pitch-{uid}.npy", bad_pitch)
            ok, err, _ = validate_utterance_feature_files(tmp_dir, uid, spk, fs2_text)
            self.assertFalse(ok)
            self.assertEqual(err, "nan_or_inf_in_pitch")

            # Inject duration sum mismatch
            write_valid()
            bad_dur = dur.copy()
            bad_dur[0] += 3
            np.save(tmp_dir / "duration" / f"{spk}-duration-{uid}.npy", bad_dur)
            ok, err, _ = validate_utterance_feature_files(tmp_dir, uid, spk, fs2_text)
            self.assertFalse(ok)
            self.assertIn("duration_sum_mismatch", err or "")

            # Inject zero duration
            write_valid()
            zero_dur = dur.copy()
            zero_dur[0] = 0
            zero_dur[1] += 4
            np.save(tmp_dir / "duration" / f"{spk}-duration-{uid}.npy", zero_dur)
            ok, err, _ = validate_utterance_feature_files(tmp_dir, uid, spk, fs2_text)
            self.assertFalse(ok)
            self.assertIn("non_positive_duration_min", err or "")

    def test_05_manifest_consistency_and_fastspeech2_dataset_compatibility(self) -> None:
        """Verify train.txt (2,300) and val.txt (421), speaker IDs, Latin-o normalization, and FastSpeech 2 Dataset loading."""
        self.assertEqual(len(self.train_lines), 2300)
        self.assertEqual(len(self.val_lines), 421)
        self.assertEqual(self.speakers, {"spk_000": 0, "spk_001": 1})

        train_ids = set()
        val_ids = set()
        latin_o_checked = set()

        for split_label, lines, id_set in (
            ("train", self.train_lines, train_ids),
            ("val", self.val_lines, val_ids),
        ):
            for line in lines:
                parts = line.split("|")
                self.assertEqual(len(parts), 4, f"Expected 4 fields in {split_label}.txt: {line}")
                uid, spk_str, fs2_text, norm_text = parts
                self.assertNotIn(uid, id_set, f"Duplicate ID {uid} in {split_label}.txt")
                id_set.add(uid)
                self.assertIn(spk_str, self.speakers)
                self.assertTrue(fs2_text.startswith("{") and fs2_text.endswith("}"))
                seq = text_to_sequence(fs2_text)
                self.assertGreater(len(seq), 0)
                for tok_id in seq:
                    self.assertGreater(tok_id, 0)
                    self.assertLess(tok_id, VOCAB_SIZE)
                    self.assertNotEqual(tok_id, UNK_ID)

                if uid in ("train_telugumale_01966", "train_telugumale_01981"):
                    self.assertNotIn("o", fs2_text.lower().replace("@sp", ""))
                    self.assertNotIn("o", norm_text.lower())
                    self.assertIn("ం", fs2_text)
                    self.assertIn("ం", norm_text)
                    latin_o_checked.add(uid)

        self.assertEqual(latin_o_checked, {"train_telugumale_01966", "train_telugumale_01981"})
        self.assertEqual(len(train_ids & val_ids), 0, "Train and Val IDs must be strictly disjoint")

        # Verify compatibility with original_fastspeech2/dataset.py Dataset and DataLoader
        preprocess_config = {
            "dataset": "Telugu_Option_1B",
            "path": {
                "preprocessed_path": str(self.out_dir),
            },
            "preprocessing": {
                "text": {
                    "text_cleaners": [],
                    "language": "te",
                }
            },
        }
        train_config = {
            "optimizer": {
                "batch_size": 4,
            }
        }
        train_ds = Dataset(
            str(self.out_dir / "train.txt"),
            preprocess_config,
            train_config,
            sort=True,
            drop_last=False,
        )
        val_ds = Dataset(
            str(self.out_dir / "val.txt"),
            preprocess_config,
            train_config,
            sort=False,
            drop_last=False,
        )
        text_ds = TextDataset(str(self.out_dir / "val.txt"), preprocess_config)
        self.assertEqual(len(train_ds), 2300)
        self.assertEqual(len(val_ds), 421)
        self.assertEqual(len(text_ds), 421)

        sample = train_ds[0]
        self.assertEqual(sample["mel"].shape[1], 80)
        self.assertEqual(len(sample["pitch"]), sample["mel"].shape[0])
        self.assertEqual(len(sample["energy"]), sample["mel"].shape[0])
        self.assertEqual(int(np.sum(sample["duration"])), sample["mel"].shape[0])
        self.assertEqual(len(sample["text"]), len(sample["duration"]))

        loader = DataLoader(
            val_ds,
            batch_size=4,
            shuffle=False,
            collate_fn=val_ds.collate_fn,
        )
        batch_group = next(iter(loader))
        self.assertGreater(len(batch_group), 0)
        ids, raw_texts, speakers, texts, text_lens, max_text_len, mels, mel_lens, max_mel_len, pitches, energies, durations = batch_group[0]
        self.assertEqual(len(ids), 4)
        self.assertEqual(mels.shape[2], 80)

    def test_06_normalization_statistics_strictly_from_training_split(self) -> None:
        """Verify stats.json structure, positive scales, and that statistics were fit strictly on training data."""
        self.assertIn("pitch", self.stats)
        self.assertIn("energy", self.stats)
        self.assertIn("phoneme_level", self.stats)
        self.assertEqual(len(self.stats["pitch"]), 4)
        self.assertEqual(len(self.stats["energy"]), 4)
        self.assertEqual(len(self.stats["phoneme_level"]["pitch"]), 2)
        self.assertEqual(len(self.stats["phoneme_level"]["energy"]), 2)

        p_min, p_max, p_mean, p_std = self.stats["pitch"]
        e_min, e_max, e_mean, e_std = self.stats["energy"]
        self.assertGreater(p_std, 0.0)
        self.assertGreater(e_std, 0.0)
        self.assertLess(p_min, p_max)
        self.assertLess(e_min, e_max)
        self.assertEqual(
            self.summary["acoustic_parameters"]["normalization_fit_split"],
            "train_only_iqr_outlier_filtered",
        )

        # Verify global pitch/energy min and max in stats.json match the exact training files
        train_p_min, train_p_max = float("inf"), float("-inf")
        train_e_min, train_e_max = float("inf"), float("-inf")
        train_p_pool = []
        train_e_pool = []

        for line in self.train_lines:
            uid, spk_str, _, _ = line.split("|")
            p = np.load(self.out_dir / "pitch" / f"{spk_str}-pitch-{uid}.npy")
            e = np.load(self.out_dir / "energy" / f"{spk_str}-energy-{uid}.npy")
            train_p_min = min(train_p_min, float(p.min()))
            train_p_max = max(train_p_max, float(p.max()))
            train_e_min = min(train_e_min, float(e.min()))
            train_e_max = max(train_e_max, float(e.max()))
            p_raw = p.astype(np.float64) * p_std + p_mean
            e_raw = e.astype(np.float64) * e_std + e_mean
            train_p_pool.extend(remove_outliers(p_raw))
            train_e_pool.extend(remove_outliers(e_raw))

        self.assertAlmostEqual(p_min, train_p_min, places=5)
        self.assertAlmostEqual(p_max, train_p_max, places=5)
        self.assertAlmostEqual(e_min, train_e_min, places=5)
        self.assertAlmostEqual(e_max, train_e_max, places=5)

        # Reconstructed raw training pool matches stats.json mean and std (z-scored mean ~ 0.0, std ~ 1.0 within float32 round-trip precision)
        train_p_arr = (np.asarray(train_p_pool, dtype=np.float64) - p_mean) / p_std
        train_e_arr = (np.asarray(train_e_pool, dtype=np.float64) - e_mean) / e_std
        self.assertAlmostEqual(float(np.mean(train_p_arr)), 0.0, places=3)
        self.assertAlmostEqual(float(np.std(train_p_arr)), 1.0, places=3)
        self.assertAlmostEqual(float(np.mean(train_e_arr)), 0.0, places=3)
        self.assertAlmostEqual(float(np.std(train_e_arr)), 1.0, places=3)

    def test_07_resumable_execution_preserves_valid_files_and_recovers_missing_file(self) -> None:
        """Verify resume skips existing valid files without overwriting them and selectively re-extracts deleted files."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            mini_split_dir = tmp_root / "splits"
            mini_out_dir = tmp_root / "out"
            mini_split_dir.mkdir(parents=True)

            # Copy 2 real train lines and 1 real val line
            with open(self.split_dir / "train.txt", encoding="utf-8") as f:
                orig_train = [f.readline(), f.readline()]
            with open(self.split_dir / "val.txt", encoding="utf-8") as f:
                orig_val = [f.readline()]

            with open(mini_split_dir / "train.txt", "w", encoding="utf-8") as f:
                f.writelines(orig_train)
            with open(mini_split_dir / "val.txt", "w", encoding="utf-8") as f:
                f.writelines(orig_val)

            # First run: extracts 2 train + 1 val from scratch
            s1 = run_telugu_option_1b_preprocessing(
                out_dir=mini_out_dir,
                split_dir=mini_split_dir,
                textgrid_root=self.tg_root,
                batch_size=10,
            )
            self.assertEqual(s1["splits"]["train"]["newly_extracted_utterances"], 2)
            self.assertEqual(s1["splits"]["train"]["resumed_valid_utterances"], 0)
            self.assertEqual(s1["splits"]["val"]["newly_extracted_utterances"], 1)
            self.assertEqual(s1["splits"]["val"]["resumed_valid_utterances"], 0)

            mel_files = sorted((mini_out_dir / "mel").glob("*.npy"))
            self.assertEqual(len(mel_files), 3)
            mtimes_before = {p.name: p.stat().st_mtime_ns for p in mel_files}

            time.sleep(0.02)
            # Second run: resumes all 2 train + 1 val without touching existing .npy files
            s2 = run_telugu_option_1b_preprocessing(
                out_dir=mini_out_dir,
                split_dir=mini_split_dir,
                textgrid_root=self.tg_root,
                batch_size=10,
            )
            self.assertEqual(s2["splits"]["train"]["newly_extracted_utterances"], 0)
            self.assertEqual(s2["splits"]["train"]["resumed_valid_utterances"], 2)
            self.assertEqual(s2["splits"]["val"]["newly_extracted_utterances"], 0)
            self.assertEqual(s2["splits"]["val"]["resumed_valid_utterances"], 1)

            mtimes_after = {p.name: p.stat().st_mtime_ns for p in mel_files}
            self.assertEqual(mtimes_before, mtimes_after)

            # Delete 1 train mel file and re-run: only that 1 train utterance is re-extracted
            deleted_uid = orig_train[0].split("|")[0]
            deleted_spk = f"spk_{int(orig_train[0].split('|')[1]):03d}"
            deleted_mel = mini_out_dir / "mel" / f"{deleted_spk}-mel-{deleted_uid}.npy"
            kept_uid = orig_train[1].split("|")[0]
            kept_spk = f"spk_{int(orig_train[1].split('|')[1]):03d}"
            kept_mel = mini_out_dir / "mel" / f"{kept_spk}-mel-{kept_uid}.npy"

            deleted_mel.unlink()
            time.sleep(0.02)
            s3 = run_telugu_option_1b_preprocessing(
                out_dir=mini_out_dir,
                split_dir=mini_split_dir,
                textgrid_root=self.tg_root,
                batch_size=10,
            )
            self.assertEqual(s3["splits"]["train"]["newly_extracted_utterances"], 1)
            self.assertEqual(s3["splits"]["train"]["resumed_valid_utterances"], 1)
            self.assertTrue(deleted_mel.is_file())
            self.assertEqual(kept_mel.stat().st_mtime_ns, mtimes_before[kept_mel.name])

    def test_08_strict_exclusion_of_failed_alignments(self) -> None:
        """Verify the 3 failed alignment recordings are strictly excluded from features/manifests and logged."""
        expected_excluded = {
            "train_telugufemale_00930",
            "train_telugufemale_00928",
            "train_telugumale_02390",
        }
        self.assertEqual(set(EXCLUDED_FAILED_IDS), expected_excluded)

        logged_ids = {item["id"] for item in self.failures}
        self.assertEqual(logged_ids, expected_excluded)

        manifest_ids = {
            line.split("|")[0] for line in (self.train_lines + self.val_lines)
        }
        self.assertEqual(len(manifest_ids & expected_excluded), 0)

        for sub in ("mel", "pitch", "energy", "duration", "pitch_phoneme", "energy_phoneme"):
            for p in (self.out_dir / sub).glob("*.npy"):
                for bad_id in expected_excluded:
                    self.assertNotIn(bad_id, p.name, f"Excluded ID {bad_id} found in {p}")

        with self.assertRaises(ValueError):
            _load_parquet_audio_for_ids(
                {"train_telugufemale_00930"},
                forbidden_ids=set(EXCLUDED_FAILED_IDS),
            )

    def test_09_strict_exclusion_of_test_data(self) -> None:
        """Verify held-out test data is never accessed, read, or present in any output."""
        self.assertFalse(self.summary["test_split_accessed"])

        with open(self.split_dir / "test.txt", encoding="utf-8") as f:
            test_ids = {line.strip().split("|")[0] for line in f if line.strip()}
        self.assertEqual(len(test_ids), 422)

        manifest_ids = {
            line.split("|")[0] for line in (self.train_lines + self.val_lines)
        }
        self.assertEqual(len(manifest_ids & test_ids), 0)

        all_npy_names = [
            p.name
            for sub in ("mel", "pitch", "energy", "duration", "pitch_phoneme", "energy_phoneme")
            for p in (self.out_dir / sub).glob("*.npy")
        ]
        joined_names = "\n".join(all_npy_names)
        for tid in test_ids:
            self.assertNotIn(tid, joined_names, f"Held-out test ID {tid} leaked into feature files!")

        # Guardrail function must refuse test split
        with self.assertRaises(ValueError):
            _load_option_1b_split_manifest(self.split_dir / "test.txt", "test")
        with self.assertRaises(ValueError):
            _load_option_1b_split_manifest(self.split_dir / "test.txt", "train")


if __name__ == "__main__":
    unittest.main()

