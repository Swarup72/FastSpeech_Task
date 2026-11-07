"""Unit tests for English alignment duration extraction and mel reconciliation."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
import numpy as np

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import DATA_DIR, ORIGINAL_FS2_DIR, PROJECT_ROOT, setup_project_paths

setup_project_paths()

from English.alignment.duration import (
    DurationExtractionResult,
    PhoneInterval,
    compute_frame_durations_telescoping,
    extract_durations_from_textgrid,
    fill_unannotated_gaps,
    phone_to_fastspeech2_symbol,
    reconcile_durations_with_mel,
    trim_silences,
    validate_intervals,
    validate_symbols_against_vocabulary,
)


class TestIntervalValidation(unittest.TestCase):
    """Tests for interval structure validation."""

    def test_valid_intervals(self):
        intervals = [
            PhoneInterval(text="", start_time=0.0, end_time=0.1),
            PhoneInterval(text="HH", start_time=0.1, end_time=0.2),
            PhoneInterval(text="W", start_time=0.2, end_time=0.35),
        ]
        validate_intervals(intervals)

    def test_empty_intervals_rejected(self):
        with self.assertRaises(ValueError):
            validate_intervals([])

    def test_negative_start_time_rejected(self):
        intervals = [PhoneInterval(text="HH", start_time=-0.05, end_time=0.1)]
        with self.assertRaises(ValueError):
            validate_intervals(intervals)

    def test_negative_duration_rejected(self):
        intervals = [PhoneInterval(text="HH", start_time=0.2, end_time=0.1)]
        with self.assertRaises(ValueError):
            validate_intervals(intervals)

    def test_overlapping_intervals_rejected(self):
        intervals = [
            PhoneInterval(text="HH", start_time=0.0, end_time=0.2),
            PhoneInterval(text="W", start_time=0.15, end_time=0.3),  # 50ms overlap
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_intervals(intervals, tolerance=1e-4)
        self.assertIn("Overlapping intervals detected", str(ctx.exception))

    def test_zero_length_interval_valid_if_non_negative(self):
        intervals = [
            PhoneInterval(text="", start_time=0.0, end_time=0.0),
            PhoneInterval(text="HH", start_time=0.0, end_time=0.1),
        ]
        validate_intervals(intervals)


class TestGapFilling(unittest.TestCase):
    """Tests for unannotated gap detection and filling."""

    def test_no_gaps(self):
        intervals = [
            PhoneInterval(text="HH", start_time=0.0, end_time=0.1),
            PhoneInterval(text="W", start_time=0.1, end_time=0.2),
        ]
        filled = fill_unannotated_gaps(intervals)
        self.assertEqual(len(filled), 2)
        self.assertEqual(filled[0].text, "HH")
        self.assertEqual(filled[1].text, "W")

    def test_internal_gap_filled_with_sp(self):
        intervals = [
            PhoneInterval(text="HH", start_time=0.0, end_time=0.1),
            PhoneInterval(text="W", start_time=0.25, end_time=0.35),  # 150ms gap
        ]
        filled = fill_unannotated_gaps(intervals)
        self.assertEqual(len(filled), 3)
        self.assertEqual(filled[1].text, "sp")
        self.assertAlmostEqual(filled[1].start_time, 0.1)
        self.assertAlmostEqual(filled[1].end_time, 0.25)

    def test_leading_gap_filled(self):
        intervals = [
            PhoneInterval(text="HH", start_time=0.1, end_time=0.2),
        ]
        filled = fill_unannotated_gaps(intervals)
        self.assertEqual(len(filled), 2)
        self.assertEqual(filled[0].text, "sp")
        self.assertAlmostEqual(filled[0].start_time, 0.0)
        self.assertAlmostEqual(filled[0].end_time, 0.1)


class TestSilenceTrimming(unittest.TestCase):
    """Tests for silence trimming matching FastSpeech 2 frontend."""

    def test_leading_and_trailing_silence_trimmed(self):
        intervals = [
            PhoneInterval(text="", start_time=0.0, end_time=0.05),
            PhoneInterval(text="sil", start_time=0.05, end_time=0.10),
            PhoneInterval(text="HH", start_time=0.10, end_time=0.20),
            PhoneInterval(text="AH1", start_time=0.20, end_time=0.30),
            PhoneInterval(text="sp", start_time=0.30, end_time=0.40),
            PhoneInterval(text="T", start_time=0.40, end_time=0.50),
            PhoneInterval(text="", start_time=0.50, end_time=0.60),
            PhoneInterval(text="sil", start_time=0.60, end_time=0.70),
        ]
        trimmed, start_t, end_t = trim_silences(intervals)
        self.assertEqual(len(trimmed), 4)  # HH, AH1, sp, T
        self.assertAlmostEqual(start_t, 0.10)
        self.assertAlmostEqual(end_t, 0.50)
        self.assertEqual([x.text for x in trimmed], ["HH", "AH1", "sp", "T"])

    def test_all_silence_returns_empty(self):
        intervals = [
            PhoneInterval(text="", start_time=0.0, end_time=0.1),
            PhoneInterval(text="sil", start_time=0.1, end_time=0.2),
        ]
        trimmed, start_t, end_t = trim_silences(intervals)
        self.assertEqual(len(trimmed), 0)
        self.assertEqual(start_t, 0.0)
        self.assertEqual(end_t, 0.0)


class TestDurationCalculation(unittest.TestCase):
    """Tests for frame duration discretization and telescoping property."""

    def test_telescoping_property(self):
        sr = 22050
        hop = 256
        intervals = [
            PhoneInterval(text="HH", start_time=0.10, end_time=0.22),
            PhoneInterval(text="AH1", start_time=0.22, end_time=0.35),
            PhoneInterval(text="sp", start_time=0.35, end_time=0.45),
            PhoneInterval(text="T", start_time=0.45, end_time=0.60),
        ]
        start_offset = 0.10
        end_offset = 0.60
        durations = compute_frame_durations_telescoping(
            intervals, start_offset, end_offset, sr, hop
        )
        self.assertEqual(len(durations), 4)
        expected_total = int(np.round((end_offset - start_offset) * sr / hop))
        self.assertEqual(sum(durations), expected_total)

    def test_short_phone_gets_at_least_min_duration(self):
        sr = 22050
        hop = 256
        # A tiny interval of 2ms (< 0.5 frames)
        intervals = [
            PhoneInterval(text="T", start_time=0.0, end_time=0.002),
            PhoneInterval(text="AH0", start_time=0.002, end_time=0.10),
        ]
        durations = compute_frame_durations_telescoping(
            intervals, 0.0, 0.10, sr, hop, min_phone_duration=1
        )
        self.assertGreaterEqual(durations[0], 1)
        self.assertGreaterEqual(durations[1], 1)


class TestMelReconciliation(unittest.TestCase):
    """Tests for mel spectrogram length reconciliation."""

    def test_exact_match_no_change(self):
        durations = [5, 10, 8]
        reconciled, disc = reconcile_durations_with_mel(durations, target_length=23)
        self.assertEqual(reconciled, [5, 10, 8])
        self.assertEqual(disc, 0)
        self.assertEqual(sum(reconciled), 23)

    def test_mel_has_extra_frame(self):
        # Target is 24, current sum is 23 (mel +1 frame)
        durations = [5, 10, 8]
        intervals = [
            PhoneInterval(text="HH", start_time=0.0, end_time=0.06),
            PhoneInterval(text="sp", start_time=0.06, end_time=0.18),
            PhoneInterval(text="T", start_time=0.18, end_time=0.27),
        ]
        reconciled, disc = reconcile_durations_with_mel(
            durations, target_length=24, intervals=intervals
        )
        self.assertEqual(sum(reconciled), 24)
        self.assertEqual(disc, 1)

    def test_mel_has_fewer_frames(self):
        # Target is 22, current sum is 23 (mel -1 frame)
        durations = [5, 10, 8]
        reconciled, disc = reconcile_durations_with_mel(
            durations, target_length=22
        )
        self.assertEqual(sum(reconciled), 22)
        self.assertEqual(disc, -1)
        # Ensure no duration dropped below 1
        for d in reconciled:
            self.assertGreaterEqual(d, 1)


class TestFastSpeech2Symbols(unittest.TestCase):
    """Tests for symbol mapping and model vocabulary compatibility."""

    def test_phone_to_symbol_mapping(self):
        self.assertEqual(phone_to_fastspeech2_symbol("HH"), "@HH")
        self.assertEqual(phone_to_fastspeech2_symbol("AA1"), "@AA1")
        self.assertEqual(phone_to_fastspeech2_symbol(""), "@sp")
        self.assertEqual(phone_to_fastspeech2_symbol("sp"), "@sp")
        self.assertEqual(phone_to_fastspeech2_symbol("sil"), "@sil")
        self.assertEqual(phone_to_fastspeech2_symbol("spn"), "@spn")
        self.assertEqual(phone_to_fastspeech2_symbol("@AA1"), "@AA1")

    def test_all_symbols_in_fastspeech2_vocabulary(self):
        import sys
        fs2_path = str(Path(__file__).resolve().parent.parent / "original_fastspeech2")
        if fs2_path not in sys.path:
            sys.path.insert(0, fs2_path)
        from text.symbols import symbols as fs2_symbols

        test_symbols = [
            "@HH", "@W", "@AH1", "@T", "@AY1", "@P", "@AH1", "@V",
            "@L", "@ER1", "@N", "@IH0", "@NG", "@sp", "@sil", "@spn"
        ]
        is_valid, unknown = validate_symbols_against_vocabulary(test_symbols, fs2_symbols)
        self.assertTrue(is_valid)
        self.assertEqual(unknown, [])


class TestRealPilotTextGrids(unittest.TestCase):
    """End-to-end tests on real TextGrids generated from Svarah pilot utterances."""

    @classmethod
    def setUpClass(cls):
        import glob
        cls.tg_files = sorted(glob.glob(str(DATA_DIR / "svarah_pilot_100/textgrids/*/*.TextGrid")))

    def test_pilot_textgrids_exist(self):
        self.assertEqual(len(self.tg_files), 100)

    def test_extract_all_100_pilot_textgrids(self):
        import sys
        fs2_path = str(Path(__file__).resolve().parent.parent / "original_fastspeech2")
        if fs2_path not in sys.path:
            sys.path.insert(0, fs2_path)
        from text.symbols import symbols as fs2_symbols

        for tg_path in self.tg_files:
            result = extract_durations_from_textgrid(
                tg_path,
                sampling_rate=22050,
                hop_length=256,
                trim_silence=True,
            )
            self.assertIsInstance(result, DurationExtractionResult)
            self.assertGreater(len(result.phones), 0)
            self.assertEqual(len(result.phones), len(result.durations))
            self.assertEqual(len(result.phones), len(result.symbols))
            self.assertGreater(result.end_time, result.start_time)
            self.assertEqual(result.total_frames, sum(result.durations))
            self.assertGreater(result.total_frames, 0)

            # All durations must be positive integers
            for d in result.durations:
                self.assertIsInstance(d, int)
                self.assertGreaterEqual(d, 1)

            # All symbols must be valid in FastSpeech 2 vocabulary
            is_valid, unknown = validate_symbols_against_vocabulary(result.symbols, fs2_symbols)
            self.assertTrue(is_valid, f"Unknown symbols {unknown} in {tg_path}")


if __name__ == "__main__":
    unittest.main()

