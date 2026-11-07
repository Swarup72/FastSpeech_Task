"""Unit tests for Telugu MFA forced-alignment pilot and duration extraction (Step 2.2B)."""

import os
from pathlib import Path
import sys
import json
import unittest
import numpy as np

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import DATA_DIR, PROJECT_ROOT, RESEARCH_DIR, TELUGU_DIR
from Telugu.text.normalizer import normalize_text, _LATIN_O_ANUSVARA_REGEX
from Telugu.text.symbols import symbols, symbol_to_id, _silences

try:
    import tgt
except ImportError:
    tgt = None


class TestTeluguMFAAlignmentPilot(unittest.TestCase):
    """Verifies G2G tokenization, Latin 'o' correction, alignment parsing, and duration extraction."""

    @classmethod
    def setUpClass(cls):
        cls.pilot_results_path = str(RESEARCH_DIR / "telugu_mfa_pilot_results.json")
        if os.path.exists(cls.pilot_results_path):
            with open(cls.pilot_results_path, "r", encoding="utf-8") as f:
                cls.pilot_results = json.load(f)
        else:
            cls.pilot_results = None

        cls.dict_path = str(RESEARCH_DIR / "Telugu_Dictionary_g2g.txt")
        if os.path.exists(cls.dict_path):
            with open(cls.dict_path, "r", encoding="utf-8") as f:
                cls.dict_words = {line.split()[0] for line in f if line.strip()}
        else:
            cls.dict_words = set()

    def test_telugu_g2g_tokenization_and_dictionary_coverage(self):
        """Verifies splitting Telugu words into graphemes and coverage in IndicMFA dictionary."""
        sample_text = "నమస్కారం బ్రాహ్మడు"
        norm = normalize_text(sample_text)
        chars = [c for c in norm if c not in " .,!?;:'\"-"]
        
        # Verify each char is in IndicMFA dictionary
        self.assertGreater(len(chars), 10)
        for c in chars:
            self.assertIn(c, self.dict_words, f"Character {c!r} missing in IndicMFA dictionary!")

    def test_latin_o_correction_handling(self):
        """Verifies targeted contextual correction of Latin 'o' typo into Telugu Anusvara (ం)."""
        # Case 1: Active training split occurrence
        raw_train = "చేస్తూవుoడేవాడు."
        corr_train = normalize_text(raw_train)
        self.assertNotIn("o", corr_train)
        self.assertIn("ం", corr_train)
        self.assertEqual(corr_train, "చేస్తూవుండేవాడు.")

        # Case 2: Active validation split occurrence
        raw_val = "అచ్చoగా"
        corr_val = normalize_text(raw_val)
        self.assertNotIn("o", corr_val)
        self.assertIn("ం", corr_val)
        self.assertEqual(corr_val, "అచ్చంగా")

        # Case 3: Suffix word boundary
        raw_suffix = "దేవాళయo"
        corr_suffix = normalize_text(raw_suffix)
        self.assertEqual(corr_suffix, "దేవాళయం")

        # Case 4: Non-Telugu Latin words should not be corrupted if standalone
        self.assertFalse(_LATIN_O_ANUSVARA_REGEX.search("hello world"))
        self.assertFalse(_LATIN_O_ANUSVARA_REGEX.search("open"))

    def test_pilot_alignment_success_rate(self):
        """Verifies that all 50 pilot utterances aligned with 100% success."""
        if not self.pilot_results:
            self.skipTest("Pilot results JSON not found")

        self.assertEqual(self.pilot_results["utterance_count"], 50)
        self.assertEqual(self.pilot_results["success_count"], 50)
        self.assertEqual(self.pilot_results["failure_count"], 0)

    def test_alignment_textgrid_parsing_and_integrity(self):
        """Verifies TextGrid interval monotonicity, bounds, and silence tags."""
        sample_tg = str(DATA_DIR / "telugu_mfa_pilot/output/train_telugumale_02568.TextGrid")
        if not os.path.exists(sample_tg) or tgt is None:
            self.skipTest("Sample TextGrid not found or tgt not installed")

        tg = tgt.io.read_textgrid(sample_tg)
        self.assertIn("words", [t.name for t in tg.tiers])
        self.assertIn("phones", [t.name for t in tg.tiers])

        phones = tg.get_tier_by_name("phones")
        self.assertGreater(len(phones.intervals), 10)

        # Verify monotonicity and positive durations
        prev_end = 0.0
        for i, inv in enumerate(phones.intervals):
            dur = inv.end_time - inv.start_time
            self.assertGreater(dur, 0.0, f"Non-positive duration at interval {i}")
            self.assertGreaterEqual(inv.start_time, prev_end - 1e-4, f"Overlapping intervals at {i}")
            prev_end = inv.end_time

        # Verify initial and final silence
        self.assertIn(phones.intervals[0].text, ["sil", "<eps>"])
        self.assertIn(phones.intervals[-1].text, ["sil", "<eps>"])

    def test_frame_boundary_conversion_telescoping(self):
        """Verifies telescoping boundary quantization eliminates cumulative rounding error."""
        sr = 22050
        hop = 256
        timestamps = [0.0, 0.12, 0.25, 0.42, 0.65, 0.88, 1.05]

        # Quantize boundaries
        boundaries = [int(np.round(t * sr / hop)) for t in timestamps]
        durations = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]

        # Telescoping property: sum of differences equals total difference
        self.assertEqual(sum(durations), boundaries[-1] - boundaries[0])
        for d in durations:
            self.assertGreaterEqual(d, 1)

    def test_duration_extraction_feasibility(self):
        """Verifies extracted durations reconcile to mel frames with zero discrepancy."""
        feasibility_path = str(RESEARCH_DIR / "telugu_duration_extraction_feasibility.json")
        if not os.path.exists(feasibility_path):
            self.skipTest("Feasibility JSON not found")

        with open(feasibility_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["utterances_evaluated"], 50)
        self.assertEqual(len(data["oov_tokens"]), 0)
        self.assertEqual(data["max_discrepancy_after"], 0)
        self.assertLessEqual(data["mean_discrepancy_before"], 1.0)

    def test_unknown_token_detection(self):
        """Verifies that unknown or out-of-vocabulary tokens are detected."""
        fake_token = "XYZ_NOT_IN_VOCAB"
        self.assertNotIn(fake_token, symbol_to_id)
        # FastSpeech 2 UNK token index is 1
        self.assertEqual(symbol_to_id.get(fake_token, symbol_to_id["<unk>"]), 1)


class TestTeluguOption1BFullAlignment(unittest.TestCase):
    """Verifies full Option 1B IndicMFA alignment (2,302 train + 422 val, excluding test)."""

    META_DIR = str(TELUGU_DIR / "metadata/splits/option_1b/alignment")
    TG_ROOT = str(DATA_DIR / "telugu_mfa_option_1b/TextGrid")

    def test_01_alignment_manifests_and_test_split_isolation(self):
        """Verify summary, progress, failure, and duration manifests exist and test split was never aligned."""
        for fname in (
            "alignment_summary.json",
            "alignment_progress.json",
            "alignment_failures.json",
            "duration_metadata_train.json",
            "duration_metadata_val.json",
        ):
            fpath = os.path.join(self.META_DIR, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing alignment artifact: {fpath}")

        # Verify test split isolation
        self.assertFalse(os.path.exists(os.path.join(self.TG_ROOT, "test")))

        # Verify existing 50-utterance pilot directory was preserved untouched
        pilot_out_dir = str(DATA_DIR / "telugu_mfa_pilot/output")
        self.assertTrue(os.path.exists(pilot_out_dir))
        pilot_tgs = [f for f in os.listdir(pilot_out_dir) if f.endswith(".TextGrid")]
        self.assertEqual(len(pilot_tgs), 50)

    def test_02_train_and_val_alignment_counts_and_failure_accounting(self):
        """Verify 2,724 total attempted (2,302 train, 422 val), 2,721 succeeded (99.89%), and 3 failures recorded."""
        with open(os.path.join(self.META_DIR, "alignment_summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        with open(os.path.join(self.META_DIR, "alignment_failures.json"), encoding="utf-8") as f:
            failures = json.load(f)

        self.assertFalse(summary["test_split_aligned"])
        self.assertTrue(summary["test_split_isolation_verified"])
        self.assertEqual(summary["total_attempted_utterances"], 2724)
        self.assertEqual(summary["total_successful_alignments"], 2721)
        self.assertEqual(summary["total_failed_alignments"], 3)
        self.assertEqual(len(failures), 3)

        tr = summary["splits"]["train"]
        self.assertEqual(tr["attempted_utterances"], 2302)
        self.assertEqual(tr["successful_alignments"], 2300)
        self.assertEqual(tr["failed_alignments"], 2)
        self.assertEqual(tr["dict_oov_token_count"], 0)
        self.assertEqual(tr["vocab_oov_token_count"], 0)
        self.assertEqual(tr["unknown_phone_interval_count"], 0)
        self.assertEqual(tr["max_abs_frame_discrepancy_after_reconciliation"], 0)

        va = summary["splits"]["val"]
        self.assertEqual(va["attempted_utterances"], 422)
        self.assertEqual(va["successful_alignments"], 421)
        self.assertEqual(va["failed_alignments"], 1)
        self.assertEqual(va["dict_oov_token_count"], 0)
        self.assertEqual(va["vocab_oov_token_count"], 0)
        self.assertEqual(va["unknown_phone_interval_count"], 0)
        self.assertEqual(va["max_abs_frame_discrepancy_after_reconciliation"], 0)

        # Verify physical TextGrid files on disk match successful counts
        train_tgs = [f for f in os.listdir(os.path.join(self.TG_ROOT, "train")) if f.endswith(".TextGrid")]
        val_tgs = [f for f in os.listdir(os.path.join(self.TG_ROOT, "val")) if f.endswith(".TextGrid")]
        self.assertEqual(len(train_tgs), 2300)
        self.assertEqual(len(val_tgs), 421)

    def test_03_duration_frame_consistency_across_all_2721_aligned_utterances(self):
        """Verify sum(reconciled_durations) == expected_trimmed_mel_frames and all tokens in vocabulary for all 2,721 utterances."""
        for split_name, exp_total, exp_succ in (("train", 2302, 2300), ("val", 422, 421)):
            with open(os.path.join(self.META_DIR, f"duration_metadata_{split_name}.json"), encoding="utf-8") as f:
                records = json.load(f)
            self.assertEqual(len(records), exp_total)
            succ_records = [r for r in records if r["status"] == "success"]
            self.assertEqual(len(succ_records), exp_succ)

            for r in succ_records:
                self.assertEqual(
                    sum(r["reconciled_durations"]),
                    r["expected_trimmed_mel_frames"],
                    f"Frame sum mismatch for {r['id']}",
                )
                self.assertEqual(r["discrepancy_after_reconciliation"], 0)
                self.assertGreaterEqual(r["min_speech_frame_dur"], 1)
                self.assertEqual(len(r["extracted_tokens"]), len(r["reconciled_durations"]))
                for tok in r["extracted_tokens"]:
                    self.assertIn(tok, symbol_to_id, f"OOV token {tok!r} in {r['id']}")


if __name__ == "__main__":
    unittest.main()

