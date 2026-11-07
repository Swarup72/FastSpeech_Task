"""Unit tests verifying Telugu dataset split and vocabulary audit (Step 2.2A-R)."""

import json
import os
from pathlib import Path
import sys
import unittest

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import RESEARCH_DIR, TELUGU_DIR
from Telugu.text.normalizer import normalize_text
from Telugu.text.symbols import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    _punctuation,
    _silences,
    _special,
    _telugu_consonants,
    _telugu_digits,
    _telugu_matras,
    _telugu_modifiers,
    _telugu_virama,
    _telugu_vowels,
    id_to_symbol,
    symbol_to_id,
    symbols,
)


class TestTeluguSplitVocabularyAudit(unittest.TestCase):
    """Verifies duration accounting, split isolation, and vocabulary consistency."""

    @classmethod
    def setUpClass(cls):
        # Load audit results
        audit_path = str(RESEARCH_DIR / "telugu_exact_duration_audit.json")
        if os.path.exists(audit_path):
            with open(audit_path, "r", encoding="utf-8") as f:
                cls.audit = json.load(f)
        else:
            cls.audit = None

        # Load splits
        cls.splits = {}
        for split in ["train", "val", "test"]:
            path = str(TELUGU_DIR / "metadata" / f"{split}.txt")
            entries = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) >= 6:
                            entries.append({
                                "id": parts[0],
                                "speaker_id": parts[1],
                                "raw_text": parts[2],
                                "norm_text": parts[3],
                                "est_dur": float(parts[4]),
                                "speaker": parts[5]
                            })
            cls.splits[split] = entries

    def test_duration_accounting_identities(self):
        """Verifies exact utterance count and duration identities across splits."""
        if not self.audit:
            self.skipTest("Audit JSON not found")

        total_utts = self.audit["total_utterances"]
        self.assertEqual(total_utts, 8576)

        train = self.audit["splits"]["train"]
        val = self.audit["splits"]["val"]
        test = self.audit["splits"]["test"]
        excl = self.audit["splits"]["excluded"]

        # Utterance count identities
        inc_utts = train["utterances"] + val["utterances"] + test["utterances"]
        self.assertEqual(inc_utts + excl["utterances"], total_utts)
        self.assertEqual(inc_utts, 5153)
        self.assertEqual(excl["utterances"], 3423)

        # Duration identities
        tot_sec = self.audit["total_duration_seconds"]
        inc_sec = train["actual_duration_seconds"] + val["actual_duration_seconds"] + test["actual_duration_seconds"]
        excl_sec = excl["actual_duration_seconds"]

        self.assertAlmostEqual(inc_sec + excl_sec, tot_sec, places=4)
        self.assertAlmostEqual(tot_sec, 132164.18, places=1) # ~36.71 hours

    def test_split_id_uniqueness_and_isolation(self):
        """Verifies 100% unique IDs in each split and zero set intersections."""
        train_ids = [e["id"] for e in self.splits["train"]]
        val_ids = [e["id"] for e in self.splits["val"]]
        test_ids = [e["id"] for e in self.splits["test"]]

        # Uniqueness
        self.assertEqual(len(train_ids), len(set(train_ids)))
        self.assertEqual(len(val_ids), len(set(val_ids)))
        self.assertEqual(len(test_ids), len(set(test_ids)))

        # Isolation
        self.assertEqual(len(set(train_ids) & set(val_ids)), 0)
        self.assertEqual(len(set(train_ids) & set(test_ids)), 0)
        self.assertEqual(len(set(val_ids) & set(test_ids)), 0)

    def test_zero_text_leakage_between_splits(self):
        """Verifies no normalized text strings overlap between train, val, and test."""
        train_texts = {e["norm_text"] for e in self.splits["train"]}
        val_texts = {e["norm_text"] for e in self.splits["val"]}
        test_texts = {e["norm_text"] for e in self.splits["test"]}

        self.assertEqual(len(train_texts & val_texts), 0)
        self.assertEqual(len(train_texts & test_texts), 0)
        self.assertEqual(len(val_texts & test_texts), 0)

    def test_cross_gender_duplicate_quarantine(self):
        """Verifies known duplicate chapter headings are completely absent from splits."""
        all_split_texts = {
            e["norm_text"] for s in ["train", "val", "test"] for e in self.splits[s]
        }
        known_duplicates = [
            "భువన సుందరి.",
            "నావికుడు సింద్ బాద్.",
            "వంశ గౌరవం.",
            "తోక చుక్క.",
            "సూర్యుడు."
        ]
        for dup in known_duplicates:
            self.assertNotIn(dup, all_split_texts)

    def test_split_target_durations(self):
        """Verifies split audio durations satisfy training and validation targets."""
        if not self.audit:
            self.skipTest("Audit JSON not found")

        train = self.audit["splits"]["train"]
        val = self.audit["splits"]["val"]

        # Training split: ~5 hours total
        self.assertGreaterEqual(train["actual_duration_hours"], 5.0)
        self.assertLessEqual(train["actual_duration_hours"], 6.0)

        # Training speaker balance: ~2.45h female, ~3.00h male
        self.assertGreaterEqual(train["female_duration_hours"], 2.4)
        self.assertGreaterEqual(train["male_duration_hours"], 2.5)

        # Validation split: ~1 hour total
        self.assertGreaterEqual(val["actual_duration_hours"], 1.0)
        self.assertLessEqual(val["actual_duration_hours"], 1.25)
        self.assertAlmostEqual(val["female_duration_hours"], 0.50, delta=0.05)

    def test_vocabulary_completeness_and_special_tokens(self):
        """Verifies special tokens, silences, and graphemes in symbols.py."""
        self.assertEqual(PAD_ID, 0)
        self.assertEqual(UNK_ID, 1)
        self.assertEqual(BOS_ID, 2)
        self.assertEqual(EOS_ID, 3)

        self.assertIn("@sp", _silences)
        self.assertIn("@spn", _silences)
        self.assertIn("@sil", _silences)

        # Total symbols count
        self.assertEqual(len(symbols), 101)
        self.assertEqual(len(symbol_to_id), 101)
        self.assertEqual(len(id_to_symbol), 101)

    def test_mfa_telugu_dictionary_inventory_compatibility(self):
        """Verifies IndicMFA Telugu_Dictionary_g2g.txt phones vs symbols.py."""
        dict_path = str(RESEARCH_DIR / "Telugu_Dictionary_g2g.txt")
        if not os.path.exists(dict_path):
            self.skipTest("Telugu_Dictionary_g2g.txt not found")

        with open(dict_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        mfa_phones = set()
        for line in lines:
            parts = line.split()
            mfa_phones.update(parts[1:])

        # Core elementary Telugu characters in symbols.py are covered
        sym_set = set(symbols)
        # All basic vowels and consonants must be supported
        for v in _telugu_vowels:
            self.assertIn(v, mfa_phones)
        for c in _telugu_consonants:
            self.assertIn(c, mfa_phones)
        for m in ["ా", "ి", "ీ", "ు", "ూ", "ృ", "ౄ", "ె", "ే", "ై", "ొ", "ో", "ౌ"]:
            # Either bare or with dotted circle in MFA
            self.assertTrue(m in mfa_phones or f"◌{m}" in mfa_phones)


if __name__ == "__main__":
    unittest.main()

