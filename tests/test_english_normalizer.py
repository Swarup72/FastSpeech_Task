"""Unit tests for English text normalizer (Svarah dataset)."""

from __future__ import annotations

import unittest

from English.text.normalizer import normalize_text, normalize_text_verbose


class TestEnglishNormalizer(unittest.TestCase):
    """Test suite for English text normalization rules."""

    def test_whitespace_and_quotes(self) -> None:
        raw = "  Hello    world!  ‘It’s’   “great”.   "
        expected = "Hello world! 'It's' \"great\"."
        self.assertEqual(normalize_text(raw), expected)

    def test_symbols_and_percentages(self) -> None:
        self.assertEqual(
            normalize_text("The students gave 100% in learning."),
            "The students gave one hundred percent in learning.",
        )
        self.assertEqual(
            normalize_text("Discount of 40 % on Utensils."),
            "Discount of forty percent on Utensils.",
        )
        self.assertEqual(
            normalize_text("Rice & Almond Drink"),
            "Rice and Almond Drink",
        )
        self.assertEqual(
            normalize_text("Age group 18+ years"),
            "Age group eighteen plus years",
        )
        self.assertEqual(
            normalize_text("user@domain"),
            "user at domain",
        )

    def test_currencies(self) -> None:
        self.assertEqual(
            normalize_text("A transaction of 400 Rs"),
            "A transaction of four hundred rupees",
        )
        self.assertEqual(
            normalize_text("Pay ₹500 immediately"),
            "Pay five hundred rupees immediately",
        )
        self.assertEqual(
            normalize_text("Amount is $25"),
            "Amount is twenty-five dollars",
        )

    def test_ordinals_and_centuries(self) -> None:
        raw = "from 11th to 14th century CE."
        expected = "from eleventh to fourteenth century C E."
        self.assertEqual(normalize_text(raw), expected)

        self.assertEqual(normalize_text("1st and 2nd dose"), "first and second dose")
        self.assertEqual(normalize_text("21st century"), "twenty-first century")

    def test_number_ranges(self) -> None:
        raw = "available for age groups 15-17 years"
        expected = "available for age groups fifteen to seventeen years"
        self.assertEqual(normalize_text(raw), expected)

    def test_initialisms_and_acronyms(self) -> None:
        self.assertEqual(
            normalize_text("Pay using UPI apps or SBI ATM."),
            "Pay using U P I apps or S B I A T M.",
        )
        self.assertEqual(
            normalize_text("Show my DTH bill readings."),
            "Show my D T H bill readings.",
        )
        self.assertEqual(
            normalize_text("First Dose of COVAXIN is available"),
            "First Dose of Covaxin is available",
        )
        self.assertEqual(
            normalize_text("The UT of J&K"),
            "The U T of J and K",
        )

    def test_preserves_indian_names_and_loanwords(self) -> None:
        raw = "Our district Hasana was the land of the great Hoysala dynasty."
        expected = "Our district Hasana was the land of the great Hoysala dynasty."
        self.assertEqual(normalize_text(raw), expected)

        raw2 = "Paytm and Aadhaar are used across Karnataka and Kerala."
        expected2 = "Paytm and Aadhaar are used across Karnataka and Kerala."
        self.assertEqual(normalize_text(raw2), expected2)

    def test_phone_numbers_and_long_digits(self) -> None:
        raw = "Contact number 7908317489"
        expected = "Contact number seven nine zero eight three one seven four eight nine"
        self.assertEqual(normalize_text(raw), expected)

    def test_verbose_audit_log(self) -> None:
        raw = "Pay ₹500 using UPI for 100% cashback."
        normalized, audit = normalize_text_verbose(raw)
        self.assertTrue(len(audit) > 0)
        stages = [entry["stage"] for entry in audit]
        self.assertIn("currency_expansion", stages)
        self.assertIn("symbol_expansion", stages)
        self.assertIn("acronym_handling", stages)
        self.assertIn("five hundred rupees", normalized)
        self.assertIn("U P I", normalized)
        self.assertIn("one hundred percent", normalized)


if __name__ == "__main__":
    unittest.main()

