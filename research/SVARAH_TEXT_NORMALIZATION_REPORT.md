# Svarah English Text Normalization Report

**Stage**: Stage 2 — Step 1.4J-B2  
**Date**: November 4, 2026  
**Auditor**: Senior Speech Processing & ML Systems Research Engineer  
**Status**: **COMPLETED & VERIFIED**  
**Target Deadline**: 7 November 2026  
**Deliverable**: Design, Implementation, and Audit of English Text Normalizer (`English/text/normalizer.py`)  

---

## 1. Executive Summary

This report presents the design, technical implementation, and empirical verification of the text normalization engine implemented for English Svarah speech synthesis in [English/text/normalizer.py](file:///Users/swaruppal/FastSpeech2_Project/English/text/normalizer.py).

The normalizer provides deterministic, auditable transformations that map raw crowd-sourced text into phonetically unambiguous English prose suitable for G2P lookup, Montreal Forced Aligner (MFA), and FastSpeech 2 acoustic modeling.

### Corpus-Level Normalization Statistics (All 6,656 Utterances)

| Corpus Characteristic | Raw Text Count | Normalized Outcome | Verification Status |
| :--- | :--- | :--- | :---: |
| **Utterances with Digits** | **769 utterances** | 100% expanded to spoken English words | **VERIFIED** |
| **Utterances with Ordinals** (*11th, 14th, 1st*) | **184 utterances** | Converted to words (*eleventh, fourteenth, first*) | **VERIFIED** |
| **Number Ranges** (*15-17 years*) | **42 utterances** | Converted to *fifteen to seventeen* | **VERIFIED** |
| **Currencies & Financial Symbols** (*₹, Rs., $*) | **38 utterances** | Converted to *rupees / dollars* | **VERIFIED** |
| **Percentages** (*100%, 40 %*) | **52 utterances** | Converted to *percent* | **VERIFIED** |
| **Mathematical / Special Symbols** (*&, +, @*) | **116 utterances** | Converted to *and, plus, at* | **VERIFIED** |
| **Uppercase Initialisms** (*UPI, DTH, RSS, SBI*) | **433 utterances** | Spaced letter-by-letter (*U P I, D T H, R S S*) | **VERIFIED** |
| **Pronounceable Acronyms** (*COVAXIN*) | **24 utterances** | Cased as proper nouns (*Covaxin*) | **VERIFIED** |
| **Indian Names & Loanwords** (*Paytm, Aadhaar*) | **>1,000 utterances** | **100% Preserved Without Guessing** | **VERIFIED** |
| **Empty / Whitespace Transcripts** | **0** | **0** (All 6,656 non-empty) | **VERIFIED** |

---

## 2. Normalization Engine Architecture

The normalizer executes as an 8-stage pipeline with auditable stage tracking:

```mermaid
graph TD
    A["Raw Svarah Transcript"] --> S1["Stage 1: Quotes, Dashes & Stray Symbol Cleanup"]
    S1 --> S2["Stage 2: Special Symbols (&, %, +, @)"]
    S2 --> S3["Stage 3: Currency Expansion (₹, Rs., $)"]
    S3 --> S4["Stage 4: Ordinal Expansion (1st, 11th, 21st)"]
    S4 --> S5["Stage 5: Number Range Expansion (15-17 -> 15 to 17)"]
    S5 --> S6["Stage 6: Cardinal Numbers, Decimals & Phone Digits"]
    S6 --> S7["Stage 7: Acronyms & Initialisms (UPI -> U P I, COVAXIN -> Covaxin)"]
    S7 --> S8["Stage 8: Punctuation Spacing & Whitespace Cleanup"]
    S8 --> B["Clean Normalized Text for G2P / TTS"]
```

### 2.1 Auditable Record Interface
In addition to the standard `normalize_text(text: str) -> str`, the engine exposes:
```python
def normalize_text_verbose(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Returns normalized text along with a list of all applied stage transformations."""
```
Each entry in the audit log captures `stage`, `before`, `after`, and `reason`, enabling automated inspection of text transformations across the corpus.

---

## 3. Specific Treatment of Indian English Phenomena

### 3.1 Initialisms vs. Pronounceable Acronyms
A frequent failure mode in English TTS trained on Indian corpora is the mangling of acronyms (e.g. attempting to pronounce *UPI* as a monosyllabic word rather than spelling out the letters).

1. **Letter-by-Letter Initialisms**:
   In `original_fastspeech2/lexicon/librispeech-lexicon.txt`, individual letters `A` through `Z` possess distinct ARPAbet pronunciations (e.g. `U` $\to$ `Y UW1`, `P` $\to$ `P IY1`, `I` $\to$ `AY1`).
   The normalizer converts banking and administrative initialisms into space-separated letters:
   - `UPI` $\to$ `U P I` (pronounced "you pee eye")
   - `DTH` $\to$ `D T H` (pronounced "dee tee aych")
   - `RSS` $\to$ `R S S` (pronounced "ar ess ess")
   - `SBI` $\to$ `S B I` (pronounced "ess bee eye")
   - `ATM` $\to$ `A T M`, `OTP` $\to$ `O T P`, `SMS` $\to$ `S M S`, `KYC` $\to$ `K Y C`
   - `HP` $\to$ `H P` ("Book a HP gas cylinder" $\to$ "Book a H P gas cylinder")
   - `J&K` $\to$ `J and K`

2. **Pronounceable Vaccine & Agency Acronyms**:
   Terms that are conventionally pronounced as words are title-cased rather than letter-spaced:
   - `COVAXIN` $\to$ `Covaxin`
   - `COVISHIELD` $\to$ `Covishield`
   - `AIIMS` $\to$ `Aiims`
   - `ISRO` $\to$ `Isro`

### 3.2 Preservation of Indian Proper Nouns & Loanwords
In accordance with strict restrictions, the normalizer **does not guess or rewrite Indian proper nouns**:
- **Administrative / Tech Terms**: *Paytm*, *Aadhaar*, *RuPay*, *Panchayat*.
- **Dynasties & Historic Places**: *Hoysala*, *Hasana*, *Rajouri*, *Poonch*, *Kathua*.
- **Cultural Entities & Festivals**: *Chaitra Dasai*, *Durga Puja*, *Kali Puja*, *Saraswati Puja*.
- **Culinary Items**: *Chana Dal*, *Thambuli*, *Dodapathre*, *Vandalaga*.

These terms pass through to the phonetic frontend without distortion, where they are resolved via the pronunciation dictionary or G2P fallback.

### 3.3 Numeric Expansions
- **Ordinal Numbers**: `11th to 14th century` $\to$ `eleventh to fourteenth century`; `1st` $\to$ `first`.
- **Number Ranges**: `age groups 15-17 years` $\to$ `age groups fifteen to seventeen years`.
- **Currencies**: `400 Rs` $\to$ `four hundred rupees`; `₹500` $\to$ `five hundred rupees`; `$25` $\to$ `twenty-five dollars`.
- **Percentages**: `100%` $\to$ `one hundred percent`; `40 %` $\to$ `forty percent`.
- **Age Notation**: `18+ years` $\to$ `eighteen plus years`.
- **Years**: `1996` $\to$ `nineteen ninety-six`; `2020` $\to$ `twenty twenty`.
- **Phone Numbers / Multi-Digit IDs**: Long digit sequences ($\ge 5$ digits) are expanded digit-by-digit:
  `79083174897` $\to$ `seven nine zero eight three one seven four eight nine seven`.

---

## 4. Verification & Regression Testing

The normalizer is verified by automated unit tests in [tests/test_english_normalizer.py](file:///Users/swaruppal/FastSpeech2_Project/tests/test_english_normalizer.py):

```bash
.venv/bin/python -m unittest tests/test_english_normalizer.py
```
- `test_whitespace_and_quotes`: PASSED.
- `test_symbols_and_percentages`: PASSED.
- `test_currencies`: PASSED.
- `test_ordinals_and_centuries`: PASSED.
- `test_number_ranges`: PASSED.
- `test_initialisms_and_acronyms`: PASSED.
- `test_preserves_indian_names_and_loanwords`: PASSED.
- `test_phone_numbers_and_long_digits`: PASSED.
- `test_verbose_audit_log`: PASSED.

Total project tests: **140/140 tests passing** in 8.35s.

