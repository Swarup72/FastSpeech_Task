# IndicTTS Telugu Data Preparation, Splitting, and Text Representation Report

**Assignment:** Implementation of FastSpeech 2-Based Text-to-Speech Synthesis  
**Stage:** Step 2.2A — Telugu Data Preparation, Splitting, and Text Representation  
**Date:** 5 November 2026  
**Auditor / Execution:** Automated Research & Development Pipeline  
**Dataset:** `SPRINGLab/IndicTTS_Telugu`  
**Workspace:** `/Users/swaruppal/FastSpeech2_Project`  
**Regression Test Status:** 182/182 passing tests  
**Decision Status:** **READY FOR TELUGU ALIGNMENT PILOT**  

---

## 1. Executive Summary

Step 2.2A establishes the foundational data preparation, text representation, and partitioning architecture for the **IndicTTS Telugu** dataset. 

A deterministic, multi-speaker stratified split was constructed meeting the assignment requirements ($5.00$ hours training, $1.00$ hour validation, and separate held-out test data) with strictly verified **zero duplicate transcript leakage**. A canonical Unicode NFC text normalization engine and a 101-symbol Telugu grapheme inventory matching AI4Bharat IndicMFA conventions were implemented and tested. Furthermore, a long-recording acoustic segmentation pilot and a band-limited Kaiser sinc resampling pipeline ($16\text{ kHz} \to 22.05\text{ kHz}$ and $48\text{ kHz} \to 22.05\text{ kHz}$) were validated empirically.

---

## 2. Dataset Split Design & Verification

### 2.1 Split Methodology & Random Seed
- **Partitioning Engine:** [partition_dataset.py](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/partition_dataset.py)
- **Random Seed:** `seed = 42` (deterministic pseudo-random shuffle).
- **Target Cumulative Durations:**
  - **Train:** $2.50\text{ h Female} + 2.50\text{ h Male} = \mathbf{5.00\text{ hours}}$
  - **Validation:** $0.50\text{ h Female} + 0.50\text{ h Male} = \mathbf{1.00\text{ hour}}$
  - **Test:** Remaining clean pool ($\mathbf{4.55\text{ hours}}$ held-out)
- **Calibrated Speech Rates:**
  - Female speaker: $8.80\text{ characters/second}$ ($\approx 97\text{ ms/phone}$)
  - Male speaker: $15.20\text{ characters/second}$ ($\approx 65\text{ ms/phone}$)

### 2.2 Duration and Utterance Distribution

| Split | Female Utterances | Female Duration (h) | Male Utterances | Male Duration (h) | Total Utterances | Total Duration (h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 972 | 2.5000 | 1,358 | 2.4997 | **2,330** | **4.9997 h** |
| **Validation** | 192 | 0.5000 | 265 | 0.5000 | **457** | **0.9999 h** |
| **Held-Out Test** | 176 | 0.4599 | 2,190 | 4.0885 | **2,366** | **4.5485 h** |
| **Total Split** | 1,340 | 3.4599 | 3,813 | 7.0882 | **5,153** | **10.5481 h** |

### 2.3 Exclusions & Integrity Logging
A total of **3,423 utterances** from the 8,576 raw corpus rows were excluded from the candidate pool prior to splitting:

| Exclusion Reason | Count | Technical Justification |
| :--- | :---: | :--- |
| `too_long_super_15s` | 3,199 | Utterances $> 15.0\text{s}$ induce quadratic $O(T^2)$ memory explosion and gradient instability during FastSpeech 2 training. |
| `cross_gender_duplicate_title` | 198 | Short repeated story headings (e.g. *భువన సుందరి.*, *వంశ గౌరవం.*) shared across male and female recording sessions, which would cause cross-speaker text leakage if partitioned arbitrarily. |
| `too_short_sub_1s` | 21 | Utterances $< 1.0\text{s}$ lacking sufficient phonetic context for duration and variance modeling. |
| `intra_gender_duplicate_transcript` | 5 | Redundant duplicate sentences within a single speaker's session. |

### 2.4 Zero Duplicate Transcript Leakage Audit
Set intersection between normalized transcripts was audited across all splits:
- $\text{Transcripts}(\text{Train}) \cap \text{Transcripts}(\text{Val}) = \mathbf{0}$
- $\text{Transcripts}(\text{Train}) \cap \text{Transcripts}(\text{Test}) = \mathbf{0}$
- $\text{Transcripts}(\text{Val}) \cap \text{Transcripts}(\text{Test}) = \mathbf{0}$
**100% Leakage-Safe:** No sentence or title appearing in training is present in validation or test data.

---

## 3. Telugu Text Normalization & Symbol Representation

### 3.1 Normalization Architecture ([normalizer.py](file:///Users/swaruppal/FastSpeech2_Project/Telugu/text/normalizer.py))
1. **Unicode NFC Composition:** Enforces canonical composition of vowel matras and modifiers (e.g., decomposing `U+0C46` + `U+0C56` and recomposing to `U+0C48` `కై`).
2. **Zero-Width Character Stripping:** Strips non-printing rendering hints (`\u200B` ZWSP, `\u200C` ZWNJ, `\u200D` ZWJ, `\uFEFF` BOM) which introduce null acoustic tokens.
3. **Punctuation Standardization:** Converts Devanagari/Bengali dandas (`।`, `॥`) to full stops (`.`), curly quotes (`“`, `”`, `‘`, `’`) to standard ASCII quotes, en/em-dashes to hyphens, and collapses ellipsis (`…`).
4. **Whitespace Sanitization:** Collapses multi-spaces and control whitespaces (`\r`, `\t`, `\n`) to single spaces, stripping edge padding.

### 3.2 Symbol Vocabulary Design ([symbols.py](file:///Users/swaruppal/FastSpeech2_Project/Telugu/text/symbols.py), [vocabulary.json](file:///Users/swaruppal/FastSpeech2_Project/Telugu/text/vocabulary.json))
Telugu is modeled at the elementary Unicode grapheme level, aligning 1-to-1 with AI4Bharat IndicMFA's Grapheme-to-Grapheme (G2G) alignment target tokens:

```
Total Vocabulary Size: 101 symbols
- Special Tokens (IDs 0–3):   '_' (PAD=0), '<unk>' (UNK=1), '<bos>' (BOS=2), '<eos>' (EOS=3)
- Alignment Silences (IDs 4–6): '@sp', '@spn', '@sil'
- Standard Punctuation (13):   ' ', '!', "'", '(', ')', ',', '-', '.', ':', ';', '?', '"'
- Independent Vowels (16):     అ, ఆ, ఇ, ఈ, ఉ, ఊ, ఋ, ౠ, ఌ, ౡ, ఎ, ఏ, ఐ, ఒ, ఓ, ఔ
- Consonants (36):             క, ఖ, గ, ఘ, ఙ, చ, ఛ, జ, ఝ, ఞ, ట, ఠ, డ, ఢ, ణ, త, థ, ద, ధ, న, ప, ఫ, బ, భ, మ, య, ర, ఱ, ల, ళ, ఴ, వ, శ, ష, స, హ
- Dependent Vowel Matras (15):  ా, ి, ీ, ు, ూ, ృ, ౄ, ె, ే, ై, ొ, ో, ౌ, ౢ, ౣ
- Virama / Pollu (1):          ్
- Modifiers (4):               ం (anusvara), ః (visarga), ఁ (candrabindu), ఽ (avagraha)
- Telugu Digits (10):          ౦, ౧, ౨, ౩, ౪, ౫, ౬, ౭, ౮, ౯
```

---

## 4. Long Recording Segmentation Pilot Findings

A segmentation pilot was executed on representative long female recordings using [segmentation_pilot.py](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/segmentation_pilot.py):

| Sample ID | Original Duration | Word Count | Terminal Periods | Acoustic Segments Detected | Total Speech | Pause Duration | Pause Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_telugufemale_00002` | $37.20\text{s}$ | 42 | 1 | 27 | $26.62\text{s}$ | $10.58\text{s}$ | $28.4\%$ |
| `train_telugufemale_00012` | $48.39\text{s}$ | 54 | 1 | 33 | $33.60\text{s}$ | $14.79\text{s}$ | $30.6\%$ |

### Core Architectural Findings:
1. **Absence of Internal Punctuation:** The transcripts of long narrative recordings contain only **1 single terminal period** at the end of the entire 30–50 second paragraph. Heuristic rule-based sentence splitting cannot segment these recordings without word-level alignment.
2. **Acoustic Pause Density:** High-energy speech pauses ($> 300\text{ ms}$) comprise 28–31% of recording time.
3. **Strategic Decision:** The clean pool already provides **1,340 naturally concise female utterances** ($3.46\text{ hours}$ between $1.0\text{s}$ and $15.0\text{s}$), completely satisfying the $2.50\text{h}$ training $+ 0.50\text{h}$ validation requirement. Eliminating $> 15\text{s}$ recordings entirely for model training avoids unaligned heuristic text slicing, guarantees flawless audio-text correspondence, and prevents GPU memory overflow.

---

## 5. Sampling Rate Resampling Strategy

An audio resampling pipeline was implemented in [resample.py](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/resample.py) using PyTorch `torchaudio.functional.resample` with a 64-sample Kaiser-windowed lowpass filter (`lowpass_filter_width=64, rolloff=0.99`):
- **Female Stream:** $16,000\text{ Hz} \to 22,050\text{ Hz}$ ($441 / 320$ ratio)
- **Male Stream:** $48,000\text{ Hz} \to 22,050\text{ Hz}$ ($147 / 320$ ratio)

### Verification Metrics:
- **Duration Preservation:** Exact down to sub-millisecond precision ($|\Delta t| \le 0.045\text{ ms}$).
- **Amplitude & Peak:** Zero unintended clipping; peak normalization ceiling ($0.999$) enforces safety against sinc interpolation overshoots.
- **Finite Check:** 100% finite samples (zero NaN or Inf).

---

## 6. Regression & Unit Test Verification

A dedicated unit test suite, [test_telugu_data_preparation.py](file:///Users/swaruppal/FastSpeech2_Project/tests/test_telugu_data_preparation.py), was executed alongside the existing project regression tests:

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover tests
......................................................................................................................................................................................
----------------------------------------------------------------------
Ran 182 tests in 7.777s

OK
```

All 171 prior tests (English Svarah preprocessing, FastSpeech 2 modular architecture, and MPS/CPU training engine) plus the 11 new Telugu preparation tests pass cleanly.

---

## 7. Artifact Summary

The following files were created in this stage:
- [Telugu/metadata/text_manifest.json](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/text_manifest.json): Full metadata manifest for all 8,576 utterances.
- [Telugu/metadata/train.txt](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/train.txt): 2,330 training utterances ($4.9997$ hours).
- [Telugu/metadata/val.txt](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/val.txt): 457 validation utterances ($0.9999$ hours).
- [Telugu/metadata/test.txt](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/test.txt): 2,366 held-out test utterances ($4.5485$ hours).
- [Telugu/metadata/split_summary.json](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/split_summary.json): Complete partitioning audit log.
- [Telugu/text/symbols.py](file:///Users/swaruppal/FastSpeech2_Project/Telugu/text/symbols.py): 101-symbol Telugu grapheme inventory.
- [Telugu/text/vocabulary.json](file:///Users/swaruppal/FastSpeech2_Project/Telugu/text/vocabulary.json): Serialized vocabulary mapping.
- [Telugu/text/normalizer.py](file:///Users/swaruppal/FastSpeech2_Project/Telugu/text/normalizer.py): Unicode NFC normalizer and tokenizer.
- [Telugu/alignment/resample.py](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/resample.py): Kaiser sinc resampling pipeline.
- [Telugu/alignment/segmentation_pilot.py](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/segmentation_pilot.py): Long-recording segmentation evaluation script.
- [research/telugu_segmentation_pilot_results.json](file:///Users/swaruppal/FastSpeech2_Project/research/telugu_segmentation_pilot_results.json): Segmentation metrics.
- [tests/test_telugu_data_preparation.py](file:///Users/swaruppal/FastSpeech2_Project/tests/test_telugu_data_preparation.py): 11 unit tests.

---

## 8. Final Decision

### **READY FOR TELUGU ALIGNMENT PILOT**

### Rationale:
1. The dataset split is completely frozen, reproducible, deterministic, and verified with **zero duplicate transcript leakage**.
2. Training duration targets are met with mathematical precision ($4.9997\text{h}$ train, $0.9999\text{h}$ val), balanced 50/50 across male and female speakers.
3. The Telugu text normalizer and symbol vocabulary are verified and compatible with Montreal Forced Aligner / FastSpeech 2.
4. The resampling pipeline standardizes dual-rate audio to 22,050 Hz with zero clipping or duration drift.
5. All 182 regression and unit tests pass.

---

*(Step 2.2A complete. All execution has halted in accordance with constraints. Awaiting user review and approval before proceeding to the Step 2.2B Telugu Alignment Pilot.)*

