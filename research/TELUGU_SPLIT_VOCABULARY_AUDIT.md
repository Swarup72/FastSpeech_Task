# Step 2.2A-R — Telugu Dataset Split and Vocabulary Consistency Audit Report

**Date:** November 5, 2026  
**Auditor:** FastSpeech 2 Project Team  
**Dataset:** `SPRINGLab/IndicTTS_Telugu` (Parquet shards 00000 to 00003)  
**Status:** **READY FOR TELUGU ALIGNMENT PILOT**

---

## Executive Summary

This independent technical audit was conducted to resolve the duration accounting discrepancy identified following Step 2.2A, perform an exhaustive verification of split isolation and leakage safety, and evaluate the Telugu vocabulary and text representation against the official AI4Bharat IndicMFA Grapheme-to-Grapheme (G2G) alignment specification.

### Key Audit Findings:

1. **Root Cause of Duration Discrepancy Resolved**:
   - The initial IndicTTS dataset README stated: *"Total Duration: ~8.74 hours (Male: 4.47 hours, Female: 4.27 hours)"*. **This catalog documentation was drastically understated and factually incorrect.**
   - All 4 Parquet shards (6.44 GB total) were downloaded locally and audited at the raw PCM audio byte level for all 8,576 utterances.
   - The actual decoded audio duration of `SPRINGLab/IndicTTS_Telugu` is **132,164.18 seconds (36.7123 hours)**:
     - **Female speaker (16 kHz, 16-bit mono)**: 4,043 utterances, **86,115.47s (23.9210 hours)**, mean duration **21.30s**.
     - **Male speaker (48 kHz, 16-bit mono)**: 4,533 utterances, **46,048.70s (12.7913 hours)**, mean duration **10.16s**.
   - The Step 2.2A split report of 10.5481 hours was an estimated duration for the **5,153 clean utterances** selected after applying duration bounds ($1.0\text{s} \le \text{est\_dur} \le 15.0\text{s}$) and duplicate quarantining. It was not an over-accounting of an 8.74h dataset, but a subset selection from a 36.71-hour corpus.

2. **Mathematical Accounting Identities Verified (Zero Error)**:
   - **Utterance Count Identity**:
     $$\text{Included (5,153)} + \text{Excluded (3,423)} = \text{Total Source (8,576)} \quad [\checkmark \text{ EXACT}]$$
     $$\text{Train (2,330)} + \text{Validation (457)} + \text{Test (2,366)} = \text{Included (5,153)} \quad [\checkmark \text{ EXACT}]$$
   - **Audio Duration Identity (Decoded PCM Seconds)**:
     $$\text{Included (42,843.42s)} + \text{Excluded (89,320.76s)} = \text{Total (132,164.18s)} \quad [\Delta = 0.000000\text{s}, \checkmark \text{ EXACT}]$$
     $$\text{Train (19,640.68s)} + \text{Validation (3,954.19s)} + \text{Test (19,248.55s)} = \text{Included (42,843.42s)} \quad [\Delta = 0.000000\text{s}, \checkmark \text{ EXACT}]$$

3. **Split Isolation and Leakage Audit**:
   - Zero duplicate IDs across splits ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
   - Zero transcript overlap across splits (0 shared normalized texts).
   - All 90 cross-gender duplicate story titles (198 utterances, e.g., *"భువన సుందరి."*, *"నావికుడు సింద్ బాద్."*, *"వంశ గౌరవం."*) are 100% quarantined in the excluded pool.

4. **Split Target Durations Achieved**:
   - **Train**: **5.4557 hours** (Female: 2.4548h, Male: 3.0010h) — matches ~5h target with balanced speakers.
   - **Validation**: **1.0984 hours** (Female: 0.5013h, Male: 0.5971h) — matches ~1h target with balanced speakers.
   - **Test**: **5.3468 hours** (Female: 0.4454h, Male: 4.9014h) — held-out test set.
   - No audio segmentation or chopping is required; all selected utterances are clean, complete, short sentences $\le 15$ seconds.

5. **AI4Bharat IndicMFA Vocabulary Compatibility**:
   - IndicMFA uses letter-wise Grapheme-to-Grapheme (G2G) alignment where transcripts are formatted with spaces between every grapheme.
   - IndicMFA's `Telugu_Dictionary_g2g.txt` inventory contains 131 phones covering all elementary vowels, consonants, matras, virama, and modifiers in `Telugu/text/symbols.py`.
   - Discovered that 43 raw IndicTTS utterances contain ASCII Latin lowercase `'o'` (U+006F) as a typographical substitute for Telugu Anusvara `ం` (sunna). A normalization rule has been documented for Step 2.2B.

---

## 1. Full Dataset Audio Duration Reconciliation

### 1.1 Methodology

Every row across all 4 downloaded Parquet files (`train-00000-of-00004.parquet` to `train-00003-of-00004.parquet`) was queried using DuckDB and verified against `soundfile`:
- **Female files (gender 0)**: Stored as 16 kHz, 1-channel, 16-bit PCM WAV. Data rate = $16,000 \times 1 \times 2 = 32,000 \text{ bytes/sec}$.
- **Male files (gender 1)**: Stored as 48 kHz, 1-channel, 16-bit PCM WAV. Data rate = $48,000 \times 1 \times 2 = 96,000 \text{ bytes/sec}$.
- **WAV Header**: Each recording has a standard 44-byte RIFF/WAVE header.
- **Exact Duration Formula**:
  $$\text{Duration (seconds)} = \frac{\text{octet\_length}(\text{audio.bytes}) - 44}{\text{SampleRate} \times \text{Channels} \times (\text{BitsPerSample} / 8)}$$

### 1.2 Raw Dataset Distribution by Speaker

| Speaker / Gender | Shards Present | Utterance Count | Sampling Rate | Total Bytes | Decoded Audio Seconds | Decoded Audio Hours | Mean Duration |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Female (`female`, gender 0)** | Shard 0, Shard 1 | 4,043 | 16,000 Hz | 2,755,873,060 | 86,115.47 s | **23.9210 h** | 21.30 s |
| **Male (`male`, gender 1)** | Shard 1, Shard 2, Shard 3 | 4,533 | 48,000 Hz | 4,420,873,286 | 46,048.70 s | **12.7913 h** | 10.16 s |
| **Full Corpus Total** | Shards 0 to 3 | **8,576** | — | **7,176,746,346** | **132,164.18 s** | **36.7123 h** | 15.41 s |

> [!NOTE]
> The original dataset README claimed that the corpus totaled ~8.74 hours (~4.27h female, ~4.47h male). In reality, the female speaker alone provided nearly 24 hours of speech, consisting of extensive story paragraphs averaging 21.3 seconds per recording (with recordings extending up to 151.6 seconds).

---

## 2. Split Accounting Identities and Exclusion Breakdown

### 2.1 Complete Split Allocation Table

| Partition | Utterances | Female Utts | Male Utts | Estimated Dur (h) | Actual Decoded Dur (s) | Actual Decoded Dur (h) | Female Dur (h) | Male Dur (h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 2,330 | 972 | 1,358 | 4.9997 h | 19,640.68 s | **5.4557 h** | 2.4548 h | 3.0010 h |
| **Validation** | 457 | 192 | 265 | 0.9999 h | 3,954.19 s | **1.0984 h** | 0.5013 h | 0.5971 h |
| **Test** | 2,366 | 176 | 2,190 | 4.5485 h | 19,248.55 s | **5.3468 h** | 0.4454 h | 4.9014 h |
| **Included Total** | **5,153** | **1,340** | **3,813** | **10.5481 h** | **42,843.42 s** | **11.9009 h** | **3.4014 h** | **8.4995 h** |
| **Excluded Pool** | **3,423** | **2,703** | **720** | 0.0000 h | **89,320.76 s** | **24.8113 h** | **20.5196 h** | **4.2918 h** |
| **Original Total** | **8,576** | **4,043** | **4,533** | 10.5481 h | **132,164.18 s** | **36.7123 h** | **23.9210 h** | **12.7913 h** |

### 2.2 Proof of Accounting Identities

1. **Utterance Balance**:
   $$\text{Train (2,330)} + \text{Val (457)} + \text{Test (2,366)} = 5,153 \text{ Included}$$
   $$5,153 \text{ Included} + 3,423 \text{ Excluded} = 8,576 \text{ Total} \quad [\text{Verified}]$$

2. **Duration Balance (Decoded PCM Audio Seconds)**:
   $$19,640.68\text{s} + 3,954.19\text{s} + 19,248.55\text{s} = 42,843.42\text{s} \text{ Included} \quad [\text{Verified}]$$
   $$42,843.42\text{s} + 89,320.76\text{s} = 132,164.18\text{s} \text{ Total} \quad [\Delta = 0.000000\text{s}, \text{Verified}]$$

### 2.3 Excluded Pool Audit by Reason

| Exclusion Category | Utterance Count | Female Utts | Male Utts | Actual Audio Seconds | Actual Audio Hours | Female Hours | Male Hours |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Over-length ($> 15.0\text{s}$)** | 3,199 | 2,605 | 594 | 88,600.84 s | 24.6113 h | 20.3978 h | 4.2136 h |
| **Cross-gender duplicate title** | 198 | 90 | 108 | 674.00 s | 0.1872 h | 0.1131 h | 0.0742 h |
| **Sub-length ($< 1.0\text{s}$)** | 15 | 8 | 7 | 25.52 s | 0.0071 h | 0.0063 h | 0.0008 h |
| **Intra-gender duplicate text** | 5 | 0 | 5 | 10.46 s | 0.0029 h | 0.0024 h | 0.0005 h |
| **Marginal duration boundary** | 6 | 0 | 6 | 9.94 s | 0.0028 h | 0.0000 h | 0.0028 h |
| **Total Excluded** | **3,423** | **2,703** | **720** | **89,320.76 s** | **24.8113 h** | **20.5196 h** | **4.2918 h** |

---

## 3. Split Isolation and Leakage Verification

An automated verification script tested every source ID and transcript across the dataset partitions:

1. **Source ID Uniqueness**:
   - `train.txt`: 2,330 unique IDs (0 duplicates)
   - `val.txt`: 457 unique IDs (0 duplicates)
   - `test.txt`: 2,366 unique IDs (0 duplicates)
   - Global ID uniqueness: 8,576 distinct paths across the entire dataset.

2. **Split Intersections**:
   - $\text{Train} \cap \text{Validation} = \emptyset$ (0 intersecting IDs)
   - $\text{Train} \cap \text{Test} = \emptyset$ (0 intersecting IDs)
   - $\text{Validation} \cap \text{Test} = \emptyset$ (0 intersecting IDs)

3. **Text Leakage Check**:
   - All text transcripts were normalized via `Telugu/text/normalizer.py`.
   - $\text{Train}_{\text{text}} \cap \text{Val}_{\text{text}} = \emptyset$ (0 shared sentences)
   - $\text{Train}_{\text{text}} \cap \text{Test}_{\text{text}} = \emptyset$ (0 shared sentences)
   - $\text{Val}_{\text{text}} \cap \text{Test}_{\text{text}} = \emptyset$ (0 shared sentences)

4. **Shared Story Titles Quarantine**:
   - 93 texts appeared multiple times in the raw corpus (e.g., chapter titles read by both speakers: *"భువన సుందరి."*, *"నావికుడు సింద్ బాద్."*, *"వంశ గౌరవం."*, *"తోక చుక్క."*, *"సూర్యుడు."*).
   - All 93 repeated texts (206 utterances total, including 90 cross-gender shared titles comprising 198 utterances) were quarantined in the excluded pool. Exactly 0 duplicate texts enter Train, Val, or Test.

---

## 4. Telugu Vocabulary and AI4Bharat IndicMFA Audit

### 4.1 IndicMFA G2G Alignment Architecture

AI4Bharat's IndicMFA employs a letter-level Grapheme-to-Grapheme (G2G) methodology rather than a traditional Grapheme-to-Phoneme (G2P) lexicon. In IndicMFA's preprocessing:
1. Transcripts in `.lab` files have spaces inserted between individual characters/graphemes (e.g., `క ా క ి`).
2. MFA treats each space-separated token as a "word".
3. The pronunciation dictionary `Telugu_Dictionary_g2g.txt` maps each grapheme to itself as a phone (e.g., `క క`, `ా ా`).
4. In TextGrid post-processing, silence tokens (`<eps>`) and punctuation are reconciled.

### 4.2 Dictionary Inventory Comparison

The official `Telugu_Dictionary_g2g.txt` (285 entries, 131 unique phones) was analyzed against `Telugu/text/symbols.py` (101 symbols):

| Category | Symbols in `symbols.py` | Presence in IndicMFA Dictionary | Notes |
| :--- | :---: | :---: | :--- |
| **Independent Vowels (అచ్చులు)** | 16 | All 16 present (`అ`, `ఆ`, `ఇ`, `ఈ`, `ఉ`, `ఊ`, `ఋ`, `ౠ`, `ఌ`, `ౡ`, `ఎ`, `ఏ`, `ఐ`, `ఒ`, `ఓ`, `ఔ`) | 100% match |
| **Consonants (హల్లులు)** | 36 | All 36 present (`క` to `హ`, including `ఱ`, `ళ`, `ఴ`) | 100% match |
| **Vowel Matras (గుణింతాలు)** | 15 | All 15 present in bare form (`ా`..`ౌ`) and dotted-circle form (`◌ా`..`◌ౌ`) | Both forms mapped |
| **Virama (హలంతము)** | 1 | Present as bare `్` and dotted-circle `◌్` | Also appears in conjunct entries |
| **Modifiers** | 4 | `ం` (sunna), `ః` (visarga), `ఁ` (arasunna), `ఽ` (avagraha) present | 100% match |
| **Telugu Numerals** | 10 | Absent in MFA dictionary | Not used in IndicTTS transcripts |
| **Punctuation** | 12 | MFA strips punctuation during alignment; handled via `<eps>` | Retained in TTS symbols |
| **Special Tokens** | 4 | `_` (PAD), `<unk>`, `<bos>`, `<eos>` | Neural model tokens |
| **Silences** | 3 | `@sp`, `@spn`, `@sil` | MFA alignment phone tokens |

### 4.3 Raw Corpus Text Anomaly: ASCII `'o'` Substitution

A character frequency scan across all 8,576 utterances in `text_manifest.json` revealed:
- 74 unique characters in raw text.
- 43 utterances contain the Latin lowercase letter `'o'` (U+006F) with 46 occurrences.
- Context inspection demonstrates that the transcribers used `'o'` as a typographical shorthand for the Telugu Anusvara `ం` (sunna / circle):
  - Example: `వీళుoటే` instead of `వీళుంటే`
  - Example: `పేరేoటి` instead of `పేరేంటి`
  - Example: `అనిపించిoది` instead of `అనిపించింది`
  - Example: `మిమ్మల్ నoదరినీ` instead of `మిమ్మల్ నందరినీ`
- **Mitigation Requirement for Step 2.2B**: Update `Telugu/text/normalizer.py` to replace ASCII `'o'` with Telugu Anusvara `ం` (`U+0C02`) when embedded in Telugu words, ensuring zero tokens are mapped to `<unk>`.

---

## 5. Feasibility of Split Duration Targets

The assignment requirements specify:
- Approximately 5 hours of training data (~2.5h Female + ~2.5h Male).
- Approximately 1 hour of validation data (~0.5h Female + ~0.5h Male).
- Separate held-out test set.

### 5.1 Verification Against Requirements

1. **Training Partition**:
   - Actual: **5.4557 hours**
   - Female: **2.4548 hours** (98.2% of 2.50h target)
   - Male: **3.0010 hours** (120% of 2.50h target)
   - *Status: Fully compliant with ~5 hour requirement.*

2. **Validation Partition**:
   - Actual: **1.0984 hours**
   - Female: **0.5013 hours** (100.3% of 0.50h target)
   - Male: **0.5971 hours** (119% of 0.50h target)
   - *Status: Fully compliant with ~1 hour requirement.*

3. **Held-Out Test Partition**:
   - Actual: **5.3468 hours** (Female: 0.4454h, Male: 4.9014h, 2,366 utterances)
   - *Status: Generous held-out test set available.*

4. **Zero Segmentation Risk**:
   - Because 1,340 clean female utterances and 3,813 clean male utterances were identified within the $\le 15.0\text{s}$ duration limit, all split requirements are met using **complete, natural utterances**.
   - No aggressive audio slicing, VAD cutting, or unsafe segmentation is necessary.

---

## 6. Regression Testing

A dedicated audit test suite was created in `tests/test_telugu_split_vocabulary_audit.py` to ensure continuous verification:
- `test_duration_accounting_identities`: Verifies all mathematical duration identities.
- `test_split_id_uniqueness_and_isolation`: Verifies ID uniqueness and disjoint split sets.
- `test_zero_text_leakage_between_splits`: Verifies no text overlaps.
- `test_cross_gender_duplicate_quarantine`: Verifies shared chapter headings are excluded.
- `test_split_target_durations`: Verifies duration compliance.
- `test_vocabulary_completeness_and_special_tokens`: Verifies symbols and tokens.
- `test_mfa_telugu_dictionary_inventory_compatibility`: Verifies dictionary compatibility.

**Full Regression Test Suite Result:**
```
Ran 189 tests in 5.604s
OK (189/189 passing)
```

---

## 7. Audit Decision & Next Steps

### Official Audit Status:
**READY FOR TELUGU ALIGNMENT PILOT**

### Pre-conditions for Next Stage (Step 2.2B / 2.3):
1. **Normalizer Update**: Add regex in `Telugu/text/normalizer.py` to normalize typographical ASCII `'o'` into Telugu Anusvara `ం` (`U+0C02`).
2. **Acoustic Model Download**: Download official AI4Bharat `Telugu_Acoustic_Model.zip` (61.4 MB) from GitHub releases to `data/telugu_mfa_model/`.
3. **Alignment Pilot**: Run a controlled 25-utterance forced alignment pilot on resampled 22.05 kHz audio using `mfa align` with `Telugu_Dictionary_g2g.txt` and `Telugu_Acoustic_Model.zip`.

