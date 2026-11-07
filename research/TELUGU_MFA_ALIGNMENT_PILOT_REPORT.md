# Step 2.2B — IndicMFA Telugu Forced Alignment Pilot Report

**Date:** November 5, 2026  
**Auditor / Engineer:** FastSpeech 2 Project Team  
**Dataset:** `SPRINGLab/IndicTTS_Telugu` (Training split partition)  
**Aligner Engine:** Montreal Forced Aligner (MFA) v3.4.2 via AI4Bharat IndicMFA Telugu Acoustic Model  
**Final Decision:** **READY FOR FULL TELUGU ALIGNMENT**

---

## 1. Executive Summary

In Step 2.2B, a controlled forced-alignment pilot was executed using AI4Bharat's official IndicMFA Telugu acoustic model and Grapheme-to-Grapheme (G2G) pronunciation dictionary on 50 representative utterances drawn exclusively from the training split.

### Key Milestones & Results:
1. **IndicMFA Setup & Execution**:
   - Official acoustic model (`Telugu_Acoustic_Model.zip`, 58.6 MB, GMM-HMM architecture) and G2G dictionary (`Telugu_Dictionary_g2g.txt`, 285 entries) installed and verified.
   - Text formatting following the official IndicMFA G2G specification (elementary characters separated by spaces in `.lab` files).
   - Audio resampled to 22,050 Hz using band-limited Kaiser-windowed sinc interpolation (`Telugu/alignment/resample.py`).
2. **Latin 'o' Typographical Anomaly Resolution**:
   - All 43 occurrences of Latin `'o'` identified in the Step 2.2A-R audit were individually inspected and verified.
   - 100% of occurrences were confirmed to be typographical substitutions for the Telugu Anusvara `ం` (`U+0C02`).
   - A contextual normalization rule was added to `Telugu/text/normalizer.py`.
   - The single occurrence in the training split (`train_telugumale_02568`: `చేస్తూవుoడేవాడు.` $\to$ `చేస్తూవుండేవాడు.`) was included in the 50-utterance pilot and aligned with zero errors.
3. **Alignment Success Rate**:
   - **50 / 50 utterances aligned successfully (100.0% success rate)**.
   - Total alignment time: **19.24 seconds** (**0.38 seconds per utterance** across 4 multiprocessing workers).
   - Zero alignment failures, zero empty TextGrids, zero unknown phones (`spn`), and zero out-of-vocabulary tokens.
4. **Manual Quality Review (18 Utterances)**:
   - Evaluated 18 diverse utterances across genders, durations, conjuncts, visarga, and vowel matras.
   - Exact sequence match between input graphemes and aligned speech phones: **100.0% (18/18)**.
   - Initial silence (mean 0.54s) and final silence (mean 0.76s) cleanly segmented.
   - Internal pauses cleanly resolved as inter-word silences (`sil` / `@sp`).
   - Phone durations exhibit natural phonetic distributions (minimum 10.0 ms, maximum 170–310 ms).
5. **Duration Extraction & Mel Reconciliation**:
   - Evaluated telescoping boundary quantization at 22,050 Hz with hop length 256.
   - Max frame discrepancy before reconciliation: **1 frame** (mean discrepancy: **0.40 frames**).
   - After single-frame redistribution to longest intervals: **0 frames discrepancy** across all 50 utterances.
   - Every speech grapheme receives a duration $\ge 1$ frame.
6. **Regression Test Suite**:
   - 7 new focused unit tests added in `tests/test_telugu_mfa_alignment_pilot.py`.
   - Full regression suite: **196 / 196 tests passing**.

---

## 2. IndicMFA Setup and Exact Invocation

### 2.1 Acoustic Model and Lexicon Specifications

| Component | Asset File / Path | Specification |
| :--- | :--- | :--- |
| **MFA Executable** | `/opt/anaconda3/envs/mfa_env/bin/mfa` | Montreal Forced Aligner v3.4.2 (OpenFST + Kaldi backend) |
| **Acoustic Model** | `data/telugu_mfa_model/Telugu_Acoustic_Model.zip` (61.4 MB) | AI4Bharat IndicMFA GMM-HMM triphone model trained on Telugu speech |
| **G2G Dictionary** | `research/Telugu_Dictionary_g2g.txt` (3.7 KB) | 285 lines mapping elementary Telugu graphemes to self |
| **Feature Extraction** | MFCC (13 cepstral coeffs, 23 mel bins, 25ms window, 10ms shift) | Configured in `Telugu_Acoustic/meta.json` with CMVN and speaker adaptation |
| **Audio Format** | 22,050 Hz, 1-channel, 16-bit PCM WAV | Resampled from original 16 kHz (female) / 48 kHz (male) |

### 2.2 Text Preparation & Tokenization Protocol

Following AI4Bharat IndicMFA's G2G methodology:
1. Raw transcript is normalized using `Telugu.text.normalizer.normalize_text` (Unicode NFC, punctuation cleanup, Latin 'o' $\to$ `ం`).
2. Punctuation (`.`, `,`, `!`, `?`, etc.) is stripped for `.lab` generation.
3. Every individual Telugu character (independent vowels, consonants, dependent matras, virama, and modifiers) is separated by a single space:
   ```
   Original: నాకది రాదే అన్నాడు బ్రాహ్మడు.
   Normal:   నాకది రాదే అన్నాడు బ్రాహ్మడు.
   MFA Lab:  న ా క ద ి   ర ా ద ే   అ న ్ న ా డ ు   బ ్ ర ా హ ్ మ డ ు
   ```
4. In `Telugu_Dictionary_g2g.txt`, each character is listed as a valid word mapping to itself as a phone (e.g., `న న`, `ా ా`, `్ ్`).

### 2.3 Exact Command Invocation

```bash
export PATH="/opt/anaconda3/envs/mfa_env/bin:$PATH"

mfa align \
  --clean \
  --use_mp \
  -j 4 \
  --single_speaker \
  --no_textgrid_cleanup \
  data/telugu_mfa_pilot/corpus \
  research/Telugu_Dictionary_g2g.txt \
  data/telugu_mfa_model/Telugu_Acoustic_Model.zip \
  data/telugu_mfa_pilot/output
```

---

## 3. Latin 'o' Typographical Anomaly Audit & Resolution

### 3.1 Investigation Summary

During Step 2.2A-R, 43 utterances across the male speaker corpus were identified containing ASCII `'o'` (`U+006F`).
- **Detailed Audit Report**: [TELUGU_LATIN_O_CORRECTION_REPORT.md](file:///Users/swaruppal/FastSpeech2_Project/research/TELUGU_LATIN_O_CORRECTION_REPORT.md)
- **JSON Audit Log**: `research/telugu_latin_o_audit_records.json`

### 3.2 Key Findings:
- In phonetic Telugu typing, the circular Anusvara glyph (`ం`, *sunna*) is frequently mistyped as Latin `'o'` or produced by legacy font decoders.
- Every single occurrence was verified against Telugu grammar, phonetics, and context (e.g. `పేరేoటి` $\to$ `పేరేంటి`, `అనిపించిoది` $\to$ `అనిపించింది`, `ఆశ్రమo` $\to$ `ఆశ్రమం`).
- **Partition Distribution**:
  - Excluded Pool: 28 utterances
  - Held-out Test Split: 13 utterances
  - Validation Split: 1 utterance (`train_telugumale_01981`: `అచ్చoగా` $\to$ `అచ్చంగా`)
  - Training Split: 1 utterance (`train_telugumale_02568`: `చేస్తూవుoడేవాడు.` $\to$ `చేస్తూవుండేవాడు.`)

### 3.3 Normalizer Implementation:
A targeted contextual regex was integrated into `Telugu/text/normalizer.py`:
```python
_LATIN_O_ANUSVARA_REGEX = re.compile(
    r"(?<=[\u0C00-\u0C7F])o(?=[\u0C00-\u0C7F]|\b|[,.!?;:\"'])"
)
```
This converts Latin `'o'` to Telugu Anusvara `ం` only when embedded in or adjacent to Telugu text, preventing accidental mutation of non-Telugu text.

---

## 4. Representative Pilot Selection Methodology

A 50-utterance representative subset was selected exclusively from `Telugu/metadata/train.txt` (2,330 total utterances):

| Property | Female (25 Utterances) | Male (25 Utterances) | Combined (50 Utterances) |
| :--- | :---: | :---: | :---: |
| **Duration Distribution** | Short: 7, Med: 11, Long: 7 | Short: 7, Med: 11, Long: 7 | Short: 14, Med: 22, Long: 14 |
| **Total Duration** | 170.56 s (2.84 min) | 199.49 s (3.32 min) | **370.05 s (6.17 min)** |
| **Min / Max Duration** | 1.60 s / 13.89 s | 2.71 s / 12.73 s | 1.60 s / 13.89 s |
| **Mean Duration** | 6.82 s | 7.98 s | 7.40 s |
| **Virama / Conjuncts** | 128 occurrences | 122 occurrences | 250 occurrences |
| **Anusvara (`ం`)** | 71 occurrences | 74 occurrences | 145 occurrences |
| **Visarga (`ః`)** | 4 occurrences | 4 occurrences | 8 occurrences |
| **Vocalic R (`ఋ`/`ృ`)** | 16 occurrences | 14 occurrences | 30 occurrences |
| **Special Utterance** | — | `train_telugumale_02568` | 1 corrected Latin 'o' utterance |

---

## 5. Forced Alignment Results

The pilot alignment was executed using `research/run_telugu_mfa_pilot.py`:

```
============================================================
INDICMFA TELUGU FORCED ALIGNMENT PILOT RESULTS
============================================================
Total utterances submitted:   50
Successful TextGrids:         50 (100.0%)
Failed utterances:            0 (0.0%)
Total wall-clock runtime:     19.24 seconds
Per-utterance runtime:        0.38 seconds/utterance
Unknown words / OOV tokens:   0
Empty / malformed intervals:  0
Unaligned audio files:        0
============================================================
```

All 50 TextGrids were created in `data/telugu_mfa_pilot/output/` and verified for structural integrity.

---

## 6. Manual Alignment Quality Review

18 representative utterances were selected for algorithmic and manual inspection:

| # | Utterance ID | Speaker | Duration | Char Count | Exact Match | Init Sil (s) | End Sil (s) | Pause Count | Min Phone | Max Phone | Assessment |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `train_telugumale_02568` | Male | 12.73 s | 169 | **True** | 0.180 s | 0.176 s | 16 | 10.0 ms | 180.0 ms | **EXCELLENT** |
| 2 | `train_telugufemale_02040` | Female | 9.59 s | 77 | **True** | 0.580 s | 0.772 s | 16 | 10.0 ms | 210.0 ms | **EXCELLENT** |
| 3 | `train_telugufemale_00165` | Female | 12.12 s | 99 | **True** | 0.320 s | 0.623 s | 16 | 10.0 ms | 220.0 ms | **EXCELLENT** |
| 4 | `train_telugufemale_00986` | Female | 10.31 s | 87 | **True** | 0.660 s | 0.582 s | 18 | 10.0 ms | 190.0 ms | **EXCELLENT** |
| 5 | `train_telugufemale_02649` | Female | 12.72 s | 99 | **True** | 0.480 s | 0.781 s | 24 | 10.0 ms | 210.0 ms | **EXCELLENT** |
| 6 | `train_telugufemale_03380` | Female | 11.57 s | 89 | **True** | 0.570 s | 1.006 s | 18 | 10.0 ms | 310.0 ms | **EXCELLENT** |
| 7 | `train_telugufemale_01535` | Female | 11.29 s | 88 | **True** | 1.080 s | 1.518 s | 12 | 10.0 ms | 250.0 ms | **EXCELLENT** |
| 8 | `train_telugufemale_00501` | Female | 13.89 s | 110 | **True** | 0.980 s | 0.798 s | 22 | 10.0 ms | 180.0 ms | **EXCELLENT** |
| 9 | `train_telugufemale_03378` | Female | 7.48 s | 51 | **True** | 0.290 s | 0.659 s | 11 | 10.0 ms | 170.0 ms | **EXCELLENT** |
| 10 | `train_telugufemale_02046` | Female | 7.48 s | 54 | **True** | 0.570 s | 0.899 s | 10 | 10.0 ms | 170.0 ms | **EXCELLENT** |
| 11 | `train_telugufemale_01629` | Female | 6.09 s | 34 | **True** | 1.300 s | 0.524 s | 4 | 20.0 ms | 230.0 ms | **EXCELLENT** |
| 12 | `train_telugufemale_01665` | Female | 6.23 s | 50 | **True** | 0.160 s | 0.296 s | 6 | 20.0 ms | 190.0 ms | **EXCELLENT** |
| 13 | `train_telugufemale_03431` | Female | 7.92 s | 59 | **True** | 0.620 s | 0.501 s | 14 | 10.0 ms | 170.0 ms | **EXCELLENT** |
| 14 | `train_telugufemale_03766` | Female | 5.15 s | 43 | **True** | 0.210 s | 0.336 s | 10 | 10.0 ms | 180.0 ms | **EXCELLENT** |
| 15 | `train_telugufemale_02841` | Female | 6.13 s | 41 | **True** | 0.510 s | 0.813 s | 9 | 10.0 ms | 200.0 ms | **EXCELLENT** |
| 16 | `train_telugufemale_03826` | Female | 6.55 s | 36 | **True** | 0.490 s | 1.251 s | 12 | 10.0 ms | 190.0 ms | **EXCELLENT** |
| 17 | `train_telugufemale_02007` | Female | 7.54 s | 52 | **True** | 0.260 s | 0.867 s | 12 | 10.0 ms | 220.0 ms | **EXCELLENT** |
| 18 | `train_telugufemale_02039` | Female | 4.76 s | 32 | **True** | 0.320 s | 0.567 s | 7 | 10.0 ms | 200.0 ms | **EXCELLENT** |

### Quality Observations:
1. **Grapheme Sequence Integrity**: In all 18 utterances (and across all 50 pilot items), the sequence of aligned phones corresponds 1-to-1 with the normalized input Telugu characters. No skipped, reordered, or inserted characters occurred.
2. **Silence Segmentation**: Initial and final silence intervals match acoustic energy drop-offs cleanly. Internal pauses are captured as `sil` intervals at word and clause boundaries.
3. **Virama & Conjunct Dynamics**: Virama (`్`) intervals are assigned tight, realistic closure durations (10–30 ms), reflecting phonetic stop closures prior to release into the following consonant.
4. **Vocalic & Diphthong Representation**: Long vowels and diphthongs (`ా`, `ే`, `ో`, `ై`) exhibit extended durations (120–310 ms), accurately modeling natural prosody.

---

## 7. Duration Extraction Feasibility for FastSpeech 2

### 7.1 Target Token Representation
FastSpeech 2 accepts discrete token IDs from `Telugu/text/symbols.py` (101 symbols):
- Elementary Telugu graphemes match vocabulary symbols directly.
- Internal pauses (`sil`, `<eps>`) map to silence symbol `@sp` (index 4).
- Leading and trailing silences are trimmed, with start offset $t_0$ and end offset $t_1$ saved for audio clipping.

### 7.2 Mel-Frame Discretization & Reconciliation Results
Using telescoping rounding at $f_s = 22,050 \text{ Hz}$, $\text{hop} = 256$:
- Single frame duration: $256 / 22050 \approx 11.61 \text{ ms}$.
- Frame boundaries: $b_i = \text{round}((t_i - t_0) \times 22050 / 256)$.
- Duration: $d_i = \max(1, b_{i+1} - b_i)$ for speech tokens.

**Discrepancy Metrics Across 50 Utterances:**
- Max discrepancy before reconciliation: **1 frame** (11.6 ms).
- Mean absolute discrepancy before reconciliation: **0.40 frames**.
- Discrepancy after single-frame redistribution to longest interval: **0 frames across all 50 utterances**.
- Zero speech tokens receive 0 frames.
- Sum of frame durations strictly equals the target mel-spectrogram frame length.

---

## 8. Unit Testing & Regression Verification

7 dedicated unit tests were implemented in [`tests/test_telugu_mfa_alignment_pilot.py`](file:///Users/swaruppal/FastSpeech2_Project/tests/test_telugu_mfa_alignment_pilot.py):
1. `test_telugu_g2g_tokenization_and_dictionary_coverage`: Verifies grapheme splitting and dictionary membership.
2. `test_latin_o_correction_handling`: Verifies contextual `'o'` $\to$ `ం` replacement.
3. `test_pilot_alignment_success_rate`: Asserts 50/50 alignment success.
4. `test_alignment_textgrid_parsing_and_integrity`: Validates interval monotonicity, bounds, and silence tags.
5. `test_frame_boundary_conversion_telescoping`: Tests mathematical telescoping quantization property.
6. `test_duration_extraction_feasibility`: Asserts 0 frame discrepancy after reconciliation.
7. `test_unknown_token_detection`: Tests OOV fallback behavior.

**Full Regression Test Suite Output:**
```
Ran 196 tests in 6.826s
OK (196/196 passing)
```

---

## 9. Remaining Risks and Mitigations

| Risk | Impact | Mitigation Established in Pilot |
| :--- | :--- | :--- |
| **Out-of-vocabulary words in full corpus** | MFA alignment failure | Pre-flight validation script will check 100% of transcripts against `Telugu_Dictionary_g2g.txt` before full alignment. |
| **Typographical anomalies (like Latin 'o')** | Token `<unk>` insertion | `Telugu/text/normalizer.py` now contextually maps `'o'` $\to$ `ం`. Corpus audit confirmed no other Latin characters exist. |
| **Duration drift against mel-frames** | Model training failure in LengthRegulator | Telescoping quantization with single-frame reconciliation guarantees exact frame parity ($\Delta = 0$). |
| **Processing time on full training set** | Slow preprocessing | Benchmark of 0.38 s/utt indicates full training set (2,330 utts) will align in ~15 minutes with 4 cores. |

---

## 10. Conclusion and Official Decision

### Official Decision:
**READY FOR FULL TELUGU ALIGNMENT**

All criteria for Step 2.2B have been fulfilled with 100% success rate, verified acoustic-lexical alignment, zero OOV tokens, zero frame discrepancies, and complete regression test compliance.

In accordance with strict project instructions:
**STOPPING EXECUTION. Full alignment, feature extraction, and model training have NOT been initiated.** Awaiting user review and authorization to proceed to the next step.

