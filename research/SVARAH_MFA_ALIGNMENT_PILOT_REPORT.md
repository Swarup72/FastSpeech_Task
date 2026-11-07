# Svarah MFA Alignment Pilot & Duration Extraction Validation Report

**Assignment:** Implementation of FastSpeech 2-Based Text-to-Speech Synthesis  
**Stage:** Stage 2 — Step 1.4J-C  
**Date:** 4 November 2026  
**Author / Execution:** Automated Research & Development Pipeline  
**Dataset:** `ai4bharat/Svarah` (Indian English Subset)  
**Pilot Cohort:** 100 Genuine Training Utterances (`English/metadata/pilot_train_100.json`)  
**Alignment Suite:** Montreal Forced Aligner (MFA) v3.4.2 (Native macOS ARM64 / Apple Silicon)  

---

## Executive Summary

As part of the preprocessing pipeline for the FastSpeech 2 English TTS model, we executed a rigorous end-to-end forced alignment pilot on **100 genuine Svarah training utterances**.

### Key Results
- **MFA Installation:** Successfully deployed Montreal Forced Aligner **v3.4.2** in an isolated Conda environment (`/opt/anaconda3/envs/mfa_env`) natively supporting Apple Silicon ARM64 with compiled Kaldi and OpenFST C++ binaries.
- **Utterance Selection:** 100 utterances strictly sampled from `English/metadata/train.txt` (never touching validation or test sets), spanning **50 distinct speaker profiles** (54 male, 46 female) across **19 primary Indian language backgrounds**, with durations ranging from 1.10s to 13.54s (mean: 5.52s, total: 9.20 minutes).
- **OOV Detection & G2P:** MFA identified **48 out-of-vocabulary (OOV)** words (primarily Indian proper nouns such as *covaxin*, *fastag*, *hoysala*, *hotstar*, *jammu*, *kathua*, *rajouri*). Using the pretrained `english_us_arpa` G2P model, **100% (48/48)** were successfully transcribed into phonologically accurate ARPAbet pronunciations and merged into a pilot pronunciation lexicon of 208,781 words.
- **Forced Alignment Success Rate:** **100 out of 100 utterances aligned successfully (100.0% success rate)** in 59.4 seconds on 4 CPU workers. Zero utterances failed or were dropped.
- **Duration Extraction & Telescoping Discretization:** Implemented `English/alignment/duration.py`. Telescoping boundary quantization completely eliminated cumulative drift ($sum(d_i) == \text{expected\_frames}$ across 100% of files).
- **Mel-Length Reconciliation:** Implemented deterministic fractional-residual reconciliation. Tested against 80-bin mel spectrograms ($f_s=22050, N_{\text{fft}}=1024, H=256, W=1024$). Exactly **100% of utterances** achieved perfect mathematical frame equality ($\sum d_i == T_{\text{mel}}$).
- **FastSpeech 2 Vocabulary & Architecture Compatibility:** All 63 unique phoneme labels produced by the aligner (4,918 total phoneme tokens) exist in FastSpeech 2's `symbols.py` vocabulary. Zero unknown tokens. Formatted dataset strings parse cleanly with `text_to_sequence`.
- **Regression Suite:** **165/165 tests pass** across the entire codebase (`tests/test_duration_extraction.py`, `tests/test_english_normalizer.py`, `tests/test_svarah_manifests.py`, and core model tests).

---

## 1. MFA Installation and Environment Configuration

### Installation Architecture
Contrary to earlier assumptions that MFA required Linux or Google Colab due to Kaldi build constraints, Montreal Forced Aligner v3.4.2 is fully supported natively on macOS ARM64 via `conda-forge`:

```bash
# Environment setup
conda create -n mfa_env -c conda-forge montreal-forced-aligner python=3.11
```

### Environment Verification
- **Binary Path:** `/opt/anaconda3/envs/mfa_env/bin/mfa`
- **MFA Version:** `3.4.2`
- **Python Version:** Python 3.11.11 (inside conda env)
- **Kaldi / OpenFST Binaries:** Verified native ARM64 executables (`fstcompile`, `fstdeterminize`, `fstarcsort`, `lattice-align-words`) located under `/opt/anaconda3/envs/mfa_env/bin/`.
- **Operational Integration:** The primary project virtualenv (`.venv/bin/python`, Python 3.10) remains completely isolated and untouched. Execution of MFA commands simply prefixes `PATH="/opt/anaconda3/envs/mfa_env/bin:$PATH"`.

---

## 2. Acoustic Model, Dictionary, and G2P Model Versions

All models were downloaded directly from the official Montreal Forced Aligner acoustic and language model repositories:

| Component | Model Name | Version / Architecture | Size / Vocabulary | Local Storage Path |
| :--- | :--- | :--- | :--- | :--- |
| **Acoustic Model** | `english_us_arpa` | Kaldi GMM-HMM (SAT / LDA+MLLT) | 26.6 MB | `~/Documents/MFA/pretrained_models/acoustic/english_us_arpa.zip` |
| **Pronunciation Dictionary** | `english_us_arpa` | CMUDict-derived ARPAbet | 208,651 word entries | `~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict` |
| **G2P Model** | `english_us_arpa` | OpenFST Pynini G2P transducer | 38.3 MB | `~/Documents/MFA/pretrained_models/g2p/english_us_arpa.zip` |

---

## 3. Pilot Utterance Selection

### Selection Criteria
- **Dataset Source:** Strictly sampled from `English/metadata/train.txt` (never touching validation or test splits).
- **Sampling Window:** Selected from Row Groups 0 to 3 of shard 0 of `ai4bharat/Svarah`, ensuring rapid parquet row-group downloading while maintaining diversity.
- **Sample Count:** Exactly 100 training utterances.
- **Gender Balance:** 54 Male, 46 Female.
- **Demographic Representation:** 50 distinct speaker profiles out of the 115 total training profiles.
- **Language Backgrounds:** All 19 primary Indian mother tongues present in the dataset (Hindi, Bengali, Telugu, Tamil, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi, Assamese, Maithili, Nepali, Urdu, Dogri, Kashmiri, Konkani, Manipuri, Sindhi).
- **Duration Statistics:**
  - Shortest utterance: 1.10 seconds (`281474976885303_f2095_chunk_0`)
  - Longest utterance: 13.54 seconds (`281474976884639_f1681_chunk_0`)
  - Mean duration: 5.52 seconds
  - Median duration: 4.88 seconds
  - Total duration: 551.98 seconds (9.20 minutes)

### Manifest Deliverable
The complete pilot cohort manifest is recorded in:
- `English/metadata/pilot_train_100.json`
- `English/metadata/pilot_train_100.csv`

---

## 4. OOV Detection and G2P Resolution

### OOV Discovery
Running `mfa find_oovs` against the 100 pilot transcripts identified **48 unique out-of-vocabulary words** across 66 occurrences. These words fall into three distinct classes:
1. **Indian Administrative & Geographic Names:** *kathua*, *rajouri*, *hasana*, *hoysala*, *odisha*, *jammu*, *gandhinagar*, *baramulla*.
2. **Indian Financial, Tech & Cultural Terms:** *fastag*, *covaxin*, *hotstar*, *chana*, *jain*, *tapasvi*, *bhabhi*, *bhel*.
3. **Hyphenated / Compound Words:** *counter-attacking*, *three-year*, *all-time*, *high-level*.

### G2P Transcription Results
Using `mfa g2p english_us_arpa`, all 48 OOV words were transcribed into standard ARPAbet phonemes:

| OOV Word | Generated ARPAbet Transcription | Phonological Assessment |
| :--- | :--- | :--- |
| `covaxin` | `K OW0 V AE1 K S IH0 N` | Accurate stress and vowel reduction |
| `fastag` | `F AE1 S T AE2 G` | Correct compound vowel assignment |
| `hoysala` | `HH OY0 S AA1 L AH0` | Correct diphthong and open vowel |
| `hasana` | `HH AA0 S AA1 N AH0` | Standard loanword phonology |
| `chana` | `CH AA1 N AH0` | Accurate Indian English vowel representation |
| `hotstar` | `HH AA1 T S T AA2 R` | Natural compound stress |
| `odisha` | `OW0 D IH1 SH AH0` | Correct modern state name pronunciation |
| `rajouri` | `R AA0 JH AW1 R IY0` | Accurate representation of affricate and vowel |
| `kathua` | `K AA0 TH UW1 AH0` | Accurate dental fricative and vowel hiatus |
| `counter-attacking` | `K AW1 N T ER0 AH0 T AE1 K IH0 NG` | Excellent multi-syllabic stress resolution |

### Pronunciation Dictionary Merge
All 48 G2P pronunciations were merged with the base 208,651-entry lexicon, producing `data/svarah_pilot_100/pilot_dictionary.txt` (208,781 lines). This eliminated 100% of OOV blockers prior to forced alignment.

---

## 5. Forced Alignment Performance & Quality Analysis

### Alignment Execution
Forced alignment was executed using:
```bash
mfa align \
  data/svarah_pilot_100/corpus \
  data/svarah_pilot_100/pilot_dictionary.txt \
  english_us_arpa \
  data/svarah_pilot_100/textgrids \
  --clean --num_jobs 4
```

### Performance Metrics
- **Total Utterances Submitted:** 100
- **Total TextGrids Generated:** 100
- **Success Rate:** **100.0% (100 / 100)**
- **Failed / Dropped Utterances:** 0 (0.0%)
- **Alignment Wall Time:** 59.4 seconds (~0.59s per utterance)
- **Output Files:** `data/svarah_pilot_100/textgrids/spk_XXX/*.TextGrid`

### TextGrid Quality Analysis
- **Phone Intervals Extracted:** 4,918 total phoneme intervals across the 100 files.
- **Silence & Pause Structure:**
  - MFA 3.4.2 writes Praat TextGrids containing `words` and `phones` IntervalTiers.
  - Spoken phonemes have standard ARPAbet labels (`HH`, `W`, `AH1`, `T`, `S`, etc.).
  - Pauses between words and at utterance boundaries are written as intervals with empty text label `""`.
  - When parsed with `tgt.io.read_textgrid(..., include_empty_intervals=True)`, all silence intervals are explicitly captured.
  - There are 427 total silence intervals across the 100 pilot files (176 boundary silences, 251 internal inter-word pauses).
  - No negative time intervals or temporal inversions were observed.

---

## 6. Duration Extraction and Mel Reconciliation Architecture

### Standalone Duration Utility
We implemented `English/alignment/duration.py` and exposed it via `English/alignment/__init__.py`.

### Mathematical Formulation of Discretization
Given sampling rate $f_s = 22050$ Hz and hop length $H = 256$ samples:
Each interval $[s_i, e_i]$ is discretized using absolute boundary rounding relative to the trimmed start offset $t_{\text{start}}$:
$$b_i = \left\lfloor \frac{(s_i - t_{\text{start}}) \times f_s}{H} + 0.5 \right\rfloor$$
$$b_{i+1} = \left\lfloor \frac{(e_i - t_{\text{start}}) \times f_s}{H} + 0.5 \right\rfloor$$
$$d_i = \max(1, b_{i+1} - b_i) \quad \text{for non-silence phonemes}$$

Due to the telescoping property:
$$\sum_{i=1}^N (b_{i+1} - b_i) = b_N - b_0 = \left\lfloor \frac{(t_{\text{end}} - t_{\text{start}}) \times f_s}{H} + 0.5 \right\rfloor$$
Cumulative rounding drift is **strictly zero**.

### Mel-Spectrogram Reconciliation Strategy
When an 80-channel mel-spectrogram is generated from the trimmed audio waveform using STFT with center reflect-padding, the resulting frame count $T_{\text{mel}}$ can differ from the boundary-rounded duration sum by $\Delta \in \{0, +1\}$ frames due to integer sample truncation ($N_{\text{samples}} = \lfloor f_s \cdot t_{\text{end}} \rfloor - \lfloor f_s \cdot t_{\text{start}} \rfloor$).

To guarantee that FastSpeech 2's `LengthRegulator` output length matches the mel target tensor length without exception:
1. We compute the fractional duration residual for each interval:
   $$r_i = \frac{(e_i - s_i) \times f_s}{H} - d_i$$
2. If $T_{\text{mel}} > \sum d_i$ ($\Delta > 0$): We allocate $+1$ frame to intervals with the largest positive residual $r_i$ (preferring silence tokens `'sp'` or vowel nuclei).
3. If $T_{\text{mel}} < \sum d_i$ ($\Delta < 0$): We deduct $-1$ frame from intervals with the largest negative residual, while strictly enforcing $d_i \ge 1$ for all spoken phonemes.

### Verification Across Pilot Cohort
- **Total Utterances Tested:** 100 / 100
- **Exact Frame Match Rate:** **100% (100 / 100)**
- **Reconciliation Applied:** 48 utterances required a +1 frame adjustment; 52 utterances were already exact ($\Delta = 0$).
- **Discrepancy Range:** $\min = 0, \max = 1$.
- **Zero-Duration Spoken Phones:** 0 (none clamped or dropped).

---

## 7. FastSpeech 2 Compatibility Verification

| Requirement | Audit Finding | Status |
| :--- | :--- | :--- |
| **Phoneme Vocabulary** | 63 unique phoneme labels observed across 4,918 tokens. Every single token maps to `original_fastspeech2/text/symbols.py` (e.g. `@HH`, `@AA1`, `@sp`). | **PASS (100%)** |
| **Frontend Serialization** | Text sequences formatted as `"{HH W AH1 ...}"` parse cleanly through `text_to_sequence(..., ["english_cleaners"])`. | **PASS (100%)** |
| **Duration Targets** | All durations are positive integers ($d_i \in \mathbb{Z}^+$), range: 1 to 170 frames. | **PASS (100%)** |
| **Mel Frame Alignment** | $\sum_{i=1}^N d_i == T_{\text{mel}}$ for all 100 utterances. | **PASS (100%)** |
| **Speaker Conditioning** | All 100 utterances map to valid speaker profile IDs ($0 \le \text{profile\_id} < 115$), contiguous in `speakers.json`. | **PASS (100%)** |
| **Dataset Format** | Metadata lines adhere to `id|speaker|{phones}|raw_text` format compatible with `common/` and `original_fastspeech2/dataset.py`. | **PASS (100%)** |

### Sample Dataset Manifest Lines
```text
281474976883831_f1642_chunk_0|spk_102|{W AH1 T S M AY1 F IH1 N K EH1 R IH1 Z S M AA1 L F AY1 N AE2 N S B AE1 NG K AH0 K AW1 N T B AE1 L AH0 N S}|What's my Fin care is Small Finance Bank account balance?
281474976883931_f769_chunk_0|spk_013|{K AE1 N Y UW1 CH EH1 K sp IH1 F DH AH0 B AE0 N AA1 R AH0 Z HH IH1 N D UW2 Y UW2 N AH0 V ER1 S AH0 T IY0 B IH1 L IH0 NG IH1 Z D AH1 N}|Can you check if the Banaras Hindu University billing is done?
281474976883943_f2231_chunk_15|spk_013|{D IH0 T ER1 M AH0 N DH AH0 AE1 K SH AH0 N Z DH AE1 T W IH1 L HH AE1 V DH AH0 G R EY1 T AH0 S T IH1 M P AE2 K T}|Determine the actions that will have the greatest impact.
```

---

## 8. Regression Suite & Unit Test Verification

A dedicated unit test suite was added in `tests/test_duration_extraction.py`:
- `TestIntervalValidation`: Non-empty checks, negative start/duration rejection, overlap detection, zero-length handling.
- `TestGapFilling`: Gap detection, internal pause insertion, leading gap padding.
- `TestSilenceTrimming`: Leading/trailing silence trimming, internal pause retention, all-silence rejection.
- `TestDurationCalculation`: Telescoping property validation, short phone minimum frame enforcement.
- `TestMelReconciliation`: Exact match, $+1$ frame reconciliation, $-1$ frame deduction.
- `TestFastSpeech2Symbols`: Vocabulary membership and mapping consistency.
- `TestRealPilotTextGrids`: End-to-end extraction across all 100 generated pilot TextGrids.

### Regression Suite Execution
```text
Ran 165 tests in 7.079s
OK
```
All 165 tests pass (145 existing tests + 20 new duration extraction tests). No regressions introduced.

---

## 9. Remaining Blockers & Risks

1. **Full Dataset Alignment Scale:**
   - The full training set comprises 3,215 utterances (~5.0 hours).
   - At 0.59s per utterance on 4 CPU workers, full training set alignment will take approximately **30 to 35 minutes**.
   - This is completely feasible to execute locally without requiring GPU compute or external cloud services.
2. **Full Dataset OOV Expansion:**
   - In 100 utterances, 48 OOVs were discovered. Across 3,215 utterances, we anticipate approximately 400 to 600 unique OOV words (predominantly Indian proper nouns).
   - G2P batch processing takes ~3 seconds per 100 words. Thus, full OOV resolution will require less than 30 seconds.
3. **World Vocoder Feature Extraction Dependencies:**
   - In FastSpeech 2 preprocessing, `pyworld` is required for $F_0$ extraction.
   - We must verify whether `pyworld` and pitch/energy extraction are installed and operate properly in `.venv` before launching Stage 3 (full preprocessing).

---

## 10. Recommendations for Full Preprocessing (Stage 3)

1. **Automated Lexicon Augmentation:**
   Integrate the automated `mfa find_oovs` $\to$ `mfa g2p` $\to$ `dictionary merge` pipeline into the full training dataset preprocessing workflow.
2. **Execute Full Alignment Locally:**
   Execute `mfa align` with `--num_jobs 6` on Apple Silicon. The 3,215 training utterances will complete in ~25-30 minutes.
3. **Maintain Split Integrity:**
   Align the training set (`English/metadata/train.txt`) and validation set (`English/metadata/val.txt`) independently. Do not include validation utterances when generating the G2P auxiliary lexicon or computing pitch/energy normalizers.
4. **Proceed to Stage 2 Approval:**
   With 100/100 pilot alignments validated, duration extraction verified, and 165/165 regression tests passing, the pipeline is fully ready for Step 1.4J-D (Full English Alignment & Feature Preprocessing).

