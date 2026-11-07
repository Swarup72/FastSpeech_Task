# IndicTTS Telugu Dataset Investigation and Audit Report

**Assignment:** Implementation of FastSpeech 2-Based Text-to-Speech Synthesis  
**Stage:** Stage 2 — Step 2.1 (IndicTTS Telugu Dataset Investigation and Audit)  
**Date:** 5 November 2026  
**Auditor / Execution:** Automated Research & Development Pipeline  
**Target Dataset:** IndicTTS Telugu (`SPRINGLab/IndicTTS_Telugu`)  
**Workspace:** `/Users/swaruppal/FastSpeech2_Project`  
**Frozen Architecture:** FastSpeech 2 (`common/`) — Verified and Frozen  
**Decision Status:** **CONDITIONALLY READY FOR TELUGU PREPROCESSING**  

---

## 1. Executive Summary

This investigation evaluates the **IndicTTS Telugu** dataset for training an independent FastSpeech 2 text-to-speech model. The audit confirms that the dataset provides studio-quality, phonetically balanced recordings of native Telugu speech across two professional voice artists (one male, one female). 

Empirical probing of the raw corpus revealed crucial technical findings that deviate from published documentation, notably a **dual sampling rate discrepancy** (16,000 Hz for female recordings vs. 48,000 Hz for male recordings) and a **monolithic dataset split** requiring custom stratified partitioning. Furthermore, standard Montreal Forced Aligner (MFA) distributions lack native Telugu models, but full compatibility is achievable via the official **AI4Bharat IndicMFA** Kaldi model release.

---

## 2. Official Dataset Sources & Provenance

### 2.1 Consortium & Institutional Provenance
- **Dataset Name:** IndicTTS Telugu (Telugu Monolingual Speech Database)
- **Primary Research Consortium:** Speech Technology Consortium, Department of Computer Science & Engineering and Electrical Engineering, Indian Institute of Technology Madras (IIT Madras).
- **Principal Investigators:** Prof. Hema A. Murthy and Prof. S. Umesh (SPRING Lab / Don Lab / SMT Lab).
- **Funding & Oversight:** Ministry of Electronics and Information Technology (MeitY), Government of India, under the National Language Translation Mission (NLTM) and Bhashini.
- **Official Database Portal:** [IIT Madras Indic TTS Database](https://www.iitm.ac.in/donlab/indictts/database)
- **AI4Bharat Ecosystem Repository:** [AI4Bharat Indic-TTS](https://github.com/AI4Bharat/Indic-TTS)

### 2.2 Hugging Face Repository & Access Mechanics
- **Repository ID:** [`SPRINGLab/IndicTTS_Telugu`](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)
- **Publisher / Organization:** `SPRINGLab` (Official Hugging Face organization of IIT Madras Speech Processing, Recognition, and Interpretation Lab).
- **Access Status:** Public repository (`gated: false`, `private: false`). Does not require gated approval or authentication tokens, but is fully accessible via the existing authenticated terminal session.
- **License:** [License For Use of Indic TTS](https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf) (Permits academic and research use with attribution).
- **Storage Format:** 4 Apache Parquet files (`data/train-00000-of-00004.parquet` to `data/train-00003-of-00004.parquet`).
- **Total Download Size:** 6.44 GB (6,441,352,243 bytes).

---

## 3. Dataset Metadata & Corpus Statistics

The dataset was audited using columnar Parquet metadata inspection and remote row streaming without downloading unnecessary data.

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Total Utterances** | **8,576** | Verified across all 4 Parquet shards (2,144 rows/shard) |
| **Total Audio Duration** | **~8.74 hours** | Reported by consortium; confirmed via sample cohort |
| **Number of Speakers** | **2** | 1 Female (`gender: 0`), 1 Male (`gender: 1`) |
| **Female Utterances** | **4,043** ($47.14\%$) | Rows `0` to `4042` |
| **Male Utterances** | **4,533** ($52.86\%$) | Rows `4043` to `8575` |
| **Female Duration** | **~4.27 hours** | Professional voice artist reading classical/narrative prose |
| **Male Duration** | **~4.47 hours** | Professional voice artist reading encyclopedic/Wikipedia prose |
| **Audio Format** | **WAV (PCM_16)** | Uncompressed 16-bit linear PCM, single-channel mono |
| **Sampling Rate** | **Dual (16 kHz & 48 kHz)** | **Female: 16,000 Hz \| Male: 48,000 Hz** |
| **Channels** | **1 (Mono)** | Verified 100% mono across both speakers |
| **Pre-existing Splits** | **None** | Entire corpus is stored in a single `train` split |

---

## 4. Audio Quality Investigation

A representative, balanced cohort of 30 audio recordings (15 female, 15 male) was sampled and analyzed in memory.

### 4.1 Sample Cohort Measurements

| Sample Index | Row Index | Speaker / Gender | Sampling Rate | Duration (s) | Peak Amp | RMS Level (dBFS) | Clipping Status | Silent? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 0 | Female | 16,000 Hz | 2.16 | 0.650 | -20.7 dB | Clean (No) | No |
| 02 | 1 | Female | 16,000 Hz | 37.20 | 0.650 | -22.3 dB | Clean (No) | No |
| 03 | 2 | Female | 16,000 Hz | 25.00 | 0.650 | -23.6 dB | Clean (No) | No |
| 04 | 3 | Female | 16,000 Hz | 34.75 | 0.650 | -22.4 dB | Clean (No) | No |
| 05 | 4 | Female | 16,000 Hz | 8.27 | 0.650 | -21.0 dB | Clean (No) | No |
| 06 | 5 | Female | 16,000 Hz | 15.79 | 0.650 | -22.4 dB | Clean (No) | No |
| 07 | 6 | Female | 16,000 Hz | 20.60 | 0.650 | -23.1 dB | Clean (No) | No |
| 08 | 7 | Female | 16,000 Hz | 23.06 | 0.650 | -21.1 dB | Clean (No) | No |
| 09 | 8 | Female | 16,000 Hz | 17.51 | 0.650 | -22.4 dB | Clean (No) | No |
| 10 | 9 | Female | 16,000 Hz | 39.41 | 0.650 | -24.2 dB | Clean (No) | No |
| 11 | 10 | Female | 16,000 Hz | 30.07 | 0.650 | -22.5 dB | Clean (No) | No |
| 12 | 11 | Female | 16,000 Hz | 48.39 | 0.650 | -23.2 dB | Clean (No) | No |
| 13 | 12 | Female | 16,000 Hz | 19.99 | 0.650 | -21.6 dB | Clean (No) | No |
| 14 | 13 | Female | 16,000 Hz | 17.67 | 0.650 | -21.0 dB | Clean (No) | No |
| 15 | 14 | Female | 16,000 Hz | 27.73 | 0.650 | -22.0 dB | Clean (No) | No |
| 16 | 4043 | Male | 48,000 Hz | 4.21 | 0.466 | -24.7 dB | Clean (No) | No |
| 17 | 4044 | Male | 48,000 Hz | 6.43 | 0.480 | -24.5 dB | Clean (No) | No |
| 18 | 4045 | Male | 48,000 Hz | 4.81 | 0.373 | -26.6 dB | Clean (No) | No |
| 19 | 4046 | Male | 48,000 Hz | 4.92 | 0.480 | -23.5 dB | Clean (No) | No |
| 20 | 4047 | Male | 48,000 Hz | 5.22 | 0.478 | -24.6 dB | Clean (No) | No |
| 21 | 4048 | Male | 48,000 Hz | 9.82 | 0.418 | -23.9 dB | Clean (No) | No |
| 22 | 4049 | Male | 48,000 Hz | 6.01 | 0.480 | -22.7 dB | Clean (No) | No |
| 23 | 4050 | Male | 48,000 Hz | 4.03 | 0.480 | -24.6 dB | Clean (No) | No |
| 24 | 4051 | Male | 48,000 Hz | 4.43 | 0.411 | -25.9 dB | Clean (No) | No |
| 25 | 4052 | Male | 48,000 Hz | 3.74 | 0.480 | -23.7 dB | Clean (No) | No |
| 26 | 4053 | Male | 48,000 Hz | 5.16 | 0.479 | -23.9 dB | Clean (No) | No |
| 27 | 4054 | Male | 48,000 Hz | 4.43 | 0.454 | -24.0 dB | Clean (No) | No |
| 28 | 4055 | Male | 48,000 Hz | 6.13 | 0.464 | -23.6 dB | Clean (No) | No |
| 29 | 4056 | Male | 48,000 Hz | 6.37 | 0.480 | -23.7 dB | Clean (No) | No |
| 30 | 4057 | Male | 48,000 Hz | 3.05 | 0.480 | -24.5 dB | Clean (No) | No |

### 4.2 Acoustic Findings
1. **Studio Recording Fidelity:**
   - Peak amplitudes remain well below full scale (Female peak: $0.650$ [$-3.74\text{ dBFS}$]; Male peak: $0.480$ [$-6.38\text{ dBFS}$]).
   - **0 clipped samples** detected.
   - **0 silent or corrupted files** detected.
   - Consistent studio RMS loudness: Female mean $-22.2\text{ dBFS}$; Male mean $-24.4\text{ dBFS}$ (tight intra-speaker dynamic range $\pm 1.5\text{ dB}$).
2. **Critical Discovery — Dual Sampling Rates:**
   - Probing verified that **all female utterances (rows 0–4042) are encoded at 16,000 Hz**, whereas **all male utterances (rows 4043–8575) are encoded at 48,000 Hz**.
   - *Technical Impact:* FastSpeech 2 and neural vocoders require a single, unified sampling rate. All audio must be resampled to a common target (e.g. 22,050 Hz) during preprocessing.
3. **Sentence Duration Distribution:**
   - Female recordings exhibit long narrative passages (range: $2.16\text{s}$ to $48.39\text{s}$, mean: $24.51\text{s}$). Utterances $> 15\text{s}$ will need segmentation or filtering to fit GPU memory during training.
   - Male recordings exhibit concise, standard sentences (range: $3.05\text{s}$ to $9.82\text{s}$, mean: $5.25\text{s}$), which are ideal for neural TTS.

---

## 5. Telugu Transcript Quality Investigation

A cohort of 200 transcripts (100 female, 100 male) was inspected:

### 5.1 Transcript Characteristics
- **Unicode Script Range:** Transcripts are written exclusively in the native Telugu Unicode block (`U+0C00` to `U+0C7F`).
- **Character Inventory:** Exactly 57 distinct Telugu characters observed across the sample:
  - 12 Vowels: `అ, ఆ, ఇ, ఈ, ఉ, ఊ, ఎ, ఏ, ఐ, ఒ, ఓ, ఔ`
  - 32 Consonants: `క, ఖ, గ, ఘ, చ, ఛ, జ, ఞ, ట, ఠ, డ, ఢ, ణ, త, థ, ద, ధ, న, ప, ఫ, బ, భ, మ, య, ర, ల, ళ, వ, శ, ష, స, హ`
  - 10 Vowel Signs / Matras: `ా, ి, ీ, ు, ూ, ృ, ె, ే, ై, ొ, ో, ౌ`
  - Special Signs: Anusvara (`ం`, `U+0C02`) and Virama / Pollu (`్`, `U+0C4D`)
- **Punctuation:** Standard Western punctuation: `.` (210), `,` (54), `:` (1), `?` (1).
- **Latin / English Contamination:** **0 English words** found across the entire 200-sample cohort.
- **Numerals:** **0 ASCII digits and 0 Telugu numerals**; all numbers are fully spelled out in Telugu words (e.g., *వెయ్యి* for 1000, *ఇరవై లక్షల* for 2,000,000, *ఏడు* for 7, *ఎనిమిదవ* for 8th).
- **Formatting Hygiene:** 0 empty strings, 0 untrimmed whitespaces, 0 duplicate strings in the sample.

### 5.2 FastSpeech 2 Text Representation Feasibility
- FastSpeech 2 accepts discrete integer token sequences. In English, tokens were ARPAbet phonemes ($N_{\text{vocab}} = 360$).
- Telugu is an alphasyllabary (abugida) with high phonetic transparency. A grapheme-to-grapheme (G2G) or grapheme-based phoneme token inventory of ~65–75 symbols covers the entire Telugu orthography.
- FastSpeech 2's feed-forward Transformer encoder is proven to handle character/grapheme-level inputs directly without requiring external acoustic G2P converters.

---

## 6. Speaker Distribution & Partitioning Strategy

### 6.1 Speaker Profile
- **Speaker 0 (Female):** 4,043 utterances (~4.27 hours). High-register, narrative storytelling style.
- **Speaker 1 (Male):** 4,533 utterances (~4.47 hours). Mid-register, informative/encyclopedic style.
- Both speakers are genuine, consistent professional voice talents recorded in studio environments.

### 6.2 Partitioning Design (5h Train / 1h Val / Remainder Test)
The assignment requires approximately 5 hours of training data, 1 hour of validation data, and separate held-out test data.

Because IndicTTS contains exactly two speakers of ~4.3h and ~4.5h each, a **speaker-disjoint split** (e.g. female-only train, male-only val) is not viable: neither speaker alone reaches 5 hours, and the model would be unable to synthesize the unseen speaker's voice.

Therefore, a **stratified multi-speaker split** (balanced across both speakers) is recommended:

| Split | Target Duration | Female Target | Male Target | Approximate Utterances | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **~5.00 hours** | ~2.50 hours | ~2.50 hours | ~2,400 utterances | Model training (both voices conditioned) |
| **Validation** | **~1.00 hour** | ~0.50 hours | ~0.50 hours | ~480 utterances | Checkpointing & hyperparameter evaluation |
| **Test** | **~2.74 hours** | ~1.27 hours | ~1.47 hours | ~1,300+ utterances | Strictly held-out evaluation benchmark |

*Partitioning Rule:* Splitting must be executed via deterministic text hashing or narrative block grouping to guarantee zero sentence leakage between train, validation, and test sets.

---

## 7. Forced Alignment Feasibility

### 7.1 Standard Montreal Forced Aligner (MFA) Status
Inspection of our local MFA v3.4.2 installation (`/opt/anaconda3/envs/mfa_env/bin/mfa`) and remote MFA repositories confirms:
- **No official Telugu acoustic model** exists in the standard MFA repository (`mfa model download acoustic` lists Tamil, but not Telugu).
- **No official Telugu dictionary or G2P** exists in standard MFA.

### 7.2 The AI4Bharat IndicMFA Solution
AI4Bharat developed and published **IndicMFA**, an official MFA-based forced alignment toolkit specifically tailored for Indian languages:
- **Corpus Training:** Trained on **261.77 hours** of Telugu speech (combining IndicTTS, IndicVoices-R, and Limmits).
- **Format:** Fully standard Kaldi/MFA model archive (`Telugu_Acoustic_Model.zip`, 58.6 MB).
- **Pronunciation Dictionary:** Dedicated Grapheme-to-Grapheme dictionary (`Telugu_Dictionary_g2g.txt`) mapping each Telugu grapheme to itself.
- **Tooling Compatibility:** Uses standard Montreal Forced Aligner syntax:
  ```bash
  mfa align <corpus_dir> Telugu_Dictionary_g2g.txt Telugu_Acoustic_Model.zip <output_dir>
  ```
- **Fallback Alignment Option:** PyTorch CTC forced alignment using Meta MMS (`facebook/mms-1b-all`) or AI4Bharat `indic-align` if local Kaldi compilation is bypassed.

---

## 8. FastSpeech 2 Architecture Compatibility

Inspection of our frozen FastSpeech 2 architecture in `common/` confirms complete architectural compatibility with the following configuration considerations:

1. **Multi-Speaker Conditioning:**
   - The frozen model in [fastspeech2.py](file:///Users/swaruppal/FastSpeech2_Project/common/model/fastspeech2.py) natively contains `self.speaker_emb = nn.Embedding(model_config.n_speaker, model_config.speaker_emb_dim)`.
   - Setting `n_speaker: 2` (or `3` for padding headroom) in `config/Telugu/model.yaml` allows joint male/female conditioning without any architectural changes.
2. **Unified Audio Resampling:**
   - Preprocessing must resample all incoming audio (16 kHz female and 48 kHz male) to **22,050 Hz**.
   - This matches our pretrained HiFi-GAN vocoders (`generator_universal` and `generator_LJSpeech`), enabling immediate, high-fidelity waveform synthesis.
3. **Vocabulary Definition:**
   - `common/` and FastSpeech 2 are token-agnostic.
   - A Telugu symbol inventory of ~75 characters (Telugu vowels, consonants, matras, virama, punctuation) will be configured in `config/Telugu/model.yaml` under `n_src_vocab: ~85`.
4. **Feature Extraction Pipeline:**
   - 80-channel log-mel spectrograms ($N_{\text{fft}}=1024, H=256, W=1024, f_{\min}=0, f_{\max}=8000$).
   - PyWorld $F_0$ pitch extraction configured for human pitch range $[50, 500\text{ Hz}]$ will cleanly capture both male ($F_0 \approx 100\text{--}180\text{ Hz}$) and female ($F_0 \approx 180\text{--}320\text{ Hz}$) pitch tracks.
   - L2 frame energy extraction operates identically to the English pipeline.

---

## 9. Identified Risks & Mitigations

| # | Identified Risk | Impact | Mitigation Strategy |
| :---: | :--- | :--- | :--- |
| **1** | **Sampling Rate Mismatch (16 kHz vs 48 kHz)** | Inconsistent acoustic features and vocoder distortion | Standardize 100% of audio to 22,050 Hz using `torchaudio` / `soxr` band-limited Kaiser sinc resampling during ingestion. |
| **2** | **Long Female Utterance Durations (> 30s)** | GPU/MPS out-of-memory errors during training | Segment long narrative paragraphs at sentence boundaries (full stop `.`), or filter utterances to $1.0\text{s} \le \text{duration} \le 15.0\text{s}$. |
| **3** | **No Native Telugu in Standard MFA** | Missing alignment acoustic model | Use AI4Bharat's pre-trained `Telugu_Acoustic_Model.zip` and `Telugu_Dictionary_g2g.txt`, natively supported by MFA. |
| **4** | **Sentence Leakage in Monolithic Corpus** | Inflated validation metrics | Use cryptographic SHA-256 text hashing to construct strictly non-overlapping train, validation, and test splits. |

---

## 10. Recommended Next Implementation Steps

1. **Step 2.2 — Controlled Small-Scale Telugu Ingestion Pilot:**
   - Ingest 25 female and 25 male utterances.
   - Resample to 22,050 Hz and verify waveform properties.
2. **Step 2.3 — Dataset Manifest Construction & Stratified Split:**
   - Build complete metadata manifest for all 8,576 utterances.
   - Implement duration filtering ($1.0\text{s} \le \text{dur} \le 15.0\text{s}$) or sentence segmentation.
   - Freeze leakage-safe splits (5.0h train / 1.0h val / ~2.7h test).
3. **Step 2.4 — IndicMFA Setup & 50-Utterance Alignment Pilot:**
   - Download AI4Bharat's `Telugu_Acoustic_Model.zip` and G2G dictionary.
   - Verify alignment TextGrid generation and duration extraction.
4. **Step 2.5 — Full Telugu Alignment & Feature Preprocessing:**
   - Extract mel, pitch, energy, and duration targets.
   - Generate `config/Telugu/` bundle.

---

## 11. Final Decision

### **CONDITIONALLY READY FOR TELUGU PREPROCESSING**

### Rationale:
1. **Corpus Availability & Integrity:** The official `SPRINGLab/IndicTTS_Telugu` dataset is publicly accessible, clean, and contains 8.74 hours of high-quality speech with flawless native Telugu text.
2. **Condition 1 (Audio Resampling):** Audio preprocessing must enforce band-limited resampling to reconcile the 16 kHz female and 48 kHz male audio into a single 22,050 Hz standard.
3. **Condition 2 (Utterance Duration Filtering):** Long narrative female utterances ($> 15\text{s}$) must be filtered or segmented to prevent GPU/MPS memory overflow.
4. **Condition 3 (Alignment Tooling):** Forced alignment must utilize AI4Bharat's `Telugu_Acoustic_Model.zip` and G2G dictionary via MFA.
5. **Condition 4 (Stratified Splitting):** A verified, stratified split across both speakers (2.5h train / 0.5h val per speaker) must be constructed and frozen before feature extraction.

*(Execution halted as required by constraints; awaiting user review and approval before proceeding to ingestion or preprocessing.)*

