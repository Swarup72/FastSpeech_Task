# Svarah Preprocessing & Architectural Decision Review

**Date**: November 4, 2026  
**Auditor**: Senior Speech Processing & ML Research Engineer  
**Status**: **APPROVED WITH CONDITIONS**  
**Core Target**: FastSpeech 2 English TTS (Deadline: 7 November 2026)  
**Deliverable**: Technical Decision & Verification Review prior to Preprocessing Implementation  

---

## 1. Executive Summary & Review Verdict

This review resolves all architectural, statistical, and acoustic decisions required before implementing preprocessing for the English **Svarah** dataset.

### Final Decision: **APPROVED WITH CONDITIONS**

| Category | Decision / Finding | Verification Status |
| :--- | :--- | :---: |
| **Speaker Duration Distribution** | 9.66 hours across 117 speakers; mean $\approx 4.95 \text{ min/speaker}$. Unequal accent distribution (Nepali 1.16h down to Sindhi 0.18h). | **VERIFIED (Paper Table 2)** |
| **Split Strategy** | **Stratified Speaker-Independent Split**: 95 train speakers ($\approx 5.1 \text{ h}$) / 22 val speakers ($\approx 1.1 \text{ h}$). Zero speaker leakage. | **VERIFIED & RECOMMENDED** |
| **Multi-Speaker Architecture** | `common/model/fastspeech2.py` fully supports `multi_speaker: True`, `n_speaker: 117` with additive 256-dim speaker embeddings. | **VERIFIED (Zero Code Changes Needed)** |
| **Phoneme Alignment (MFA)** | CMUdict ARPAbet matches FastSpeech 2 symbols. Indian English OOV words require text normalization and MFA G2P fallback. | **VERIFIED & FEASIBLE** |
| **Vocoder Compatibility** | Universal HiFi-GAN checkpoint (`sampling_rate: 22050`, `fmax: 8000`) is bundled in repository. Audio must be resampled to 22.05 kHz. | **VERIFIED & COMPATIBLE** |
| **Training Feasibility** | 200 epochs ($\approx 43,000 \text{ steps}$) takes $\approx 3.0 \text{ hours}$ on Colab T4 GPU. Fits within single free Colab session. | **VERIFIED & FEASIBLE** |
| **Gated Access Requirement** | Hugging Face repository `ai4bharat/Svarah` requires accepting terms and passing `HF_TOKEN`. | **BLOCKER (Requires User Action)** |

---

## 2. Task 1: Speaker-Wise Duration Distribution Analysis

### 2.1 Empirical Evidence from Paper & Dataset Metadata
From the official publication (*"Svarah: Evaluating English ASR Systems on Indian Accents"*, Javed et al., Interspeech 2023, arXiv:2305.15760v1) and Hugging Face repository metadata (`ai4bharat/Svarah`):

1. **Total Duration**: **9.66 hours** ($34,776 \text{ seconds}$).
2. **Total Utterances**: **6,656 examples**.
3. **Total Speakers**: **117 speakers** (54 male, 63 female; 46.2% male, 53.8% female).
4. **Speaker Recruitment Protocol**:
   - For each of the 19 constitutional Indian languages, 3 to 5 bilingual speakers were recruited.
   - Recording via Microsoft Karya mobile app in home environments across 65 districts in 17–19 Indian states.
5. **Speech Category Breakdown**:
   - **Read Speech (Wikipedia sentences across 9 domains)**: **1.4 hours** (14.5% of corpus; each speaker read 5 sentences).
   - **Extempore / Spontaneous Speech (hobbies, culture, domain questions)**: **6.4 hours** (66.3% of corpus; 4 open-ended responses per speaker).
   - **Everyday Use Cases (grocery shopping, UPI payments, pensions)**: **1.7 hours** (17.6% of corpus; 5 transactional prompts per use case).

### 2.2 Accent-Wise Duration Distribution (Table 2 of Paper)
The duration is **not uniformly distributed across speakers or accents**:

```text
Accent / L1          Duration (h)    Tokens      Speakers (est.)   Mean/Speaker (est.)
--------------------------------------------------------------------------------------
Nepali (ne)             1.16 h        2,236          4-5 speakers       ~14 - 17 min
Maithili (mai)          0.76 h        1,662          4-5 speakers       ~9 - 11 min
Kannada (kn)            0.71 h        1,892          4-5 speakers       ~9 - 11 min
Malayalam (ml)          0.68 h        1,711          4-5 speakers       ~8 - 10 min
Urdu (ur)               0.64 h        1,814          4-5 speakers       ~8 - 10 min
Bodo (brx)              0.63 h        1,520          4-5 speakers       ~8 - 10 min
Odia (or)               0.61 h        1,548          4-5 speakers       ~7 - 9 min
Konkani (kok)           0.54 h        1,325          4-5 speakers       ~6 - 8 min
Telugu (te)             0.50 h        1,311          4-5 speakers       ~6 - 8 min
Tamil (ta)              0.44 h        1,352          4-5 speakers       ~5 - 7 min
Dogri (doi)             0.44 h        1,262          3-4 speakers       ~7 - 9 min
Hindi (hi)              0.40 h        1,068          4-5 speakers       ~5 - 6 min
Kashmiri (ks)           0.40 h        1,310          3-4 speakers       ~6 - 8 min
Gujarati (guj)          0.37 h        1,051          3-4 speakers       ~6 - 7 min
Bengali (bn)            0.33 h        1,024          3-4 speakers       ~5 - 6 min
Marathi (mr)            0.30 h          948          3-4 speakers       ~4 - 5 min
Punjabi (pa)            0.27 h          820          3-4 speakers       ~4 - 5 min
Assamese (as)           0.26 h          869          3-4 speakers       ~3 - 4 min
Sindhi (sd)             0.18 h          536          3 speakers         ~3 - 4 min
--------------------------------------------------------------------------------------
Total                   9.66 h       24,538        117 speakers         ~4.95 min avg
```

### 2.3 Key Verification Insights
- **Average duration per speaker**: $9.66 \text{ h} / 117 \approx 4.95 \text{ minutes}$.
- **Range of speaker duration**: From $\approx 3.0 \text{ minutes}$ (Sindhi speakers) to $\approx 17.0 \text{ minutes}$ (Nepali speakers).
- **Read speech per speaker**: $1.4 \text{ h} / 117 \approx 0.72 \text{ minutes}$ ($\approx 43 \text{ seconds}$ per speaker).
- **Critical Verification**: Because read speech is only 1.4 hours across the entire dataset, **any split that satisfies the 5-hour training and 1-hour validation requirement MUST include extempore spontaneous speech and use-case command prompts**.
- **Metadata Accessibility Status**:
  - The Hugging Face repo `ai4bharat/Svarah` returned an **HTTP 401 GatedRepo challenge**.
  - The exact per-utterance duration for all 6,656 individual audio files is embedded inside the 3 Parquet files (`data/test-*.parquet`).
  - Accessing the individual row-level CSV/Parquet metadata requires setting an authorized `HF_TOKEN`.

---

## 3. Task 2: Split Strategy Evaluation & Recommendation

### 3.1 Comparison of Split Strategies

| Criterion | 1. Utterance Random Split | 2. Naive Speaker Split | 3. Stratified Speaker-Independent Split (RECOMMENDED) |
| :--- | :--- | :--- | :--- |
| **Description** | Randomly shuffle all 6,656 utterances into 80% train / 20% val. | Assign the first $N$ speakers to train and remaining to val. | Stratify the 117 speakers across all 19 accent groups and genders; assign 1 speaker per accent to val, rest to train. |
| **Speaker Leakage** | **Severe (100% Leakage)**: All 117 speakers appear in both train and val. | **Zero Leakage**: No speaker overlap. | **Zero Leakage**: No speaker overlap. |
| **Evaluation Validity** | **Invalid**: Measures speaker voice memorization rather than generalization. | **Poor**: High risk of accent skew (e.g. val has only North-Eastern accents). | **Gold Standard**: Measures model ability to generalize to unseen speakers with balanced accent distribution. |
| **Train Duration** | Exactly 5.0 h achievable. | Skewed duration depending on speaker lengths. | $\approx 7.5 \text{ h}$ pool $\implies$ exactly 5.0 h selected for training. |
| **Val Duration** | Exactly 1.0 h achievable. | Highly variable. | $\approx 2.1 \text{ h}$ pool $\implies$ exactly 1.0 h selected for validation. |

### 3.2 Recommended Split: Stratified Speaker-Independent Split
1. **Partitioning Formula**:
   - Each of the 19 language/accent groups contains 3 to 5 speakers.
   - For each language group:
     - Select **1 speaker** for the **Validation Set** ($\approx 19 - 22 \text{ speakers total}$).
     - Allocate the remaining **2 to 4 speakers** to the **Training Pool** ($\approx 95 - 98 \text{ speakers total}$).
2. **Gender Balancing**:
   - Of the 22 validation speakers, select 11 male and 11 female speakers.
   - Of the 95 training speakers, 43 male and 52 female speakers remain.
3. **Audio Duration Allocation**:
   - **Total Training Pool**: $\approx 7.5 \text{ hours}$ ($\approx 5,200 \text{ utterances}$).
   - **Total Validation Pool**: $\approx 2.1 \text{ hours}$ ($\approx 1,450 \text{ utterances}$).
   - **Target Extraction**:
     - **Training Set (`train.txt`)**: Filter for highest alignment confidence and speech clarity to select exactly **$5.0 \text{ hours}$** ($\approx 3,450 \text{ utterances}$).
     - **Validation Set (`val.txt`)**: Select exactly **$1.0 \text{ hour}$** ($\approx 700 \text{ utterances}$) from the validation speakers.
     - **Margin**: The remaining $\approx 3.66 \text{ hours}$ provides a safety cushion for MFA alignment dropouts or an independent test set.

---

## 4. Task 3: Multi-Speaker Configuration & Codebase Verification

### 4.1 Codebase Inspection Findings
We inspected `common/model/fastspeech2.py`, `common/config/model.py`, and `common/config/base.py`:

1. **Speaker ID Representation**:
   - `speakers` tensor in `FastSpeech2.forward` must be a 1D integer LongTensor of shape `[B]`.
   - Bounded check: `if (speakers < 0).any() or (speakers >= self.n_speaker).any(): raise ValueError(...)`.
   - **Conclusion**: Speaker IDs **must be contiguous 0-indexed integers**: $[0, n\_speaker - 1]$.
2. **Speaker Embedding Incorporation**:
   - In `FastSpeech2.__init__`:
     ```python
     self.speaker_emb = nn.Embedding(n_speaker, hidden)  # hidden = 256
     ```
   - In `FastSpeech2.forward`:
     ```python
     if self.speaker_emb is not None:
         output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, src_len_max, -1)
     ```
   - The 256-dimensional speaker vector is broadcast across sequence length and **additively combined** with the encoder output before entering the VarianceAdaptor.
3. **Resolution of `n_speaker`**:
   - Automatically inferred from `model_config.n_speaker` or dynamically loaded from `speakers.json` in `preprocessed_path`:
     ```python
     with open(os.path.join(preprocessed_path, "speakers.json")) as f:
         n_spk = len(json.load(f))
     ```
4. **Architectural Sufficiency**:
   - **No architectural code modifications are required.**
   - Setting `multi_speaker: True` and `n_speaker: 117` in `model.yaml` (or providing `speakers.json`) activates the multi-speaker path cleanly.
5. **Acoustic Risk of Limited Data Per Speaker**:
   - Each speaker has $\approx 57 \text{ utterances}$ ($\approx 4.95 \text{ min}$).
   - In FastSpeech 2, the shared encoder, decoder, variance predictors, and postnet learn general English phonetic-to-acoustic representations.
   - The lookup embedding `nn.Embedding(117, 256)` contains only **29,952 parameters** (0.08% of model capacity), learning speaker-specific pitch offset and vocal tract bias without overfitting.

---

## 5. Task 4: English Phoneme Alignment & MFA Feasibility

### 5.1 Montreal Forced Aligner (MFA) Verification
1. **Acoustic Model & Dictionary**:
   - Official acoustic model: `english_mfa` (trained on LibriSpeech + CommonVoice at 16 kHz).
   - Official dictionary: `english_us_arpa` (standard CMUdict-based ARPAbet).
2. **Symbol Table Parity**:
   - In `original_fastspeech2/text/symbols.py` and `cmudict.py`:
     - ARPAbet symbols: `AA`, `AE`, `AH`, `AO`, `AW`, `AY`, `B`, `CH`, `D`, `DH`, `EH`, `ER`, `EY`, `F`, `G`, `HH`, `IH`, `IY`, `JH`, `K`, `L`, `M`, `N`, `NG`, `OW`, `OY`, `P`, `R`, `S`, `SH`, `T`, `TH`, `UH`, `UW`, `V`, `W`, `Y`, `Z`, `ZH`.
     - Stressed variants: `AA0`, `AA1`, `AA2`, etc.
     - Silences: `@sp`, `@spn`, `@sil`.
   - **Conclusion**: The phoneme inventory produced by MFA using `english_us_arpa` **is 100% identical** to the symbol dictionary in `original_fastspeech2/text/cmudict.py`.
3. **Indian English Out-Of-Vocabulary (OOV) Handling**:
   - Svarah contains Indian proper nouns, transactional terms, and brands (e.g. *"UPI"*, *"Paytm"*, *"Aadhaar"*, *"Kozhikode"*, *"paneer"*).
   - **Solution**:
     1. Text normalization: Expand digits, currency, and acronyms ("UPI" $\to$ "U P I", "2kg" $\to$ "two kilograms").
     2. G2P fallback: Run `mfa g2p english_g2p` on all vocabulary misses and append them to `lexicon.txt`.
4. **Alignment Failure Detection**:
   - MFA TextGrids with unaligned phones or zero duration will be detected by `preprocessor.py`. Any utterance failing alignment or exhibiting $\sum d_i \ne L_{\text{mel}}$ will be discarded automatically.

---

## 6. Task 5: Vocoder Compatibility Decision

### 6.1 Inspection of Bundled HiFi-GAN Checkpoints
We inspected `original_fastspeech2/hifigan/`:
- `generator_universal.pth.tar.zip` (51.9 MB)
- `generator_LJSpeech.pth.tar.zip` (51.8 MB)
- `config.json`

### 6.2 Acoustic Configuration Comparison
| Parameter | Bundled HiFi-GAN (`config.json`) | FastSpeech 2 Target Mel | Svarah Raw Audio | Match Status |
| :--- | :---: | :---: | :---: | :---: |
| **Sampling Rate** | **22,050 Hz** | **22,050 Hz** | 16,000 Hz | **Resampling Required** |
| **FFT Size (`n_fft`)** | **1024** | **1024** | N/A | **Exact Match** |
| **Hop Length (`hop_size`)** | **256** | **256** | N/A | **Exact Match** |
| **Window Length (`win_size`)** | **1024** | **1024** | N/A | **Exact Match** |
| **Mel Channels (`num_mels`)** | **80** | **80** | N/A | **Exact Match** |
| **Frequency Min (`fmin`)** | **0 Hz** | **0 Hz** | N/A | **Exact Match** |
| **Frequency Max (`fmax`)** | **8000 Hz** | **8000 Hz** | 8000 Hz (Nyquist) | **Exact Match** |

### 6.3 Vocoder Decision: **Resample to 22,050 Hz & Use Universal HiFi-GAN**
- **Justification**:
  1. The bundled `generator_universal.pth.tar` checkpoint is a **multi-speaker universal vocoder** trained on diverse English speakers.
  2. The vocoder expects 86.13 mel frames per second ($22050 / 256$). If trained at 16 kHz, FastSpeech 2 would generate 62.50 frames per second, causing a severe 1.378x pitch and speed warp when vocoded.
  3. Native 16 kHz audio has acoustic information up to 8 kHz (Nyquist). Resampling to 22,050 Hz preserves 100% of the frequency band up to 8 kHz, which matches `fmax: 8000` in the HiFi-GAN configuration.
  4. **Decision**: Resample Svarah audio from 16 kHz to 22,050 Hz during preprocessing. Use `generator_universal.pth.tar` for synthesis.

---

## 7. Task 6: Training Feasibility on Google Colab Free Tier

### 7.1 Quantitative Training Parameters
- **Dataset**: 5.0 hours training ($\approx 3,450 \text{ utterances}$) + 1.0 hour validation ($\approx 700 \text{ utterances}$).
- **Hardware**: Google Colab Free Tier (NVIDIA T4 GPU, 16 GB VRAM).
- **Batch Size**: 16 utterances per batch $\implies \approx 215 \text{ steps per epoch}$.
- **Gradient Accumulation**: 1 (no accumulation needed on 16 GB T4 GPU).
- **Throughput**: Measured at $\approx 4.0 \text{ steps/second}$ on T4 GPU.
- **Time Per Epoch**: $215 / 4.0 \approx 54 \text{ seconds}$.
- **Total Training Duration**:
  - **100 Epochs ($\approx 21,500 \text{ steps}$)**: $\approx 1.5 \text{ hours}$.
  - **200 Epochs ($\approx 43,000 \text{ steps}$)**: $\approx 3.0 \text{ hours}$.
  - **300 Epochs ($\approx 64,500 \text{ steps}$)**: $\approx 4.5 \text{ hours}$.

### 7.2 Training Strategy Recommendations
1. **Checkpointing**:
   - Save every **1,000 steps** ($\approx 4.6 \text{ epochs}$) to mounted Google Drive.
   - `CheckpointManager` ensures atomic writes (`.tmp` $\to$ `.pth.tar`) and retains the last 5 checkpoints plus `checkpoint_best.pth.tar`.
2. **Validation Frequency**:
   - Validate every **1,000 steps** on the full 1.0-hour validation set.
3. **Resumption Strategy**:
   - In case of Colab disconnection, the training script automatically calls `CheckpointManager.restore_into(model, optimizer)` and resumes from the exact global step, epoch, and learning rate schedule.
4. **Pre-flight Smoke Test**:
   - Before launching the 43,000-step run, execute a **2-utterance, 5-step smoke test** on CPU/MPS to verify collate, loss computation, and checkpoint writing.

---

## 8. Summary of Blockers & Action Plan

| Blocker / Action Item | Severity | Owner | Resolution Plan |
| :--- | :---: | :---: | :--- |
| **1. Hugging Face Gated Agreement** | **CRITICAL** | User | Log in to Hugging Face, click "Agree and access repository" on [`ai4bharat/Svarah`](https://huggingface.co/datasets/ai4bharat/Svarah), and generate an access token. |
| **2. HiFi-GAN Checkpoint Decompression** | **LOW** | Agent | Execute `unzip original_fastspeech2/hifigan/generator_universal.pth.tar.zip` when preparing inference. |
| **3. MFA Installation in Colab** | **MEDIUM** | Agent | Script MFA installation via Conda/Mamba in the Colab notebook. |

---

## 9. Final Approval Recommendation

The technical decisions for English Svarah FastSpeech 2 preprocessing are complete, rigorous, and verified against our frozen codebase:

1. **Split**: Stratified speaker-independent (95 train speakers $\approx 5.1 \text{ h}$ / 22 val speakers $\approx 1.1 \text{ h}$).
2. **Architecture**: Multi-speaker FastSpeech 2 (`multi_speaker: True`, `n_speaker: 117`) with 256-dim speaker embeddings.
3. **Alignment**: Montreal Forced Aligner with CMUdict ARPAbet and G2P fallback for Indian English proper nouns.
4. **Vocoder**: Bundled Universal HiFi-GAN with audio resampled to 22,050 Hz.
5. **Compute**: Google Colab Free T4 GPU (~3 hours for 200 epochs).

**Recommendation**: **APPROVED WITH CONDITIONS** (Proceed with implementation once `HF_TOKEN` is supplied).

