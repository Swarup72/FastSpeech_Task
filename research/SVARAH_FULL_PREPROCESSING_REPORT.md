# Svarah Full English Alignment & Acoustic Feature Preprocessing Report

**Assignment:** Implementation of FastSpeech 2-Based Text-to-Speech Synthesis  
**Stage:** Stage 2 — Step 1.4J-D  
**Date:** 5 November 2026  
**Author / Execution:** Automated Research & Development Pipeline  
**Dataset:** `ai4bharat/Svarah` (Indian English Subset)  
**Corpus Counts:**  
- **Training Pool:** 3,215 utterances submitted $\to$ **3,208 successfully aligned** (99.78% success rate, 4.56 hours active speech)  
- **Validation Pool:** 665 utterances submitted $\to$ **664 successfully aligned** (99.85% success rate, 0.92 hours active speech)  
- **Held-Out Test Pool:** 1,300 utterances strictly preserved untouched  
**Forced Aligner:** Montreal Forced Aligner (MFA) v3.4.2 on macOS ARM64  
**Feature Extraction Engine:** `English/alignment/preprocess_features.py`  
**Test Suite Verification:** 171/171 tests passing (`tests/test_svarah_preprocessing_qc.py`, `tests/test_duration_extraction.py`, `tests/test_english_normalizer.py`, and regression tests)  

---

## 1. Alignment Statistics

Forced alignment was executed using Montreal Forced Aligner v3.4.2 with Kaldi and OpenFST C++ binaries natively on Apple Silicon.

| Split | Target Utterances | Audio Duration | Successfully Aligned | Failed Alignments | Success Rate | Active Speech Frames | Active Speech Hours |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 3,215 | 4.998 hours | **3,208** | 7 | **99.78%** | 1,415,088 | **4.5636 h** |
| **Val** | 665 | 0.996 hours | **664** | 1 | **99.85%** | 284,360 | **0.9171 h** |
| **Total** | 3,880 | 5.994 hours | **3,872** | 8 | **99.79%** | 1,699,448 | **5.4807 h** |

### Execution Performance
- **Train Alignment Time:** 93.97 seconds on 6 CPU workers (~0.029s per utterance).
- **Val Alignment Time:** 63.45 seconds on single process (~0.095s per utterance).
- **Acoustic Feature Extraction & Normalization Time:** 100.64 seconds for all 3,872 utterances (~38 utterances/second).

---

## 2. Manual TextGrid Review (15 Representative Pilot Utterances)

Prior to full scaling, an acoustic boundary audit was conducted on 15 diverse pilot utterances across 7 distinct categories. Boundaries were inspected against raw waveform RMS energy bursts and syntactic structure:

| Category | Utterance ID | Language Background | Text / Word Focus | Acoustic Boundary & Quality Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Short (<2s)** | `281474976896704_f1716_chunk_3` | Telugu (F) | *"trains,"* | Initial consonant burst `T` [0.19-0.22s] (RMS 0.0150) cleanly separated from leading room silence [0.00-0.19s] (RMS 0.0023). Trailing silence marks speech cutoff at 0.88s. |
| **Short (<2s)** | `281474976884273_f1982_chunk_0` | Tamil (M) | *"By the way"* | Lead silence [0.00-0.09s] (RMS 0.0012) transitions into plosive `B` [0.09-0.24s] (RMS 0.0349). Words *by*, *the*, *way* have distinct vowel centers. |
| **Long (>10s)** | `281474976893301_f2100_chunk_1` | Hindi (F) | *"it is something which gives you warmth..."* | Initial hesitation [0.00-1.10s] (RMS 0.0304) detected; first spoken phone `IH1` [1.10-1.18s] surges to RMS 0.3392 (11x energy step). Internal pauses marked at syntactic clause boundaries. |
| **Long (>10s)** | `281474976883998_f2011_chunk_4` | Gujarati (M) | *"And not much is left at the behest..."* | 26 words, 99 phones. Lead silence [0.00-0.24s] (RMS 0.0023) followed by `AE1` [0.24-0.58s] (RMS 0.1561). Natural pause before preposition *at* captured as silence token. |
| **Indian Proper Noun** | `281474976883931_f769_chunk_0` | Malayalam (F) | *"Banaras Hindu University"* | *Banaras* aligned as `B AE0 N AA1 R AH0 Z` [1.02-1.53s]. Correct vowel elongation on second syllable. |
| **Indian Proper Noun** | `281474976884635_f3269_chunk_0` | Kannada (F) | *"Hasana ... Hoysala dynasty"* | *Hasana* aligned as `HH AA0 S AA1 N AH0` [0.73-1.36s]; *Hoysala* aligned as `HH OY0 S AA1 L AH0` [2.40-3.05s]. Diphthong `OY0` correctly localized. |
| **Indian Proper Noun** | `281474976888891_f3059_chunk_0` | Nepali (M) | *"Hotstar website"* | Compound stress resolved as `HH AA1 T S T AA2 R` [2.62-3.15s]. Plosive closure of `T` and onset of `S` sharp and distinct. |
| **Indian Proper Noun** | `281474976883831_f1642_chunk_0` | Nepali (M) | *"Fin care Small Finance Bank"* | *Fin care* parsed and aligned across two lexical entries [0.72-1.35s]. |
| **Acronym Spell-Out** | `281474976888938_f2386_chunk_1` | Konkani (F) | *"to UPI id"* | Letter-by-letter expansion *U P I* aligned as `Y UW1`, `P IY1`, `AY1` with distinct 40ms inter-letter pauses [0.31-1.22s]. |
| **Acronym Pronunciation** | `281474976884114_f2737_chunk_0` | Marathi (F) | *"approved by AICTE"* | Acronym *AICTE* recognized as single lexical item in CMUDict with valid phone boundaries. |
| **Internal Pauses (8 pauses)**| `281474976887780_f2878_chunk_0` | Telugu (F) | *"Goan food may be divided into..."* | 8 conversational hesitation pauses between phrases. The aligner allocated silence intervals (`sp`) rather than artificially stretching adjacent vowels. |
| **Mother Tongue (Assamese)** | `281474976885077_f1480_chunk_0` | Assamese (M) | *"Can I invest in mutual funds through my Yono SBI"* | Rapid speech rate; plosives `K` and `T` localized within 25ms windows. |
| **Mother Tongue (Punjabi)** | `281474976888182_f901_chunk_0` | Punjabi (F) | *"Set a reminder for my pending payment..."* | Clean alveolar fricative `S` [0.03-0.16s] (RMS 0.0150) following 30ms lead. |
| **Conversational Hesitation**| `281474976894892_f328_chunk_0` | Konkani (M) | *"What type of learning system do you use"* | Internal hesitation pause between *learning* and *system* captured as 60ms silence. |
| **Short Command** | `281474976893690_f1906_chunk_0` | Odia (M) | *"Turn off the lights"* | Imperative sentence; initial plosive `T` and terminal unvoiced fricatives `TS` clearly delimited. |

---

## 3. Failed Utterances and Rejection Reasons

Across all 3,880 submitted samples, exactly **8 utterances failed alignment** (7 train, 1 val). None were silently dropped; all are cataloged with verified root causes:

### Training Failures (7 utterances)
1. `281474976884135_f3212_chunk_13` (spk_014, 1.0s):
   - *Text:* *"So yeah, that is the reason."* (7 words).
   - *Cause:* Truncated audio. Speaker uttered only the first two words before the audio cut off at 1.00s. Kaldi acoustic model could not fit 18 phonemes into 87 frames.
2. `281474976906753_f2448_chunk_0` (spk_063, 3.79s):
   - *Text:* Raw text was *"I have reset my MPIN to 6279"*. Normalizer expanded `6279` to *"six thousand two hundred and seventy-nine"*, but the speaker audibly recited digits *"six two seven nine"*. Severe acoustic-transcript mismatch caused lattice exhaustion.
3. `281474976883859_f1878_chunk_4` (spk_102, 14.51s):
   - *Text:* 36 words, 140 phonemes. Kaldi beam search pruned alignment path due to trailing speaker fade and room echo.
4. `281474976925735_f3143_chunk_1` (spk_003, 6.78s):
   - *Text:* *"Like if you talk about the field there are different lakes, parks, waterfalls."*
   - *Cause:* Extreme background noise and non-speech vocalization in speaker profile 3.
5. `281474976925736_f3275_chunk_0` (spk_003, 7.0s):
   - *Text:* *"In this Jammu and Kashmir Government New King's List by Manoh Sena..."*
   - *Cause:* Overlapping background talker in audio recording.
6. `281474976925739_f3299_chunk_0` (spk_003, 9.0s):
   - *Text:* *"In the 21st century, the entire world has become narrow..."*
   - *Cause:* Muffled microphone artifact / clipped waveform.
7. `281474976925739_f3299_chunk_3` (spk_003, 6.0s):
   - *Text:* *"As English has got the common qualities, it has been accepted..."*
   - *Cause:* Speaker trailing whisper and breathing noise.

### Validation Failure (1 utterance)
1. `281474976879629_f1172_chunk_0` (spk_006, 10.16s):
   - *Text:* *"You are 2 hours late than the delivery time for my products Dettol Shaving Cream and Good knight Activ+ Low Smoke Mosquito Coil"*
   - *Cause:* Literal plus sign `Activ+` was retained during normalization. Because `+` was not in the frozen training pronunciation dictionary, alignment failed on the unrepresented token.

All 8 failed utterances were rejected cleanly without corrupting the dataset or causing runtime exceptions.

---

## 4. Mel-Spectrogram Extraction Configuration

Acoustic features were configured to match the selected pretrained **HiFi-GAN Universal Vocoder** (`original_fastspeech2/hifigan/config.json`):

- **Sampling Rate ($f_s$):** 22,050 Hz (resampled from 16,000 Hz using 64-tap Kaiser sinc interpolation).
- **FFT Size ($N_{\text{fft}}$):** 1,024 samples.
- **Hop Length ($H$):** 256 samples (~11.6 ms frame shift).
- **Window Length ($W$):** 1,024 samples (~46.4 ms Hann window).
- **Mel Filterbank Channels:** 80 channels, Slaney-style area normalization.
- **Frequency Range:** $f_{\min} = 0.0$ Hz, $f_{\max} = 8,000.0$ Hz.
- **Dynamic Range Compression:** Logarithmic compression: $\text{mel} = \ln(\max(\text{basis} \cdot |X|, 10^{-5}))$.
- **Storage Layout:** 2D NumPy array transposed to shape $[T_{\text{mel}}, 80]$ (`float32`), matching FastSpeech 2 input conventions.

---

## 5. Pitch ($F_0$) Extraction Method

- **Algorithm:** PyWorld DIO (Distributed Interlocked Overlapping) algorithm followed by Stonemask refinement.
- **Time Resolution:** Continuous time step matching the STFT hop length ($256 / 22050 \times 1000 \approx 11.61$ ms).
- **Unvoiced Handling:**
  - Raw $F_0$ outputs $0.0$ Hz for unvoiced frames.
  - Linear interpolation (`scipy.interpolate.interp1d`) was applied across unvoiced frames bounded by voiced regions, with nearest-value extrapolation at utterance boundaries, producing a smooth continuous pitch contour suitable for neural predictor regression.
  - Zero-voiced utterances (e.g. whispers) fallback to $0.0$.
- **Feature Level Representations:**
  - **Frame-Level:** Stored in `pitch/` as $[T_{\text{mel}}]$ `float32` arrays.
  - **Phoneme-Level:** Stored in `pitch_phoneme/` as $[L_{\text{phone}}]$ `float32` arrays, computed as the mean frame pitch over each phone interval $[p_{\text{start}}, p_{\text{end}}]$.
- **Normalization (Strict Train Split Isolation):**
  - Outliers filtered using the IQR rule ($[Q_{25} - 1.5 \cdot \text{IQR}, Q_{75} + 1.5 \cdot \text{IQR}]$) on training data only.
  - **Training Frame-Level Pitch Mean:** $177.93$ Hz
  - **Training Frame-Level Pitch Std:** $59.45$ Hz
  - **Training Phoneme-Level Pitch Mean:** $179.83$ Hz
  - **Training Phoneme-Level Pitch Std:** $58.40$ Hz
  - Normalized pitch: $P_{\text{norm}} = (P - \mu) / \sigma$.
  - Validation pitch was normalized strictly using the frozen training mean and standard deviation.

---

## 6. Energy Extraction Method

- **Formula:** Frame-level L2 norm of the linear STFT magnitude spectrum:
  $$E(t) = \sqrt{\sum_{f=0}^{N_{\text{fft}}/2} |X(f, t)|^2}$$
- **Alignment:** Computed synchronously with the 80-bin mel-spectrogram using identical windowing and center reflect-padding.
- **Feature Level Representations:**
  - **Frame-Level:** Stored in `energy/` as $[T_{\text{mel}}]$ `float32` arrays.
  - **Phoneme-Level:** Stored in `energy_phoneme/` as $[L_{\text{phone}}]$ `float32` arrays, averaged across each phoneme interval.
- **Normalization (Strict Train Split Isolation):**
  - **Training Frame-Level Energy Mean:** $27.31$
  - **Training Frame-Level Energy Std:** $30.36$
  - **Training Phoneme-Level Energy Mean:** $30.58$
  - **Training Phoneme-Level Energy Std:** $28.53$
  - Normalized energy: $E_{\text{norm}} = (E - \mu) / \sigma$.
  - Applied frozen training statistics to validation energy.

---

## 7. Duration Statistics

Durations were discretized using telescoping boundary quantization ($b_i = \text{round}((t_i - t_{\text{start}}) \times f_s / H)$), eliminating cumulative drift across utterances.

| Metric | Training Set | Validation Set |
| :--- | :--- | :--- |
| **Total Phoneme Tokens** | 160,109 | 32,547 |
| **Mean Duration per Phone** | 8.84 frames (102.6 ms) | 8.74 frames (101.4 ms) |
| **Median Duration per Phone** | 7.0 frames (81.3 ms) | 7.0 frames (81.3 ms) |
| **Standard Deviation** | 6.82 frames (79.2 ms) | 6.51 frames (75.6 ms) |
| **Minimum Spoken Duration** | 1 frame (11.6 ms) | 1 frame (11.6 ms) |
| **Maximum Duration** | 497 frames (5.77 s pause) | 221 frames (2.57 s pause) |
| **Zero-Duration Spoken Phones** | **0** (none clamped or dropped) | **0** (none clamped or dropped) |

---

## 8. Reconciliation Adjustments

Because STFT windowing center reflect-padding evaluates audio samples with integer frame truncation ($N_{\text{samples}} = \lfloor f_s \cdot t_{\text{end}} \rfloor - \lfloor f_s \cdot t_{\text{start}} \rfloor$), raw boundary-rounded durations can occasionally differ from the STFT mel sequence length by $\Delta = T_{\text{mel}} - \sum d_i \in \{0, +1\}$.

- **Training Utterances Reconciled:** 1,586 / 3,208 (49.4%) adjusted by exactly $+1$ frame.
- **Training Utterances Exact:** 1,622 / 3,208 (50.6%) required zero adjustment ($\Delta = 0$).
- **Validation Utterances Reconciled:** 350 / 664 (52.7%) adjusted by exactly $+1$ frame.
- **Validation Utterances Exact:** 314 / 664 (47.3%) required zero adjustment ($\Delta = 0$).
- **Maximum Per-Utterance Adjustment:** **1 frame** (11.6 ms) across the entire dataset. Zero utterances required 2 or more frames of adjustment.
- **Reconciliation Algorithm:** Extra frames were allocated strictly to intervals with the largest positive rounding residual $r_i = ((e_i - s_i) \cdot f_s / H) - d_i$ (preferring silence tokens `'sp'` or elongated open vowels), strictly preserving phoneme temporal order and acoustic naturalness.

Following reconciliation, **100.0% of training and validation utterances satisfy $\sum_{i=1}^N d_i == T_{\text{mel}}$**.

---

## 9. Feature Dimensions and Directory Storage Layout

All preprocessed files are serialized under `preprocessed_data/Svarah/` following standard FastSpeech 2 filesystem structure:

```text
preprocessed_data/Svarah/
├── duration/            # 3,872 files: {spk}-duration-{uid}.npy [L_phone] int32
├── mel/                 # 3,872 files: {spk}-mel-{uid}.npy [T_mel, 80] float32
├── pitch/               # 3,872 files: {spk}-pitch-{uid}.npy [T_mel] float32 (norm)
├── energy/              # 3,872 files: {spk}-energy-{uid}.npy [T_mel] float32 (norm)
├── pitch_phoneme/       # 3,872 files: {spk}-pitch-{uid}.npy [L_phone] float32 (norm)
├── energy_phoneme/      # 3,872 files: {spk}-energy-{uid}.npy [L_phone] float32 (norm)
├── speakers.json        # 115 profiles -> integer IDs (0..114)
├── stats.json           # Train-derived min, max, mean, std for pitch and energy
├── train.txt            # 3,208 lines: uid|speaker|{phones}|raw_text
└── val.txt              # 664 lines: uid|speaker|{phones}|raw_text
```

Total disk storage footprint: **~2.2 GB** (completely ignored by `.gitignore`).

---

## 10. Automated Quality Control & Dataset Integrity Results

A dedicated quality control suite was implemented in [`tests/test_svarah_preprocessing_qc.py`](file:///Users/swaruppal/FastSpeech2_Project/tests/test_svarah_preprocessing_qc.py).

| Quality Check | Verification Procedure | Result |
| :--- | :--- | :--- |
| **Missing Files** | Verified presence of mel, duration, pitch, energy files for every sample in `train.txt` and `val.txt`. | **0 missing files** |
| **Dimension Match** | $\text{mel.shape}[0] == \text{len(pitch)} == \text{len(energy)} == \sum d_i$ | **100.0% exact match** |
| **Mel Channels** | Verified $\text{mel.shape}[1] == 80$ across all files. | **100.0% valid** |
| **Duration Positivity** | Verified $d_i \ge 1$ for all 192,656 phonemes. | **100.0% positive** |
| **Non-Finite Values** | Checked for NaN, -Inf, +Inf across all mel, pitch, energy tensors. | **0 non-finite values** |
| **Symbol Membership** | Validated all phonemes against `original_fastspeech2/text/symbols.py`. | **0 unknown symbols** |
| **Speaker Profiles** | Verified all speakers match `spk_000` to `spk_114` mapped in `speakers.json`. | **100.0% valid** |
| **Split Isolation** | Verified `set(train_ids).isdisjoint(set(val_ids))`. | **0 ID overlaps** |
| **DataLoader Test** | Loaded batches with `original_fastspeech2/dataset.py` `Dataset` and collate `reprocess`. | **Batching verified** |

Regression suite execution across all test files:
```text
Ran 171 tests in 7.367s
OK
```

---

## 11. Remaining Limitations

1. **Demographic Profile Proxy Conditioning:**
   As established in Step 1.4J-B2, Svarah provides 115 demographic profile clusters (age, gender, state, native language, education) rather than verified single-speaker biometric identities. While this provides rich multi-accent conditioning, individual profiles may contain minor intra-cluster acoustic variance.
2. **Indian Loanword Stress Patterns:**
   G2P resolution mapped 875 OOV loanwords (e.g. *chokha*, *pincode*, *covishield*) into standard ARPAbet stress patterns. While phonetically natural, minor syllable-timing variations typical of Indian English are modeled through the duration predictor rather than explicit moraic annotations.

---

## 12. Training Readiness Decision

### Final Decision: **READY FOR TRAINING**

All prerequisites for FastSpeech 2 English model training have been achieved:
1. **Full Dataset Alignment:** 3,208 training utterances (4.56 hours) and 664 validation utterances (0.92 hours) successfully aligned with Montreal Forced Aligner (99.79% overall success rate).
2. **Exact Mathematical Consistency:** 100% of samples satisfy $\sum d_i == T_{\text{mel}} == \text{len}(F_0) == \text{len}(E)$.
3. **Strict Data Leakage Guard:** Validation and test splits were never used for OOV lexicon expansion, vocabulary definitions, or normalization statistics.
4. **Full DataLoader Compatibility:** Verified end-to-end with the FastSpeech 2 training dataset loader and batch collation engine.
5. **Regression Verification:** 171/171 unit and integration tests passing.

