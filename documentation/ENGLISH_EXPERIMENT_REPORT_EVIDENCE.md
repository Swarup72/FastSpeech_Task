# English Experiment Report Evidence Inventory
## Academic Assignment: Implementation of FastSpeech 2-Based Text-to-Speech Synthesis

**Author / Researcher**: Swarup Pal  
**Project**: FastSpeech 2 Text-to-Speech Synthesis (Svarah English & IndicTTS Telugu)  
**Date**: November 2026  
**Status**: English Experiment Completed, Integrated, and Audited  

---

## 1. Dataset Specification

### 1.1 Dataset Source and Metadata
- **Corpus**: Svarah Indian English Speech Corpus (part of the AI4Bharat IndicVoices initiative / IISc Svarah project).
- **Target Domain**: Multi-speaker Indian English with natural regional accent variations.
- **Total Corpus Size (Preprocessed)**: 3,872 speech utterances across 88 distinct speaker profiles.
- **Total Audio Duration**: 5.4807 hours (19,730.5 seconds).
- **Total Mel Frames**: 1,699,448 frames (hop length 256 @ 22,050 Hz).
- **Total Phoneme Tokens**: 192,656 tokens.
- **Audio Format**: Single-channel (mono) 16-bit PCM WAV, sampled at 22,050 Hz.

### 1.2 Profile-Stratified Train / Validation Split
The dataset partition follows the `profile_stratified_v1` partition strategy (`English/metadata/splits/profile_stratified_v1/`) designed to address speaker and acoustic imbalance while eliminating data leakage:

| Metric | Training Split (`train.txt`) | Validation Split (`val.txt`) | Combined Preprocessed |
| :--- | :--- | :--- | :--- |
| **Utterance Count** | 3,208 | 664 | 3,872 |
| **Duration (Hours)** | 4.5023 hrs | 0.9784 hrs | 5.4807 hrs |
| **Total Mel Frames** | 1,396,077 | 303,371 | 1,699,448 |
| **Total Phonemes** | 158,660 | 33,996 | 192,656 |
| **Speaker Profiles** | 88 (100%) | 88 (100% seen in train) | 88 |
| **Gender Distribution** | 1,589 Female (49.53%) / 1,619 Male (50.47%) | 332 Female (50.00%) / 332 Male (50.00%) | Balanced |
| **Mean Pitch $z$-score** | -0.0166 | -0.0549 | ~0.0000 |
| **Manifest SHA-256** | `1fcd48ec9957153318f4a8626fa2e8a3d16ff37841b75fcde877edfa3c4a6116` | `aa969d5b9b069bf1b4288c4f1b60f800f7fc7de7d511938ff379207cc658aab2` | N/A |

### 1.3 Split Audit and Leakage Elimination Verification
A comprehensive audit (`documentation/English_Training_Evidence/split_metadata/split_audit.json`) confirmed:
- **Utterance ID Overlap**: 0 utterances.
- **User Prefix Overlap**: 0 prefixes.
- **Recording ID Overlap**: 0 recording sessions.
- **Raw Transcript Overlap**: 0 sentences.
- **Manifest Normalized Transcript Overlap**: 0 sentences.
- **Test Set Preservation**: 1,300 test utterances kept completely untouched and frozen.
- **Pitch Baseline Floor**: Relative MSE floor reduction of 63.25% achieved by speaker profile stratification.

### 1.4 Feature Preprocessing Configuration
Extracted via `config/Svarah/preprocess.yaml`:
- **Sampling Rate**: 22,050 Hz (`max_wav_value = 32768.0`).
- **STFT Parameters**: Filter length: 1024, Hop length: 256, Window length: 1024 (Hann window).
- **Mel Spectrogram**: 80 frequency bins, $f_{\min} = 0$ Hz, $f_{\max} = 8,000$ Hz.
- **Pitch Extraction**: Continuous fundamental frequency ($F_0$) extracted via PyWorld (DIO / Harvest), frame-level alignment, $z$-score normalized.
  - Dataset Pitch Mean: 177.9316 Hz
  - Dataset Pitch Standard Deviation: 59.4528 Hz
  - Dataset Pitch Range: [-2.9928, 10.1252] normalized ($z$-score)
- **Energy Extraction**: Frame-level STFT magnitude L2-norm ($L_2 = \sqrt{\sum |S(f, t)|^2}$), $z$-score normalized.
  - Dataset Energy Mean: 27.3145 (linear STFT magnitude units)
  - Dataset Energy Standard Deviation: 30.3573
  - Dataset Energy Range: [-0.8998, 7.4643] normalized ($z$-score)
- **Text & G2P**: `english_cleaners` text normalizer + CMU Pronouncing Dictionary with fallback ARPAbet G2P via `g2p_en`.

---

## 2. Model Architecture & Contract Specification

### 2.1 FastSpeech 2 Architecture
The model adheres strictly to the non-autoregressive feed-forward Transformer architecture proposed by Ren et al. (ICLR 2021):
- **Phoneme Encoder**: 4 Feed-Forward Transformer (FFT) blocks.
  - Attention: Multi-head attention (2 heads, hidden dim $d_{\text{model}} = 256$).
  - 1D Convolutions: 2-layer 1D Conv with filter size 1024, kernel sizes [9, 1], dropout 0.2.
- **Mel Decoder**: 6 FFT blocks.
  - Multi-head attention (2 heads, hidden dim $d_{\text{model}} = 256$).
  - 1D Convolutions: Filter size 1024, kernel sizes [9, 1], dropout 0.2.
- **Mel Post-Net**: 5-layer 1D convolutional network with 512 channels, kernel size 5, batch normalization, and tanh activations, computing residual mel corrections.
- **Multi-Speaker Conditioning**: Speaker embedding lookup table with 115 speakers, embedding dimension 256, projected and added to the encoder output.

### 2.2 Variance Adaptor
Provides explicit prosody conditioning to bridge the one-to-many mapping problem in TTS:
1. **Duration Predictor**: 2-layer 1D Conv (filter 256, kernel size 3, dropout 0.5) with LayerNorm and ReLU, followed by a Linear projection to log-duration. Supervised with ground-truth alignments extracted from Montreal Forced Aligner (MFA).
2. **Length Regulator**: Expands phoneme representations to frame-level representations via `torch.repeat_interleave` using integer-rounded durations.
3. **Pitch Predictor**: 2-layer 1D Conv (filter 256, kernel size 3, dropout 0.5) projecting to frame-level continuous $F_0$. Values are linearly quantized into 256 discrete bins and embedded into 256-dim pitch vectors added to the frame representations.
4. **Energy Predictor**: 2-layer 1D Conv (filter 256, kernel size 3, dropout 0.5) projecting to frame-level linear STFT L2-norm. Linearly quantized into 256 discrete bins and embedded into 256-dim energy vectors added to the frame representations.

### 2.3 Vocoder
- **Architecture**: HiFi-GAN (Generator V1).
- **Pretrained Checkpoint**: Universal speaker pretrained weights (LJSpeech / universal multi-speaker).
- **Input**: 80-channel predicted log-mel spectrogram.
- **Output**: 22,050 Hz 16-bit PCM waveform.

### 2.4 Frozen Interface Contracts
- `FastSpeech2.forward()` returns a **10-tuple**:
  `(output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks, src_masks, attn_weights)`
- `VarianceAdaptor.forward()` returns a **7-tuple**:
  `(x, d_prediction, p_prediction, e_prediction, mel_len, mel_mask, None)`
- `FastSpeech2Loss.forward()` returns a **6-tuple**:
  `(total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss)`
- Special Token Indices: `PAD = 0`, `UNK = 1`, `BOS = 2`, `EOS = 3`.

---

## 3. Training Regime & Execution Evidence

### 3.1 Optimizer and Hyperparameters
- **Optimization Algorithm**: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$, weight decay = $0.0$).
- **Gradient Clipping**: Threshold 1.0.
- **Gradient Accumulation**: 1 step.
- **Effective Batch Size**: 16 utterances.
- **Learning Rate Schedule**: Noam decay with initial lr = $0.0625$ and 4,000 warmup steps:
  $$\text{lr} = d_{\text{model}}^{-0.5} \cdot \min\left(step^{-0.5}, step \cdot \text{warmup}^{-1.5}\right)$$
- **Scheduled Annealing**: Annealing steps at [300,000, 400,000, 500,000] with rate 0.3.
- **Final Learning Rate**: $4.4194 \times 10^{-4}$ (at step 20,000).

### 3.2 Training Execution Summary
- **Execution Platform**: Google Colab (NVIDIA T4 / A100 GPU instance).
- **Total Epochs Completed**: 100 epochs.
- **Total Optimizer Steps**: 20,000 steps.
- **Total Wall-Clock Training Duration**: 1 hour, 23 minutes, 32 seconds (5,011.67 seconds).
- **Average Step Rate**: ~4.0 steps/second.

### 3.3 Loss Progression & Convergence

| Checkpoint Stage | Global Step | Epoch | Total Loss | Mel Loss | PostNet Mel Loss | Pitch Loss | Energy Loss | Duration Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial Step** | 100 | 1 | 8.8412 | 1.8410 | 1.8395 | 1.8210 | 1.1240 | 2.2157 |
| **Warmup End** | 4,000 | 20 | 3.1045 | 0.8124 | 0.8118 | 0.6521 | 0.4120 | 0.4162 |
| **Best Validation** | **19,000** | **95** | **2.3489** | **0.6712** | **0.6709** | **0.4551** | **0.2811** | **0.2706** |
| **Final Step (Train)** | 20,000 | 100 | 1.0537 | 0.3809 | 0.3803 | 0.1963 | 0.0804 | 0.0157 |
| **Final Step (Val)** | 20,000 | 100 | 2.3689 | 0.6746 | 0.6743 | 0.4596 | 0.2850 | 0.2754 |

### 3.4 Checkpoint Artifacts
1. **Best Validation Checkpoint (`checkpoint_best.pth.tar`)**:
   - Location: `output/ckpt/Svarah_profile_stratified_v1/checkpoint_best.pth.tar`
   - File Size: 418,465,625 bytes (~399.08 MB)
   - SHA-256: `794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e`
   - Epoch: 95 | Global Step: 19,000 | Validation Loss: 2.348900315273239
2. **Latest Training Checkpoint (`checkpoint_latest.pth.tar`)**:
   - Location: `output/ckpt/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar`
   - File Size: 418,467,535 bytes (~399.08 MB)
   - SHA-256: `6063dd284e32d7ec72b01280fc1254c9a4fc65065e2baaddfe0c31241b81cde3`
   - Epoch: 100 | Global Step: 20,000 | Validation Loss: 2.348900315273239 (best metric state)

### 3.5 Training Visualizations Available for Technical Report
Located in `documentation/English_Training_Evidence/plots/`, `output/logs/Svarah_profile_stratified_v1/plots/`, and mirrored to `report/figures/`:
- `graph1_train_total_loss.png`: Training total loss trajectory over 20,000 steps.
- `graph2_val_total_loss.png`: Validation total loss trajectory evaluated every 1,000 steps.
- `graph3_train_loss_components.png`: Disaggregated training loss components (Mel, PostNet, Pitch, Energy, Duration).
- `graph4_val_loss_components.png`: Disaggregated validation loss components demonstrating simultaneous convergence across acoustic targets.
- `graph5_learning_rate.png`: Learning rate schedule curve depicting linear warmup and Noam decay.
- `loss_curves_dashboard.png`: Unified 6-panel technical summary dashboard.

---

## 4. Evaluation: Synthesized English Samples & Prosody Analysis

### 4.1 Sample Inventory
Inference was performed using the verified `checkpoint_best.pth.tar` with HiFi-GAN vocoding. The synthesized audio files are preserved in `Results/English/audio/`:

| ID | Synthesized Text | Audio File | Duration (s) | Phonemes | Mel Frames |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings." | `E1.wav` | 6.06 s | 85 | 522 |
| **E2a** | "The orchestra's conduct during the ceremony was praised by the audience." | `E2a.wav` | 3.87 s | 50 | 333 |
| **E2b** | "The researchers will conduct an experiment to evaluate the proposed method." | `E2b.wav` | 4.16 s | 56 | 358 |
| **E3** | "The rhyth m of the orchestra changed unexpectedly." | `E3.wav` | 2.79 s | 37 | 240 |
| **E4** | "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time." | `E4.wav` | 6.70 s | 81 | 567 |

### 4.2 Prosodic Prediction Statistics (Physical Units)
The predicted prosody was converted from standardized $z$-scores to physical units ($F_0$ in Hertz; Energy in linear STFT magnitude L2-norm):
- $F_0 (\text{Hz}) = z_{\text{pitch}} \cdot 59.4528 + 177.9316$
- $\text{Energy} = \max\left(0, z_{\text{energy}} \cdot 30.3573 + 27.3145\right)$

| Sample | Pitch Range (Hz) | Pitch Mean (Hz) | Pitch Std (Hz) | Energy Range (Linear L2) | Energy Mean | Energy Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | 191.6 – 298.8 Hz | 243.9 Hz | 21.8 Hz | 2.4 – 64.7 | 27.5 | 11.2 |
| **E2a** | 146.6 – 289.8 Hz | 231.4 Hz | 26.9 Hz | 2.6 – 60.8 | 25.5 | 11.7 |
| **E2b** | 179.8 – 273.2 Hz | 230.2 Hz | 22.1 Hz | 2.2 – 53.9 | 24.4 | 11.0 |
| **E3** | 192.5 – 273.8 Hz | 237.9 Hz | 20.4 Hz | 2.7 – 74.3 | 26.6 | 12.3 |
| **E4** | 193.8 – 292.3 Hz | 238.5 Hz | 22.7 Hz | 0.05 – 80.5 | 25.8 | 12.9 |

### 4.3 Evaluation Plots
All 15 evaluation plots are preserved in `Results/English/plots/` and mirrored to `report/figures/`:
1. **Duration Plots** (`E1_duration.png` to `E4_duration.png`): High-resolution bar charts depicting predicted frame durations per phoneme symbol across each sentence.
2. **Pitch Plots** (`E1_pitch.png` to `E4_pitch.png`): Trajectories of predicted pitch in physical Hertz across continuous time frames.
3. **Energy Plots** (`E1_energy.png` to `E4_energy.png`): Trajectories of predicted energy in Linear STFT Magnitude L2-norm across continuous time frames.

---

## 5. Linguistic & Acoustic Observations

### 5.1 Morphological Complexity & Long Compound Words (E1)
- **Challenge**: "antidisestablishmentarianism" (28 letters, 12 syllables).
- **Observation**: FastSpeech 2 successfully synthesized the compound sequence without catastrophic collapse, duration degradation, or phoneme repetition. Phoneme durations across the stem remain stable, demonstrating the robustness of non-autoregressive duration regulation over recurrent or attention-based step accumulation.

### 5.2 Heteronym Disambiguation (E2a vs. E2b)
- **Challenge**: Lexical stress shift between noun ("c**o**nduct", /ˈkɒndʌkt/) in E2a and verb ("cond**u**ct", /kənˈdʌkt/) in E2b.
- **Observation**: The front-end G2P dictionary lookup lacks contextual part-of-speech disambiguation, predicting identical phonetic strings for both forms. Consequently, FastSpeech 2 produces near-identical pitch and duration contours for the word token in both contexts. This highlights an intrinsic pipeline dependency: acoustic prosody adaptors cannot compensate for upstream lexical transcription ambiguities without POS-aware phonemization.

### 5.3 Orthographic Irregularities & Explicit Token Splits (E3)
- **Challenge**: Split spelling "rhyth m" representing unnatural whitespace insertion.
- **Observation**: The model preserves an explicit acoustic transition corresponding to the space delimiter, resulting in a minor micro-pause before the isolated bilabial nasal /m/. The surrounding prosodic curve remains continuous without destabilizing subsequent words.

### 5.4 Code-Switching & Non-Standard Normalization (E4)
- **Challenge**: Mixed sentence with time ("10:45 AM"), date ("12 November 2026"), and a regional Telugu loanword ("andaroo").
- **Observation**: The normalizer expands numerical time and date expressions into standard verbal form. The out-of-vocabulary word "andaroo" is handled via `g2p_en` rule-based letter-to-sound grapheme decomposition, synthesizing a plausible phonetic approximation (/æ n d ɑː r uː/) without producing an execution error.

---

## 6. Known Limitations & Technical Constraints

1. **Vocoder Timbre Domain Gap**:
   HiFi-GAN Universal Generator V1 was utilized out-of-the-box. While it reconstructs high-fidelity phase and waveform details, subtle spectral artifacts and timbre mismatch arise because it was primarily pretrained on American English (LJSpeech). Finetuning the vocoder on Indian English multi-speaker mel spectrograms would further improve naturalness.
2. **Phoneme-Level vs. Frame-Level Prosody Resolution**:
   Predicting continuous frame-level pitch and energy provides rich acoustic contours, but rapid unvoiced-to-voiced transitions occasionally exhibit slight smoothing compared to natural ground-truth recordings.
3. **G2P Stress Invariant**:
   Without deep contextual phonemization (e.g., BERT-based POS tagging), heteronyms and homographs inherit the dictionary default stress pattern.
4. **Subjective Quality Assessment Scope**:
   Per academic integrity standards, audio naturalness is evaluated observationally based on phonetic intelligibility and spectrogram clarity; formal Mean Opinion Score (MOS) crowdsourced listening tests were not conducted.

