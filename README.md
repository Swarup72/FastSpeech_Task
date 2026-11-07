# FastSpeech 2: Non-Autoregressive Multi-Lingual Speech Synthesis
## Dual-Model Implementation for English (Svarah) and Telugu (IndicTTS)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Institution: IIIT Hyderabad](https://img.shields.io/badge/Institution-IIIT%20Hyderabad-orange.svg)](https://www.iiit.ac.in/)

**Assignment Title**: Implementation of FastSpeech 2-Based Text-to-Speech Synthesis  
**Author / Candidate**: Swarup Pal (`swaruppal1511@gmail.com`)  
**Course**: Speech Signal Processing / Advanced Speech Synthesis  
**Institution**: International Institute of Information Technology, Hyderabad (IIIT Hyderabad)  
**Submission Deadline**: 07-11-2026  
**Primary Deliverables**: Two Separately Trained Acoustic Models, 10 Test Audio Samples, 30 Variance Adaptor Plots, Training Loss Analysis, Comprehensive Technical Report ([PDF](Report/Technical_Report.pdf) & [Markdown](Report/Technical_Report.md)).

---

## 1. Executive Summary

This repository provides a complete, reproducible implementation of **FastSpeech 2** for non-autoregressive text-to-speech synthesis across two contrasting linguistic typologies:
1. **English Model**: Trained on the **Svarah** Indian English corpus across 88 voice profiles using a CMUdict phonetic lexicon.
2. **Telugu Model**: Trained on the **IndicTTS** Telugu speech corpus using character-level **phonetic grapheme tokenization** over a 101-token vocabulary.

Both models were trained independently for 20,000 steps without shared weights. To prevent pitch degradation and over-smoothing, the best observed validation checkpoints were selected at **Step 19,000** for English (validation loss: 2.3489) and **Step 13,000** for Telugu (validation loss: 2.0869). All ten canonical assignment test sentences were synthesized into 22,050 Hz 16-bit mono PCM waveforms using a universal HiFi-GAN neural vocoder.

---

## 2. Key Architectural Features

- **Non-Autoregressive Parallel Generation**: 4-layer Feed-Forward Transformer (FFT) encoder and decoder blocks with multi-head self-attention and 1D convolutions, eliminating the sequential generation bottleneck and exposure bias of autoregressive models.
- **Explicit Variance Adaptor**:
  - **Duration Predictor & Length Regulator**: Expands hidden phoneme representations to frame-level mel resolution using Montreal Forced Aligner (MFA) ground-truth alignments during training and convolutional log-duration predictions during inference.
  - **Pitch Predictor**: Predicts frame-level continuous fundamental frequency ($F_0$), quantized into 256 discrete embedding bins.
  - **Energy Predictor**: Predicts frame-level STFT magnitude $L_2$-norm energy, quantized into 256 discrete embedding bins.
  - **Speaker Embedding**: Supports multi-speaker conditioning (88 Svarah profiles; 2 IndicTTS speakers: `spk_000` female, `spk_001` male).
- **Telugu-Specific Engineering**:
  - Eliminates external pronunciation dictionaries in favor of direct alpha-syllabic grapheme tokenization.
  - 101-token inventory covering vowels, consonants, gunintalu/matras, virama, diacritics, and native numerals.
  - Unicode NFC normalization and Latin-$o$ anusvara (`o` $\to$ `ం`) correction.
  - Inter-word and clause-boundary pause mapping (`@sp`, token ID 4), avoiding untrained punctuation IDs.
- **Controllable Prosody**: Independent inference-time scaling multipliers for duration ($\alpha_d$), pitch ($\alpha_p$), and energy ($\alpha_e$).

---

## 3. Project Directory Structure

The repository adheres strictly to the assignment-prescribed submission layout:

```text
FastSpeech2_Project/
├── English/                                  # English acoustic model pipeline
│   ├── train.py                              # English training script with checkpointing
│   ├── inference.py                          # English synthesis and plotting CLI
│   ├── monitoring.py                         # Live training logger and 5-panel loss dashboard
│   ├── alignment/                            # Svarah MFA alignment & feature extraction scripts
│   ├── metadata/                             # Svarah manifests & profile_stratified_v1 splits
│   └── text/normalizer.py                    # English text normalizer & numeral/time expansion
├── Telugu/                                   # Telugu acoustic model pipeline
│   ├── train.py                              # Telugu training script with checkpointing
│   ├── inference.py                          # Telugu synthesis and plotting CLI
│   ├── alignment/                            # IndicTTS resampling, MFA Option 1B & feature scripts
│   ├── metadata/                             # IndicTTS manifests, vocabulary audit & splits
│   └── text/                                 # Telugu NFC normalizer, symbols & grapheme tokenizer
├── common/                                   # Shared architecture & utilities
│   ├── paths.py                              # Centralized project root & path manager
│   ├── config/                               # Validated YAML configuration schemas
│   ├── model/                                # FastSpeech 2 model, Variance Adaptor, PostNet, Loss
│   ├── transformer/                          # FFT Encoder/Decoder & Multi-Head Self-Attention
│   ├── training/                             # Trainer engine & SHA-256 CheckpointManager
│   └── utils/                                # Masking, padding, device & plotting utilities
├── config/                                   # Preprocessing, model, and training YAML configs
│   ├── Svarah/                               # English configurations
│   └── IndicTTS/                             # Telugu configurations
├── checkpoints/                              # Trained model weights (excluded from Git)
│   ├── Svarah_profile_stratified_v1/         # Step 19,000 best (2.3489), Step 20,000 latest
│   └── IndicTTS/                             # Step 13,000 best (2.0869), Step 20,000 latest
├── Results/                                  # Complete assignment outputs
│   ├── English/
│   │   ├── training_loss.png                 # Training loss curve (Steps 100–20,000)
│   │   ├── validation_loss.png               # Validation loss curve (Steps 1,000–20,000)
│   │   ├── test_generated_audios/            # E1.wav, E2a.wav, E2b.wav, E3.wav, E4.wav
│   │   └── variance_adaptor_plots/           # 15 plots across duration/, pitch/, energy/
│   └── Telugu/
│       ├── training_loss.png                 # Training loss curve (Steps 100–20,000)
│       ├── validation_loss.png               # Validation loss curve (Steps 1,000–20,000)
│       ├── official_samples.json             # Verified canonical Telugu test sentences
│       ├── test_generated_audios/            # T1.wav, T2_a.wav, T2_b.wav, T3.wav, T4.wav
│       └── variance_adaptor_plots/           # 15 plots across duration/, pitch/, energy/
├── Report/
│   ├── Technical_Report.pdf                  # Rendered 30-page research report
│   ├── Technical_Report.md                   # Complete Markdown report source
│   └── figures/                              # All 41 loss plots, variance plots & figures
├── documentation/
│   ├── FINAL_ARTIFACT_MANIFEST.md            # Comprehensive artifact checksum manifest
│   └── final_artifact_manifest.json          # Machine-readable JSON deliverable manifest
├── original_fastspeech2/                     # Reference implementation & HiFi-GAN vocoder
│   └── hifigan/                              # Universal generator weights & config
├── tests/                                    # Unit and integration test suite
├── README.md                                 # Project documentation
└── requirements.txt                          # Pinned Python dependencies
```

---

## 4. Trained Checkpoint & Vocoder Verification

Both acoustic models were trained for 20,000 steps on NVIDIA GPUs. Cryptographic SHA-256 checksums confirm model and vocoder authenticity:

| Component | Local Workspace Path | Global Step | Epoch | Step Val Loss | Best Val Loss | Size (MiB) | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **English Best** | `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` | **19,000** | **95** | **2.348900** | **2.348900** | 399.08 | `794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e` |
| **English Latest** | `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` | 20,000 | 100 | 2.368902 | 2.348900 | 399.08 | `6063dd284e32d7ec72b01280fc1254c9a4fc65065e2baaddfe0c31241b81cde3` |
| **Telugu Best** | `checkpoints/IndicTTS/checkpoint_best.pth.tar` | **13,000** | **91** | **2.086877** | **2.086877** | 397.99 | `0a6c443a7baea134f2c9c619d603ddcec8c4d1f86fdfd306680931f4a4cb55fd` |
| **Telugu Latest** | `checkpoints/IndicTTS/checkpoint_latest.pth.tar` | 20,000 | 140 | 2.118075 | 2.086877 | 397.99 | `8e46b77bedb23c80703ae0309a9c7fa01806e13db3b515ba88e9bb6005f4556f` |
| **HiFi-GAN Vocoder** | `original_fastspeech2/hifigan/generator_universal.pth.tar` | Pretrained | — | — | — | 53.20 | `771eaf48adb7bcbeab2fa4ecfb1abff3eb6fae99efde5dc1ffba3fb3ec6aa2e6` |

### 4.1 Checkpoint & HiFi-GAN Hosting and Download Instructions

Due to GitHub's 100 MB per-file upload limit for standard Git commits, all `.pth.tar` model weights (`checkpoints/` at ~398 MiB per checkpoint and `original_fastspeech2/hifigan/generator_universal.pth.tar` at 53.20 MiB) are excluded from ordinary Git tracking via `.gitignore` (`checkpoints/`, `*.pth.tar`).

1. **Hosting Strategy for Public Repository**:
   - **GitHub Releases (Recommended)**: Upload `checkpoint_best.pth.tar` (English Step 19,000), `checkpoint_best.pth.tar` (Telugu Step 13,000), and `generator_universal.pth.tar` (HiFi-GAN Universal v1) as release binary assets on the GitHub Releases page (supports up to 2 GB per asset).
   - **Upstream Vocoder Mirror**: The universal HiFi-GAN generator is also available from the reference repository (`ming024/FastSpeech2` / `jik876/hifi-gan` Universal v1) alongside its tracked architecture config [`original_fastspeech2/hifigan/config.json`](original_fastspeech2/hifigan/config.json).
   - **Git LFS (Optional)**: Alternatively, track weights via Git Large File Storage (`git lfs install && git lfs track "*.pth.tar"`).

2. **Local Placement & Checksum Verification After Cloning**:
   Place downloaded weights into their exact canonical paths and verify SHA-256 integrity:
   ```bash
   mkdir -p checkpoints/Svarah_profile_stratified_v1 checkpoints/IndicTTS original_fastspeech2/hifigan

   # Place downloaded files at:
   #   checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar
   #   checkpoints/IndicTTS/checkpoint_best.pth.tar
   #   original_fastspeech2/hifigan/generator_universal.pth.tar

   # Verify SHA-256 checksums:
   shasum -a 256 checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar
   # Expected: 794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e

   shasum -a 256 checkpoints/IndicTTS/checkpoint_best.pth.tar
   # Expected: 0a6c443a7baea134f2c9c619d603ddcec8c4d1f86fdfd306680931f4a4cb55fd

   shasum -a 256 original_fastspeech2/hifigan/generator_universal.pth.tar
   # Expected: 771eaf48adb7bcbeab2fa4ecfb1abff3eb6fae99efde5dc1ffba3fb3ec6aa2e6
   ```

---

## 5. Canonical Test Sentences & Audio Gallery

All 10 audio samples were synthesized at 22,050 Hz, 16-bit mono PCM.

### 5.1 English Assignment Samples (Speaker Profile 0, Female Indian English)

| ID | Input Text | Tokens | Duration | Audio File | Plots |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | 85 | 6.06 s | [`E1.wav`](Results/English/test_generated_audios/E1.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E1_duration.png) \| [Pitch](Results/English/variance_adaptor_plots/pitch/E1_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E1_energy.png) |
| **E2a** | *The orchestra's conduct during the ceremony was praised by the audience.* | 50 | 3.87 s | [`E2a.wav`](Results/English/test_generated_audios/E2a.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E2a_duration.png) \| [Pitch](Results/English/variance_adaptor_plots/pitch/E2a_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E2a_energy.png) |
| **E2b** | *The researchers will conduct an experiment to evaluate the proposed method.* | 56 | 4.16 s | [`E2b.wav`](Results/English/test_generated_audios/E2b.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E2b_duration.png) \| [Pitch](Results/English/variance_adaptor_plots/pitch/E2b_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E2b_energy.png) |
| **E3** | *The rhyth m of the orchestra changed unexpectedly.* | 37 | 2.79 s | [`E3.wav`](Results/English/test_generated_audios/E3.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E3_duration.png) \| [Pitch](Results/English/variance_adaptor_plots/pitch/E3_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E3_energy.png) |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | 81 | 6.58 s | [`E4.wav`](Results/English/test_generated_audios/E4.wav) | [Duration](Results/English/variance_adaptor_plots/duration/E4_duration.png) \| [Pitch](Results/English/variance_adaptor_plots/pitch/E4_pitch.png) \| [Energy](Results/English/variance_adaptor_plots/energy/E4_energy.png) |

### 5.2 Telugu Assignment Samples (Speaker `spk_001`, Male Native Telugu)

| ID | Exact Input Text | Tokens | Duration | Audio File | Plots |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **T1** | న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. | 67 | 5.39 s | [`T1.wav`](Results/Telugu/test_generated_audios/T1.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T1_duration.png) \| [Pitch](Results/Telugu/variance_adaptor_plots/pitch/T1_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T1_energy.png) |
| **T2_a** | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. | 33 | 3.29 s | [`T2_a.wav`](Results/Telugu/test_generated_audios/T2_a.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T2_a_duration.png) \| [Pitch](Results/Telugu/variance_adaptor_plots/pitch/T2_a_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T2_a_energy.png) |
| **T2_b** | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. | 39 | 3.38 s | [`T2_b.wav`](Results/Telugu/test_generated_audios/T2_b.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T2_b_duration.png) \| [Pitch](Results/Telugu/variance_adaptor_plots/pitch/T2_b_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T2_b_energy.png) |
| **T3** | పులి అడవిలో నిశ్శబ్దంగా నడిచింది. | 32 | 2.41 s | [`T3.wav`](Results/Telugu/test_generated_audios/T3.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T3_duration.png) \| [Pitch](Results/Telugu/variance_adaptor_plots/pitch/T3_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T3_energy.png) |
| **T4** | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. | 89 | 8.03 s | [`T4.wav`](Results/Telugu/test_generated_audios/T4.wav) | [Duration](Results/Telugu/variance_adaptor_plots/duration/T4_duration.png) \| [Pitch](Results/Telugu/variance_adaptor_plots/pitch/T4_pitch.png) \| [Energy](Results/Telugu/variance_adaptor_plots/energy/T4_energy.png) |

---

## 6. Quickstart & Reproduction Guide

### 6.1 Git Repository Initialization & Submodule Notice
The nested upstream `.git` folder that originally existed inside `original_fastspeech2/` has already been removed and archived externally, ensuring zero submodule warnings (`warning: adding embedded git repository`) and direct tracking of all 123 `original_fastspeech2/` source files:

```bash
# 1. Initialize the project repository
git init

# 2. Stage all source, config, test, report, and result files
#    (data/, preprocessed_data/, .venv/, and *.pth.tar are excluded via .gitignore)
git add .
git commit -m "feat: complete FastSpeech 2 multilingual implementation for English and Telugu"
```

### 6.2 Environment Setup & Installation
```bash
# Navigate to workspace root
cd FastSpeech2_Project

# Create and activate Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install verified dependencies
pip install -r requirements.txt
```

### 6.3 Dataset Preparation & Preprocessing
Raw audio corpora (`data/`) and extracted feature caches (`preprocessed_data/`) are preserved locally and excluded from Git tracking via `.gitignore`:

1. **English Corpus (`Svarah` Indian English)**:
   - **Raw Data Location**: `data/Svarah/` (22,050 Hz mono PCM waveforms and transcripts).
   - **Stratified Split (`English/metadata/splits/profile_stratified_v1/`)**: **3,208 training utterances** and **664 validation utterances** (3,872 total) stratified across **88 speaker voice profiles** (`English/metadata/speaker_profiles.json`).
   - **Forced Alignment & Feature Extraction**: Montreal Forced Aligner (MFA) using the LibriSpeech ARPAbet lexicon (`original_fastspeech2/lexicon/librispeech-lexicon.txt`) extracts token durations, 80-bin log-mel spectrograms (`n_fft=1024`, `hop_length=256`, `win_length=1024`), PyWorld $F_0$ pitch contours, and frame-level $L_2$ energy into `preprocessed_data/Svarah/` (`15,488` `.npy` feature files + `stats.json` + `speakers.json`):
     ```bash
     .venv/bin/python English/alignment/prepare_full_data.py
     .venv/bin/python English/alignment/extract_durations.py
     ```

2. **Telugu Corpus (`IndicTTS` Telugu)**:
   - **Raw Data Location**: `data/IndicTTS_Telugu/` (resampled from 48 kHz to 22,050 Hz mono PCM).
   - **Speaker Split (`Telugu/metadata/splits/option_1b/`)**: **2,300 training utterances** (**1,053 female** `spk_000`, **1,247 male** `spk_001`) and **421 validation utterances** (**192 female** `spk_000`, **229 male** `spk_001`) — 2,721 total utterances.
   - **Tokenization & Feature Extraction**: Direct alpha-syllabic grapheme tokenization over a 101-token inventory (`Telugu/text/vocabulary.json`) with NFC normalization and Latin-`o` to anusvara (`ం`) repair (`Telugu/text/normalizer.py`). MFA alignment and acoustic preprocessing output 2,721 complete feature sets (`dur`, `mel`, `pitch`, `energy`, `stats.json`, `speakers.json`) into `preprocessed_data/Telugu_Option_1B/`:
     ```bash
     .venv/bin/python Telugu/alignment/align_option_1b.py
     .venv/bin/python Telugu/alignment/preprocess_features.py
     ```

### 6.4 Training the FastSpeech 2 Models
Both models train independently for 20,000 optimizer steps (`batch_size=16`, Adam $\beta=(0.9, 0.98)$, 4,000 warmup steps, validation and checkpointing every 1,000 steps):

```bash
# Train English FastSpeech 2 on Svarah (20,000 steps / 100 epochs; best val loss at Step 19,000, Epoch 95)
.venv/bin/python -m English.train \
  -p config/Svarah/preprocess.yaml \
  -m config/Svarah/model.yaml \
  -t config/Svarah/train_profile_stratified_v1.yaml \
  --preprocessed_dir preprocessed_data/Svarah \
  --split_dir English/metadata/splits/profile_stratified_v1 \
  --checkpoint_dir checkpoints/Svarah_profile_stratified_v1 \
  --log_dir output/logs/Svarah_profile_stratified_v1 \
  --total_step 20000 \
  --save_step 1000 \
  --val_step 1000 \
  --log_step 100 \
  --resume

# Train Telugu FastSpeech 2 on IndicTTS (20,000 steps / 140 epochs; best val loss at Step 13,000, Epoch 91)
.venv/bin/python -m Telugu.train \
  -p config/IndicTTS/preprocess.yaml \
  -m config/IndicTTS/model.yaml \
  -t config/IndicTTS/train.yaml \
  --preprocessed_dir preprocessed_data/Telugu_Option_1B \
  --train_manifest preprocessed_data/Telugu_Option_1B/train.txt \
  --val_manifest preprocessed_data/Telugu_Option_1B/val.txt \
  --checkpoint_dir checkpoints/IndicTTS \
  --log_dir output/log/IndicTTS \
  --total_step 20000 \
  --save_step 1000 \
  --val_step 1000 \
  --log_step 100 \
  --resume
```

### 6.5 Running English Inference
```bash
# Synthesize all canonical assignment samples (E1..E4) and generate variance adaptor plots
.venv/bin/python -m English.inference \
  --mode assignment \
  --speaker_id 0 \
  --checkpoint checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar \
  -p config/Svarah/preprocess.yaml \
  -m config/Svarah/model.yaml \
  -t config/Svarah/train_profile_stratified_v1.yaml \
  --vocoder_weights original_fastspeech2/hifigan/generator_universal.pth.tar \
  --output_dir Results/English/audio

# Synthesize a single custom English sentence with prosodic control
.venv/bin/python -m English.inference \
  --mode single \
  --text "Speech synthesis using FastSpeech 2 is fast and controllable." \
  --speaker_id 0 \
  --checkpoint checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar \
  --duration_control 1.1 \
  --pitch_control 1.05 \
  --output_dir Results/English/audio \
  --output_filename custom_english.wav
```

### 6.6 Running Telugu Inference
```bash
# Synthesize all canonical assignment samples (T1..T4) using the verified manifest
.venv/bin/python -m Telugu.inference \
  --mode assignment \
  --samples_file Results/Telugu/official_samples.json \
  --speaker spk_001 \
  --digit_mode telugu_digits \
  --checkpoint checkpoints/IndicTTS/checkpoint_best.pth.tar \
  -p config/IndicTTS/preprocess.yaml \
  -m config/IndicTTS/model.yaml \
  -t config/IndicTTS/train.yaml \
  --vocoder_weights original_fastspeech2/hifigan/generator_universal.pth.tar \
  --output_dir Results/Telugu/audio \
  --plots_dir Results/Telugu/plots \
  --metadata_dir Results/Telugu/metadata

# Synthesize a single Telugu sentence
.venv/bin/python -m Telugu.inference \
  --mode single \
  --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" \
  --speaker spk_001 \
  --checkpoint checkpoints/IndicTTS/checkpoint_best.pth.tar \
  --output_dir Results/Telugu/audio \
  --output_filename custom_telugu.wav
```

### 6.7 Running the Regression Test Suite
```bash
# Execute the complete 252-test regression suite across all 23 test modules
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v

# Or run individual language inference & training verification suites
.venv/bin/python -m unittest tests/test_telugu_inference.py -v
.venv/bin/python -m unittest tests/test_english_training_audit.py -v
```

---

## 7. Experimental Highlights & Findings

1. **Validation Convergence & Checkpoint Discovery**:
   - Both models converged rapidly within the first 5,000 steps.
   - English reached its lowest validation loss at **Step 19,000** (2.3489).
   - Telugu reached its lowest validation loss at **Step 13,000** (2.0869). After Step 13,000, Telugu pitch loss drifted upward from 0.7481 to 0.7688, confirming that stopping at Step 13,000 prevented pitch over-smoothing.
   - *Methodological note*: Absolute validation losses cannot be directly compared between languages due to differences in token inventories and acoustic distributions.
2. **Contextual Word Analysis (Homographs)**:
   - **English E2(a) vs E2(b) (*"conduct"*)**: Both sentences received identical phonemic transcriptions (`['K', 'AA1', 'N', 'D', 'AH0', 'K', 'T']`). The model exhibited minor prosodic modulation ($42$ vs $46$ frames, pitch peak $1.881$ vs $1.423$) due to local Transformer self-attention, but did not invert stress to /kənˈdʌkt/.
   - **Telugu T2(a) vs T2(b) (*"పట్టు"*)**: Both sentences allocated **identically 20 mel frames ($0.232\text{ s}$)** to the word "పట్టు" with matching geminate hold-and-release durations.
   - **Scientific Conclusion**: These experiments **do not demonstrate semantic disambiguation of contextual meanings**; acoustic variance is driven by positional self-attention and phrasal position.
3. **Subjective AI-Assisted Evaluation (Telugu)**:
   - Clarity: 3.30 / 5.0
   - Fluency: 3.00 / 5.0
   - Naturalness: 2.50 / 5.0 (Overall Mean: 2.93 / 5.0)
   - *Identified limitations*: Dense temporal numbers in T4 create perceived rushing; technical compound in T1 exhibits a somewhat mechanical cadence.

---

## 8. Deliverables & Documentation Index

- **Full Academic Technical Report (PDF)**: [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf) (28 pages, 38 figures, 9 tables)
- **Technical Report Source (Markdown)**: [`Report/Technical_Report.md`](Report/Technical_Report.md)
- **Artifact Manifest**: [`documentation/FINAL_ARTIFACT_MANIFEST.md`](documentation/FINAL_ARTIFACT_MANIFEST.md)
- **Machine-Readable Manifest**: [`documentation/final_artifact_manifest.json`](documentation/final_artifact_manifest.json)
- **Official Telugu Sentences**: [`Results/Telugu/official_samples.json`](Results/Telugu/official_samples.json)

---

## 9. References

1. **Ren, Y., et al.** (2020). *FastSpeech 2: Fast and high-quality end-to-end text to speech*. arXiv:2006.04558.
2. **Kong, J., et al.** (2020). *HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis*. NeurIPS 2020.
3. **Vaswani, A., et al.** (2017). *Attention is all you need*. NeurIPS 2017.
4. **Baby, A., et al.** (2016). *Resources for Indian languages: IndicTTS database for text-to-speech synthesis*. In Proc. DravidianLangTech.
5. **McAuliffe, M., et al.** (2017). *Montreal Forced Aligner: Trainable text-speech alignment using Kaldi*. Interspeech 2017.
