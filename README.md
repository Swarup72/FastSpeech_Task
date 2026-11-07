# FastSpeech 2: English & Telugu Speech Synthesis

A PyTorch implementation of FastSpeech 2 for non-autoregressive text-to-speech synthesis in Indian English (Svarah corpus) and Telugu (IndicTTS corpus), developed for the Speech Signal Processing assignment at IIIT Hyderabad.

The repository includes two independently trained acoustic models, variance adaptors for duration, pitch, and energy prediction, multi-speaker conditioning, and HiFi-GAN neural vocoder integration.

---

## Pretrained Weights & Checkpoints

Model checkpoints and the neural vocoder weights are hosted on Google Drive:

**[Google Drive Folder: Checkpoints & Weights](https://drive.google.com/drive/folders/1TH6ywWdo3rTZyhHNDSlQQhLoEgS2-ubL?usp=sharing)**

To run inference or resume training, download the files and place them in the following paths:

```bash
mkdir -p checkpoints/Svarah_profile_stratified_v1 checkpoints/IndicTTS checkpoints/hifigan

# 1. English Best Checkpoint (Step 19,000, 399.08 MiB)
# Save to: checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar
# SHA-256: 794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e

# 2. Telugu Best Checkpoint (Step 13,000, 397.99 MiB)
# Save to: checkpoints/IndicTTS/checkpoint_best.pth.tar
# SHA-256: 0a6c443a7baea134f2c9c619d603ddcec8c4d1f86fdfd306680931f4a4cb55fd

# 3. HiFi-GAN Universal Vocoder (53.20 MiB)
# Save to: checkpoints/hifigan/generator_universal.pth.tar
# (or legacy path: original_fastspeech2/hifigan/generator_universal.pth.tar)
# SHA-256: 771eaf48adb7bcbeab2fa4ecfb1abff3eb6fae99efde5dc1ffba3fb3ec6aa2e6
```

Verify checksums with:
```bash
shasum -a 256 checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar
shasum -a 256 checkpoints/IndicTTS/checkpoint_best.pth.tar
shasum -a 256 checkpoints/hifigan/generator_universal.pth.tar
```

---

## Model Summary

| Property | English Model | Telugu Model |
| :--- | :--- | :--- |
| **Dataset** | Svarah Indian English corpus | IndicTTS Telugu corpus |
| **Utterances** | 3,208 train / 664 val (3,872 total) | 2,300 train / 421 val (2,721 total) |
| **Speakers** | 88 voice profiles | 2 speakers (`spk_000` female, `spk_001` male) |
| **Tokenization** | ARPAbet phonemes via CMUdict | Direct phonetic graphemes (101 tokens) |
| **Text Normalization** | Numeral/abbreviation expansion | Unicode NFC + Latin-*o* anusvara (`o` → `ం`) repair |
| **Alignments** | Montreal Forced Aligner (MFA) | Montreal Forced Aligner (Option 1B) |
| **Best Step** | **Step 19,000** (Epoch 95) | **Step 13,000** (Epoch 91) |
| **Best Val Loss** | **2.3489** | **2.0869** |
| **Acoustic Features** | 80-bin mel, continuous F0, energy | 80-bin mel, continuous F0, energy |
| **Vocoder** | HiFi-GAN Universal v1 (22,050 Hz) | HiFi-GAN Universal v1 (22,050 Hz) |

---

## Repository Structure

```text
FastSpeech2_Project/
├── English/                          # English training, inference & data scripts
│   ├── train.py                      # Training loop with validation & checkpointing
│   ├── inference.py                  # English synthesis CLI
│   ├── monitoring.py                 # Loss logging & plotting
│   ├── alignment/                    # Svarah alignment & feature extraction
│   ├── metadata/                     # Profile-stratified train/val splits
│   ├── lexicon/                      # LibriSpeech pronunciation dictionary
│   └── text/                         # Text normalization, G2P & CMUdict expansion
├── Telugu/                           # Telugu training, inference & data scripts
│   ├── train.py                      # Training loop with validation & checkpointing
│   ├── inference.py                  # Telugu synthesis CLI & plot generator
│   ├── alignment/                    # IndicTTS alignment & feature extraction
│   ├── metadata/                     # Option 1B splits & vocabulary inventory
│   └── text/                         # NFC normalizer, tokenizer & 101-token vocab
├── common/                           # Shared model architecture & utilities
│   ├── paths.py                      # Centralized project root & path management
│   ├── dataset.py                    # Multi-speaker FastSpeech 2 Dataset loader
│   ├── vocoder/                      # HiFi-GAN vocoder implementation & config
│   ├── model/                        # FastSpeech 2, Variance Adaptor, PostNet, Loss
│   ├── transformer/                  # FFT blocks, Multi-Head Attention
│   ├── training/                     # CheckpointManager with SHA-256 verification
│   └── utils/                        # Mel, mask, device, and plotting utilities
├── config/                           # YAML configurations
│   ├── Svarah/                       # Preprocess, model, and training YAMLs
│   ├── IndicTTS/                     # Preprocess, model, and training YAMLs
│   └── hifigan.json                  # HiFi-GAN neural vocoder configuration
├── checkpoints/                      # Local model weights (excluded from Git)
│   ├── Svarah_profile_stratified_v1/ # English checkpoints (Step 19,000 best)
│   ├── IndicTTS/                     # Telugu checkpoints (Step 13,000 best)
│   └── hifigan/                      # HiFi-GAN universal generator weights
├── Results/                          # Canonical assignment outputs
│   ├── English/
│   │   ├── test_generated_audios/    # E1.wav, E2a.wav, E2b.wav, E3.wav, E4.wav
│   │   ├── variance_adaptor_plots/   # 15 plots across duration/, pitch/, energy/
│   │   ├── training_loss.png         # English training loss curve
│   │   └── validation_loss.png       # English validation loss curve
│   └── Telugu/
│       ├── test_generated_audios/    # T1.wav, T2_a.wav, T2_b.wav, T3.wav, T4.wav
│       ├── variance_adaptor_plots/   # 15 plots across duration/, pitch/, energy/
│       ├── official_samples.json     # Canonical assignment sentences
│       ├── training_loss.png         # Telugu training loss curve
│       └── validation_loss.png       # Telugu validation loss curve
├── Report/
│   ├── Technical_Report.pdf          # 28-page research report
│   ├── Technical_Report.md           # Markdown report source
│   └── figures/                      # Architecture SVGs & result figures
├── documentation/
│   ├── FINAL_ARTIFACT_MANIFEST.md    # Verified deliverable inventory & checksums
│   └── final_artifact_manifest.json  # Machine-readable deliverable manifest
├── research/
│   └── Telugu_Dictionary_g2g.txt     # Grapheme pronunciation dictionary
├── tests/                            # 252 unit and regression tests
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.10
- PyTorch 2.1+ (CUDA recommended for training; CPU / Apple Silicon MPS supported for inference)
- ffmpeg / libsndfile (for audio I/O)

### Setup
```bash
# Clone the repository
git clone https://github.com/Swarup72/FastSpeech_Task.git
cd FastSpeech_Task

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Preprocessing & Feature Extraction

### English (Svarah)
Raw waveforms (`data/Svarah/`) are converted to 22,050 Hz mono audio, aligned using MFA with LibriSpeech lexicon, and preprocessed into 80-bin mel spectrograms, pitch contours, and energy values:
```bash
.venv/bin/python English/alignment/prepare_full_data.py
.venv/bin/python English/alignment/extract_durations.py
```

### Telugu (IndicTTS)
Waveforms (`data/IndicTTS_Telugu/`) are resampled to 22,050 Hz. Transcripts are NFC-normalized and tokenized into grapheme sequences. MFA Option 1B alignments and acoustic features are extracted:
```bash
.venv/bin/python Telugu/alignment/align_option_1b.py
.venv/bin/python Telugu/alignment/preprocess_features.py
```

---

## Training

Both models train independently using Adam optimizer with warmup and Scheduled Decay:

```bash
# Train English FastSpeech 2 (20,000 steps, Svarah)
.venv/bin/python -m English.train \
  -p config/Svarah/preprocess.yaml \
  -m config/Svarah/model.yaml \
  -t config/Svarah/train_profile_stratified_v1.yaml \
  --checkpoint_dir checkpoints/Svarah_profile_stratified_v1 \
  --total_step 20000 \
  --save_step 1000 \
  --val_step 1000

# Train Telugu FastSpeech 2 (20,000 steps, IndicTTS)
.venv/bin/python -m Telugu.train \
  -p config/IndicTTS/preprocess.yaml \
  -m config/IndicTTS/model.yaml \
  -t config/IndicTTS/train.yaml \
  --checkpoint_dir checkpoints/IndicTTS \
  --total_step 20000 \
  --save_step 1000 \
  --val_step 1000
```

---

## Inference & Speech Synthesis

### 1. Synthesize Canonical Assignment Samples

**English (E1–E4)**:
```bash
.venv/bin/python -m English.inference \
  --mode assignment \
  --speaker_id 0 \
  --checkpoint checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar \
  --output_dir Results/English/test_generated_audios
```

**Telugu (T1–T4)**:
```bash
.venv/bin/python -m Telugu.inference \
  --mode assignment \
  --samples_file Results/Telugu/official_samples.json \
  --speaker spk_001 \
  --digit_mode telugu_digits \
  --checkpoint checkpoints/IndicTTS/checkpoint_best.pth.tar \
  --output_dir Results/Telugu/test_generated_audios \
  --plots_dir Results/Telugu/variance_adaptor_plots
```

### 2. Synthesize Custom Sentences

**English**:
```bash
.venv/bin/python -m English.inference \
  --mode single \
  --text "Speech synthesis using FastSpeech 2 is fast and controllable." \
  --speaker_id 0 \
  --checkpoint checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar \
  --duration_control 1.0 \
  --pitch_control 1.0 \
  --output_filename custom_english.wav
```

**Telugu**:
```bash
.venv/bin/python -m Telugu.inference \
  --mode single \
  --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" \
  --speaker spk_001 \
  --checkpoint checkpoints/IndicTTS/checkpoint_best.pth.tar \
  --output_filename custom_telugu.wav
```

---

## Canonical Test Samples & Deliverables

All 10 assignment audio files were synthesized at 22,050 Hz 16-bit mono PCM:

### English Samples (Speaker 0, Female Indian English)
- **E1**: *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* → [`Results/English/test_generated_audios/E1.wav`](Results/English/test_generated_audios/E1.wav)
- **E2a**: *The orchestra's conduct during the ceremony was praised by the audience.* → [`Results/English/test_generated_audios/E2a.wav`](Results/English/test_generated_audios/E2a.wav)
- **E2b**: *The researchers will conduct an experiment to evaluate the proposed method.* → [`Results/English/test_generated_audios/E2b.wav`](Results/English/test_generated_audios/E2b.wav)
- **E3**: *The rhyth m of the orchestra changed unexpectedly.* → [`Results/English/test_generated_audios/E3.wav`](Results/English/test_generated_audios/E3.wav)
- **E4**: *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* → [`Results/English/test_generated_audios/E4.wav`](Results/English/test_generated_audios/E4.wav)

### Telugu Samples (Speaker `spk_001`, Male Native Telugu)
- **T1**: న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. → [`Results/Telugu/test_generated_audios/T1.wav`](Results/Telugu/test_generated_audios/T1.wav)
- **T2_a**: తాడుపై అతని పట్టు చాలా బలంగా ఉంది. → [`Results/Telugu/test_generated_audios/T2_a.wav`](Results/Telugu/test_generated_audios/T2_a.wav)
- **T2_b**: ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. → [`Results/Telugu/test_generated_audios/T2_b.wav`](Results/Telugu/test_generated_audios/T2_b.wav)
- **T3**: పులి అడవిలో నిశ్శబ్దంగా నడిచింది. → [`Results/Telugu/test_generated_audios/T3.wav`](Results/Telugu/test_generated_audios/T3.wav)
- **T4**: ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. → [`Results/Telugu/test_generated_audios/T4.wav`](Results/Telugu/test_generated_audios/T4.wav)

Variance adaptor plots for all 10 sentences (duration, pitch, and energy) are available in:
- `Results/English/variance_adaptor_plots/{duration,pitch,energy}/`
- `Results/Telugu/variance_adaptor_plots/{duration,pitch,energy}/`

---

## Technical Report

The complete 28-page research report detailing architecture, phonetic alignment, training dynamics, loss curves, and linguistic evaluations is available in:
- **PDF**: [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf)
- **Markdown Source**: [`Report/Technical_Report.md`](Report/Technical_Report.md)

---

## Testing & Quality Assurance

Run the complete test suite (252 tests across 23 test modules):
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Author & Citation

- **Author**: Swarup Pal (`swaruppal1511@gmail.com`)
- **Institution**: International Institute of Information Technology, Hyderabad (IIIT Hyderabad)
- **Course**: Speech Signal Processing / Advanced Speech Synthesis
