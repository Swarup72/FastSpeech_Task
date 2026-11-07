# FastSpeech 2 Project — Final Repository Structure (`Phase 15`)

**Project Root**: `/Users/swaruppal/FastSpeech2_Project`  
**Post-Cleanup Local Workspace Size**: **12.14 GiB** (`13,034,464,669 bytes` across `110,739 files`)  
**Proposed Git-Tracked Repository Size**: **81.85 MiB** (`85,828,396 bytes` across `530 files`)  
**External Cleanup Backup Location**: `/Users/swaruppal/Downloads/FastSpeech2_Cleanup_Backup/` (`4.75 GiB` / `5,101,021,990 bytes`, SHA-256 verified)

---

## 1. Directory Hierarchy & Git Tracking Map

```text
FastSpeech2_Project/
│
├── English/                                  # [TRACKED IN GIT — 10.32 MiB]
│   ├── alignment/                            # MFA duration extraction, feature preprocessing & pilot scripts
│   ├── metadata/                             # Svarah manifests & profile_stratified_v1 train/val splits
│   ├── text/normalizer.py                    # English text normalization & currency/time expansion
│   ├── inference.py                          # English FastSpeech 2 + HiFi-GAN synthesis & plotting CLI
│   ├── monitoring.py                         # Persistent CSV logger & 5-panel live loss dashboard
│   ├── prepare_colab_bundle.py               # Reproducible Colab archive generator
│   └── train.py                              # English FastSpeech 2 training loop with resume verification
│
├── Telugu/                                   # [TRACKED IN GIT — 21.09 MiB]
│   ├── alignment/                            # IndicTTS resampling, MFA Option 1B alignment & feature extraction
│   ├── metadata/                             # IndicTTS Telugu manifests, vocabulary audit & speaker splits
│   ├── text/
│   │   ├── normalizer.py                     # Telugu NFC normalizer, digit expansion & grapheme tokenizer
│   │   └── vocabulary.json                   # 101-token Telugu phonetic grapheme inventory
│   ├── inference.py                          # Telugu FastSpeech 2 + HiFi-GAN synthesis & plotting CLI
│   └── train.py                              # Telugu FastSpeech 2 training loop
│
├── common/                                   # [TRACKED IN GIT — 0.69 MiB]
│   ├── config/                               # Validated dataclass schemas for preprocess/model/train YAML
│   ├── model/                                # Shared FastSpeech2, VarianceAdaptor, LengthRegulator, PostNet, Loss
│   ├── training/                             # Trainer engine & atomic SHA-256 CheckpointManager
│   ├── transformer/                          # FFT Encoder/Decoder, MultiHeadAttention & PositionwiseFeedForward
│   └── utils/                                # Masking, padding, device resolution & plotting utilities
│
├── original_fastspeech2/                     # [SOURCE CODE TRACKED IN GIT — 25.39 MiB; WEIGHTS/DATA EXCLUDED]
│   ├── hifigan/
│   │   ├── config.json                       # [TRACKED] Universal HiFi-GAN architecture config
│   │   ├── models.py                         # [TRACKED] HiFi-GAN Generator implementation
│   │   └── generator_universal.pth.tar       # [EXCLUDED FROM GIT / RETAINED LOCALLY — 53.20 MiB]
│   ├── lexicon/librispeech-lexicon.txt       # [TRACKED] CMUdict / LibriSpeech phoneme lexicon
│   ├── preprocessed_data/                    # [EXCLUDED FROM GIT / PROTECTED LOCALLY — 51.60 MiB]
│   ├── dataset.py                            # [TRACKED] Dataset & collate implementation
│   ├── text/                                 # [TRACKED] Symbol inventories & English text cleaners
│   └── utils/                                # [TRACKED] Model & audio helper utilities
│
├── config/                                   # [TRACKED IN GIT — < 0.01 MiB]
│   ├── Svarah/                               # English preprocess.yaml, model.yaml, train.yaml
│   └── IndicTTS/                             # Telugu preprocess.yaml, model.yaml, train.yaml
│
├── Results/                                  # [TRACKED IN GIT — 12.68 MiB]
│   ├── English/
│   │   ├── training_loss.png                 # English training loss curve (Steps 100–20,000)
│   │   ├── validation_loss.png               # English validation loss curve (Steps 1,000–20,000)
│   │   ├── test_generated_audios/            # E1.wav, E2a.wav, E2b.wav, E3.wav, E4.wav (22.05 kHz PCM)
│   │   ├── variance_adaptor_plots/           # 15 plots across duration/, pitch/, energy/
│   │   ├── audio/ & plots/ & metadata/       # Dual-compatible CLI output directories & JSON metadata
│   └── Telugu/
│       ├── training_loss.png                 # Telugu training loss curve (Steps 100–20,000)
│       ├── validation_loss.png               # Telugu validation loss curve (Steps 1,000–20,000)
│       ├── official_samples.json             # Canonical Telugu test sentences (verified T1 compound)
│       ├── test_generated_audios/            # T1.wav, T2_a.wav, T2_b.wav, T3.wav, T4.wav (22.05 kHz PCM)
│       ├── variance_adaptor_plots/           # 15 plots across duration/, pitch/, energy/
│       └── Corrected_Assignment/ & metadata/ # Verified Colab outputs & frame-level prosody JSONs
│
├── Report/                                   # [TRACKED IN GIT — 6.90 MiB]
│   ├── Technical_Report.pdf                  # 30-page compiled academic research report (36 figures, 10 tables)
│   ├── Technical_Report.md                   # Complete Markdown source for the technical report
│   └── figures/                              # All 41 high-resolution loss & variance adaptor figures
│
├── output/                                   # [LOGS TRACKED IN GIT — 1.24 MiB; CKPTS EXCLUDED]
│   ├── logs/Svarah_profile_stratified_v1/    # English training_history.csv, validation_history.csv & plots
│   └── log/IndicTTS/                         # Telugu training_history.csv, validation_history.csv & plots
│
├── checkpoints/                              # [EXCLUDED FROM GIT / RETAINED LOCALLY — 1.59 GiB]
│   ├── Svarah_profile_stratified_v1/
│   │   ├── checkpoint_best.pth.tar           # Step 19,000 (Epoch 95, Step/Best Val Loss 2.348900, 399.08 MiB)
│   │   └── checkpoint_latest.pth.tar         # Step 20,000 (Epoch 100, Step Val Loss 2.368902 / Best 2.348900, 399.08 MiB)
│   └── IndicTTS/
│       ├── checkpoint_best.pth.tar           # Step 13,000 (Epoch 91, Step/Best Val Loss 2.086877, 397.99 MiB)
│       └── checkpoint_latest.pth.tar         # Step 20,000 (Epoch 140, Step Val Loss 2.118075 / Best 2.086877, 397.99 MiB)
│
├── data/                                     # [EXCLUDED FROM GIT / STRICTLY PROTECTED LOCALLY — 7.32 GiB]
├── preprocessed_data/                        # [EXCLUDED FROM GIT / STRICTLY PROTECTED LOCALLY — 1.57 GiB]
├── .venv/                                    # [EXCLUDED FROM GIT / RETAINED LOCALLY — 1.06 GiB]
│
├── documentation/                            # [TRACKED IN GIT — 2.65 MiB]
│   ├── FINAL_ARTIFACT_MANIFEST.md            # Cryptographic deliverable checksum table
│   ├── FINAL_CLEANUP_REPORT.md               # Before-and-after cleanup & verification report
│   ├── FINAL_FILE_MANIFEST.json              # Machine-readable 85-file manifest with Git/storage status
│   ├── FINAL_REPOSITORY_STRUCTURE.md         # This document
│   ├── REPOSITORY_CLEANUP_AUDIT.md           # Phase 1–10 read-only audit report
│   ├── CLEANUP_PLAN.md                       # Phase 11 staged cleanup plan
│   └── dry_run_cleanup.py                    # Phase 12 read-only dry-run protection verifier
│
├── research/                                 # [TRACKED IN GIT — 1.40 MiB]
├── tests/                                    # [TRACKED IN GIT — 1.00 MiB (23 unittest modules)]
├── README.md                                 # [TRACKED IN GIT]
├── requirements.txt                          # [TRACKED IN GIT]
└── .gitignore                                # [TRACKED IN GIT]
```

