# FastSpeech 2 Project — Final Project Structure & Storage Specification

**Project Root**: `/Users/swaruppal/FastSpeech2_Project`  
**Central Path Manager**: [`common/paths.py`](../common/paths.py)  
**Verification Date**: 2026-11-07  
**Status**: **Verified & CWD-Independent (`252/252` unit/integration tests passing from both `PROJECT_ROOT` and `/tmp`)**

---

## 1. Complete Canonical Directory Tree

```text
/Users/swaruppal/FastSpeech2_Project/
├── .gitignore                                # Git exclusion rules (excludes data/, preprocessed_data/, checkpoints/, .venv/)
├── README.md                                 # Project overview, architecture, dataset, training & inference guide
├── requirements.txt                          # Verified Python 3.10 dependencies (22 packages)
├── SOURCE_ARCHIVE_MANIFEST.txt               # Protected historical archive listing
├── Telugu_IndicTTS_Features.tar.gz           # Protected Telugu feature archive (466.86 MiB)
├── English_FastSpeech2_Training.ipynb        # English Colab GPU training notebook (19 cells)
├── Telugu_IndicTTS_Colab.ipynb               # Telugu Colab GPU training notebook (25 cells)
│
├── common/                                   # Shared FastSpeech 2 core library (__init__.py: Yes)
│   ├── __init__.py                           # Exports common.paths, common.config, common.model, common.training, etc.
│   ├── paths.py                              # Single source of truth for project root & path resolution
│   ├── audio/                                # Placeholder audio directory (preserved per scope rules)
│   ├── config/                               # Validated YAML configuration schemas (__init__.py: Yes)
│   │   ├── __init__.py
│   │   ├── base.py                           # Dataclass validation base & ConfigValidationError
│   │   ├── loader.py                         # YAML configuration loader & FastSpeech2Config container
│   │   ├── model.py                          # ModelConfig schema (Transformer, VarianceAdaptor, MultiSpeaker)
│   │   ├── preprocess.py                     # PreprocessConfig schema (Audio, STFT, Mel, Pitch, Energy)
│   │   └── training.py                       # TrainConfig schema (Optimizer, Step, Path)
│   ├── model/                                # Core FastSpeech 2 & Variance Adaptor (__init__.py: Yes)
│   │   ├── __init__.py
│   │   ├── conv.py                           # 1D Convolution with channel-last transposition
│   │   ├── fastspeech2.py                    # FastSpeech2 model (resolves speakers.json via common.paths)
│   │   ├── length_regulator.py               # Duration-based hidden state expansion (LengthRegulator)
│   │   ├── loss.py                           # FastSpeech2Loss (mel, postnet, pitch, energy, log-duration MSE/MAE)
│   │   ├── modules.py                        # Backward-compatible facade for VarianceAdaptor components
│   │   ├── optimizer.py                      # ScheduledOptim (Transformer Noam learning rate schedule)
│   │   ├── postnet.py                        # 5-layer convolutional PostNet with tanh residual refinement
│   │   ├── variance_adaptor.py               # VarianceAdaptor (resolves stats.json via common.paths)
│   │   └── variance_predictor.py             # 2-layer 1D Conv + LayerNorm + Dropout prosody predictor
│   ├── modules/                              # Backward-compatible re-export package (__init__.py: Yes)
│   │   └── __init__.py
│   ├── training/                             # Training engine & checkpoint management (__init__.py: Yes)
│   │   ├── __init__.py
│   │   ├── checkpoint.py                     # CheckpointManager (atomic writes, verification, pruning)
│   │   └── engine.py                         # Trainer (train_step, validate, gradient clipping, resume)
│   ├── transformer/                          # Feed-Forward Transformer (FFT) stack (__init__.py: Yes)
│   │   ├── __init__.py
│   │   ├── constants.py                      # PAD_ID = 0
│   │   ├── layers.py                         # FFTBlock (MultiHeadAttention + PositionwiseFeedForward)
│   │   ├── models.py                         # Encoder, Decoder, get_sinusoid_encoding_table
│   │   ├── modules.py                        # ScaledDotProductAttention
│   │   └── sublayers.py                      # MultiHeadAttention, PositionwiseFeedForward
│   └── utils/                                # Device, masking, padding & plotting helpers (__init__.py: Yes)
│       ├── __init__.py
│       ├── device.py                         # Automatic CUDA / MPS / CPU device selection
│       ├── mask.py                           # Sequence & attention mask generators
│       ├── pad.py                            # 1D and 2D batch padding utilities
│       └── tools.py                          # to_device, expand, plot_mel
│
├── English/                                  # English (AI4Bharat Svarah) pipeline (__init__.py: Yes)
│   ├── __init__.py
│   ├── train.py                              # English FastSpeech 2 training CLI & Colab runner
│   ├── inference.py                          # English FastSpeech 2 synthesis & WAV verification CLI
│   ├── monitoring.py                         # PersistentLogger, LiveDashboard, SystemMonitor & smoke tests
│   ├── prepare_colab_bundle.py               # Colab deployment bundle packager
│   ├── alignment/                            # MFA forced alignment & acoustic feature extraction (__init__.py: Yes)
│   │   ├── __init__.py
│   │   ├── duration.py                       # MFA TextGrid duration extraction & mel-frame reconciliation
│   │   ├── prepare_full_data.py              # Svarah Parquet audio/transcript extraction
│   │   ├── prepare_pilot_data.py             # 100-utterance pilot corpus preparation
│   │   └── preprocess_features.py            # Mel, PyWorld F0, STFT energy & duration serialization
│   ├── metadata/                             # Svarah manifests & profile_stratified_v1 splits (__init__.py: Yes)
│   │   ├── __init__.py
│   │   ├── generate_profile_stratified_v1_split.py
│   │   ├── full_manifest.json, speaker_profiles.json, train.txt, val.txt, test.txt
│   │   └── splits/profile_stratified_v1/     # Active 88-profile stratified split (3,208 train / 664 val)
│   └── text/                                 # English text normalization (__init__.py: Yes)
│       ├── __init__.py
│       └── normalizer.py                     # Auditable date, time, currency, initialism & number expansion
│
├── Telugu/                                   # Telugu (SPRINGLab IndicTTS) pipeline (__init__.py: Yes)
│   ├── __init__.py
│   ├── train.py                              # Telugu FastSpeech 2 training CLI & Colab runner
│   ├── inference.py                          # Telugu FastSpeech 2 synthesis, inverse-norm & plotting CLI
│   ├── alignment/                            # IndicTTS resampling, MFA Option 1B & features (__init__.py: Yes)
│   │   ├── __init__.py
│   │   ├── align_option_1b.py                # Batch IndicMFA forced alignment with beam retry
│   │   ├── preprocess_features.py            # Option 1B Mel, F0, energy & duration feature extraction
│   │   ├── resample.py                       # Band-limited Kaiser sinc resampling (16k/48k -> 22.05kHz)
│   │   └── segmentation_pilot.py             # Narrative pause vs punctuation segmentation study
│   ├── metadata/                             # IndicTTS manifests & Option 1B split generators (__init__.py: Yes)
│   │   ├── __init__.py
│   │   ├── build_manifest.py
│   │   ├── generate_option_1b_split.py
│   │   ├── partition_dataset.py
│   │   ├── text_manifest.json, train.txt, val.txt, test.txt, split_summary.json
│   │   └── splits/option_1b/                 # Active Option 1B split (2,300 train / 421 val / 422 test)
│   └── text/                                 # Telugu NFC normalizer & 101-token vocabulary (__init__.py: Yes)
│       ├── __init__.py
│       ├── normalizer.py                     # Unicode NFC + context-sensitive Latin-o -> Anusvara fix
│       ├── symbols.py                        # 101-token Telugu grapheme + @sp inventory
│       └── vocabulary.json
│
├── original_fastspeech2/                     # Reference implementation & HiFi-GAN vocoder (__init__.py: Yes)
│   ├── __init__.py
│   ├── dataset.py                            # Bilingual Dataset & TextDataset (resolves paths via common.paths)
│   ├── evaluate.py, prepare_align.py, preprocess.py, synthesize.py, train.py
│   ├── audio/                                # Reference STFT & audio processing (__init__.py: Yes)
│   ├── hifigan/                              # Universal HiFi-GAN vocoder (__init__.py: Yes, Protected)
│   │   ├── __init__.py, models.py, env.py, config.json
│   │   └── generator_universal.pth.tar       # Pretrained universal vocoder weights (53.20 MiB)
│   ├── lexicon/                              # Pronunciation lexicons
│   │   └── librispeech-lexicon.txt           # English word -> ARPAbet pronunciation dictionary
│   ├── model/                                # Reference FastSpeech 2 loss/modules (__init__.py: Yes)
│   ├── preprocessor/                         # Reference corpus preprocessors (__init__.py: Yes)
│   ├── preprocessed_data/                    # Protected reference stats/speakers JSONs
│   ├── text/                                 # English cleaners, symbols & CMUdict (__init__.py: Yes)
│   ├── transformer/                          # Reference Transformer blocks (__init__.py: Yes)
│   └── utils/                                # Reference tools & model loaders (__init__.py: Yes)
│
├── config/                                   # Language YAML configurations
│   ├── Svarah/                               # English configurations
│   │   ├── preprocess.yaml
│   │   ├── model.yaml
│   │   ├── train.yaml
│   │   └── train_profile_stratified_v1.yaml  # Active English training configuration
│   └── IndicTTS/                             # Telugu configurations
│       ├── preprocess.yaml
│       ├── model.yaml
│       └── train.yaml                        # Active Telugu training configuration
│
├── checkpoints/                              # Authoritative trained checkpoints (Protected, 1.56 GiB)
│   ├── Svarah_profile_stratified_v1/
│   │   ├── checkpoint_best.pth.tar           # English Best (Step 19,000, Epoch 95, val_loss=2.348900)
│   │   └── checkpoint_latest.pth.tar         # English Latest (Step 20,000, Epoch 100, val_loss=2.357851)
│   └── IndicTTS/
│       ├── checkpoint_best.pth.tar           # Telugu Best (Step 13,000, Epoch 91, val_loss=2.086877)
│       └── checkpoint_latest.pth.tar         # Telugu Latest (Step 20,000, Epoch 140, val_loss=2.118075)
│
├── data/                                     # Raw audio corpora & MFA TextGrids (Protected, 7.32 GiB)
├── preprocessed_data/                        # Extracted .npy feature caches (Protected, 1.57 GiB)
│   ├── Svarah/                               # English features (15,488 .npy files + stats.json + speakers.json)
│   ├── IndicTTS/                             # Telugu features & stats.json / speakers.json
│   └── Telugu_Option_1B/                     # Telugu Option 1B features (2,721 utterances)
│
├── Results/                                  # Canonical assignment outputs (Protected)
│   ├── English/
│   │   ├── test_generated_audios/            # E1.wav, E2a.wav, E2b.wav, E3.wav, E4.wav
│   │   ├── variance_adaptor_plots/           # duration/, pitch/, energy/ (15 PNG plots)
│   │   ├── audio/, plots/, metadata/
│   │   ├── training_loss_curve.png, validation_loss_curve.png
│   │   └── english_prosody_metadata.json, english_samples_metadata.json
│   └── Telugu/
│       ├── test_generated_audios/            # T1.wav, T2_a.wav, T2_b.wav, T3.wav, T4.wav
│       ├── variance_adaptor_plots/           # duration/, pitch/, energy/ (15 PNG plots)
│       ├── Corrected_Assignment/             # Verified Colab assignment bundle
│       ├── audio/, plots/, metadata/, official_samples.json
│       └── training_loss_curve.png, validation_loss_curve.png
│
├── output/                                   # Training/validation CSV histories & loss dashboards (Protected)
│   ├── logs/Svarah_profile_stratified_v1/    # English training_history.csv, validation_history.csv, plots/
│   └── log/IndicTTS/                         # Telugu training_history.csv, validation_history.csv, plots/
│
├── Report/                                   # Technical Report & Figures (Protected)
│   ├── Technical_Report.md
│   ├── Technical_Report.pdf
│   ├── figures/                              # 41 embedded report figures
│   └── tables/
│
├── documentation/                            # Architecture, manifest, cleanup & refactoring reports
│   ├── PROJECT_ARCHITECTURE_AUDIT.md
│   ├── FINAL_PROJECT_STRUCTURE.md
│   ├── PATH_DEPENDENCY_MAP.md
│   ├── REFACTOR_CHANGELOG.md
│   ├── GITHUB_READINESS_REPORT.md
│   ├── FINAL_ARTIFACT_MANIFEST.md
│   ├── final_artifact_manifest.json
│   ├── dry_run_cleanup.py
│   └── English_Training_Evidence/, Telugu_Training_Evidence/
│
├── research/                                 # 45 Markdown research reports + 11 pilot/audit .py scripts + 9 .json files
├── scratch/                                  # 4 build/report generation utilities (retained in scratch/ per Condition 4)
│   ├── build_colab_notebook.py
│   ├── build_manifest.py
│   ├── compile_pdf.py
│   └── generate_technical_report.py
└── tests/                                    # 23 unittest modules (252 tests) (__init__.py: Yes)
    ├── __init__.py
    ├── test_paths.py                         # Unit tests for common/paths.py & CWD independence (8 tests)
    └── test_*.py                             # 22 regression test suites (244 tests)
```

---

## 2. Canonical Storage Locations by Asset Category

| Asset Category | Canonical Directory | Helper in [`common/paths.py`](../common/paths.py) | Protection / Git Policy |
| :--- | :--- | :--- | :--- |
| **Central Path Manager** | `common/paths.py` | `get_project_root()`, `resolve_project_path()` | Tracked in Git |
| **Shared Model / Training Library** | `common/` | `COMMON_DIR` | Tracked in Git |
| **Language YAML Configurations** | `config/Svarah/`, `config/IndicTTS/` | `get_config_dir(lang)`, `get_config_path(lang, kind)` | Tracked in Git |
| **English Pipeline & Scripts** | `English/` | `ENGLISH_DIR` | Tracked in Git |
| **Telugu Pipeline & Scripts** | `Telugu/` | `TELUGU_DIR` | Tracked in Git |
| **Reference Modules & Lexicon** | `original_fastspeech2/` | `ORIGINAL_FS2_DIR`, `LEXICON_PATH` | Code & lexicon tracked in Git |
| **HiFi-GAN Vocoder Weights** | `original_fastspeech2/hifigan/` | `HIFIGAN_DIR`, `HIFIGAN_CONFIG_PATH`, `HIFIGAN_WEIGHTS_PATH` | `.pth.tar` excluded from Git; `config.json` tracked |
| **Raw Audio & MFA TextGrids** | `data/` | `DATA_DIR`, `get_data_dir(lang)` | Protected locally; excluded from Git |
| **Preprocessed `.npy` Features** | `preprocessed_data/` | `PREPROCESSED_DIR`, `get_preprocessed_dir(lang)` | Protected locally; `stats.json` & `speakers.json` tracked |
| **Trained Model Checkpoints** | `checkpoints/` | `CHECKPOINTS_DIR`, `get_checkpoint_dir(lang)`, `get_checkpoint_path(lang)` | Protected locally; excluded from Git |
| **Generated Audio & Plots** | `Results/English/`, `Results/Telugu/` | `RESULTS_DIR`, `get_result_dir(lang)` | Official deliverables tracked in Git |
| **Training Logs & CSV Histories** | `output/` | `OUTPUT_DIR` | CSVs, summaries & loss plots tracked in Git |
| **Technical Report** | `Report/` | `REPORT_DIR`, `REPORT_FIGURES_DIR` | Tracked in Git |
| **Project Documentation** | `documentation/` | `DOCUMENTATION_DIR` | Tracked in Git |
| **Build & Report Scripts** | `scratch/` | `SCRATCH_DIR` | Tracked in Git |
| **Unit & Integration Tests** | `tests/` | `TESTS_DIR` | Tracked in Git |

