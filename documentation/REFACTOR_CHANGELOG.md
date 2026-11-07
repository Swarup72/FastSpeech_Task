# FastSpeech 2 Project — Structure & Path Management Refactoring Changelog

**Project Root**: `/Users/swaruppal/FastSpeech2_Project`  
**Refactoring Date**: 2026-11-07  
**Pre-Refactor Backup**: `/Users/swaruppal/Downloads/FastSpeech2_Cleanup_Backup/pre_refactor_backup_20261107/` (`284` files, SHA-256 verified)

---

## 1. Summary of Changes

1. **Created Central Path Manager ([`common/paths.py`](../common/paths.py))**:
   - Implemented marker-based `get_project_root()` (with `FASTSPEECH2_PROJECT_ROOT` environment override and Google Colab fallbacks) with zero dependency on `os.getcwd()` or `Path.cwd()`.
   - Exported canonical `pathlib.Path` constants (`PROJECT_ROOT`, `COMMON_DIR`, `ENGLISH_DIR`, `TELUGU_DIR`, `ORIGINAL_FS2_DIR`, `CONFIG_DIR`, `DATA_DIR`, `PREPROCESSED_DIR`, `CHECKPOINTS_DIR`, `HIFIGAN_DIR`, `HIFIGAN_CONFIG_PATH`, `HIFIGAN_WEIGHTS_PATH`, `HIFIGAN_WEIGHTS_ZIP_PATH`, `LEXICON_PATH`, `RESULTS_DIR`, `ENGLISH_RESULTS_DIR`, `TELUGU_RESULTS_DIR`, `OUTPUT_DIR`, `REPORT_DIR`, `REPORT_FIGURES_DIR`, `DOCUMENTATION_DIR`, `RESEARCH_DIR`, `SCRATCH_DIR`, `TESTS_DIR`).
   - Implemented `setup_project_paths()`, `resolve_project_path()`, `validate_required_path()`, `ensure_output_dir()`, `get_temp_dir()`, `normalize_language()`, `get_data_dir()`, `get_preprocessed_dir()`, `get_checkpoint_dir()`, `get_checkpoint_candidate_dirs()`, `get_checkpoint_path()`, `get_result_dir()`, `get_config_dir()`, `get_config_path()`, and `get_config_paths()`.
   - Re-exported `common.paths` in [`common/__init__.py`](../common/__init__.py).

2. **Added Missing Package `__init__.py` Files (10 files)**:
   - `Telugu/__init__.py`
   - `Telugu/text/__init__.py`
   - `Telugu/alignment/__init__.py`
   - `Telugu/metadata/__init__.py`
   - `English/text/__init__.py`
   - `English/metadata/__init__.py`
   - `tests/__init__.py`
   - `original_fastspeech2/__init__.py`
   - `original_fastspeech2/utils/__init__.py`
   - `original_fastspeech2/preprocessor/__init__.py`

3. **Refactored Core Model & Dataset Loaders for CWD Independence**:
   - [`common/model/fastspeech2.py`](../common/model/fastspeech2.py): Resolves relative `preprocess_config.path.preprocessed_path` against `PROJECT_ROOT` via `resolve_project_path()` before loading `speakers.json`.
   - [`common/model/variance_adaptor.py`](../common/model/variance_adaptor.py): Resolves relative `preprocess_config.path.preprocessed_path` against `PROJECT_ROOT` via `resolve_project_path()` before loading `stats.json`.
   - [`original_fastspeech2/dataset.py`](../original_fastspeech2/dataset.py): Resolves relative `preprocessed_path` and manifest `filename` against `PROJECT_ROOT` via `resolve_project_path()` in `Dataset.__init__`, `Dataset.process_meta`, and `TextDataset.__init__`.

4. **Refactored English & Telugu Training, Inference, Alignment & Metadata Scripts**:
   - [`English/inference.py`](../English/inference.py) & [`Telugu/inference.py`](../Telugu/inference.py): Replaced ad-hoc path constants with `common.paths` helpers (`get_checkpoint_candidate_dirs`, `resolve_project_path`, `ensure_output_dir`, `LEXICON_PATH`, `HIFIGAN_*`) while preserving `_PROJECT_ROOT` module aliases and all CLI flags.
   - [`English/train.py`](../English/train.py) & [`Telugu/train.py`](../Telugu/train.py): Resolved YAML config paths, `preprocessed_path`, `ckpt_path`, and `log_path` via `resolve_project_path()` and `ensure_output_dir()`.
   - [`English/prepare_colab_bundle.py`](../English/prepare_colab_bundle.py), [`English/metadata/generate_profile_stratified_v1_split.py`](../English/metadata/generate_profile_stratified_v1_split.py), [`English/alignment/prepare_full_data.py`](../English/alignment/prepare_full_data.py), [`English/alignment/prepare_pilot_data.py`](../English/alignment/prepare_pilot_data.py), [`English/alignment/preprocess_features.py`](../English/alignment/preprocess_features.py): Standardized project root and directory constants via `common.paths`.
   - [`Telugu/alignment/align_option_1b.py`](../Telugu/alignment/align_option_1b.py), [`Telugu/alignment/preprocess_features.py`](../Telugu/alignment/preprocess_features.py), [`Telugu/alignment/segmentation_pilot.py`](../Telugu/alignment/segmentation_pilot.py), [`Telugu/metadata/build_manifest.py`](../Telugu/metadata/build_manifest.py), [`Telugu/metadata/generate_option_1b_split.py`](../Telugu/metadata/generate_option_1b_split.py), [`Telugu/metadata/partition_dataset.py`](../Telugu/metadata/partition_dataset.py): Standardized project root and directory constants via `common.paths`.

5. **Eliminated Hardcoded Mac Paths in `scratch/`, `documentation/`, and `research/`**:
   - Kept all 4 utility scripts in `scratch/` (`scratch/build_manifest.py`, `scratch/compile_pdf.py`, `scratch/generate_technical_report.py`, `scratch/build_colab_notebook.py`) per Condition 4 and replaced hardcoded `/Users/swaruppal/FastSpeech2_Project` and CWD-relative output paths with `common.paths`.
   - Updated [`documentation/dry_run_cleanup.py`](../documentation/dry_run_cleanup.py) to use `PROJECT_ROOT` and `DOCUMENTATION_DIR` from `common.paths`.
   - Updated `research/audit_training_and_resume.py`, `research/build_colab_notebook.py`, `research/build_mfa_pilot_selection.py`, `research/test_single_mfa_alignment.py`, and `research/test_telugu_duration_extraction.py` to use `common.paths`.

6. **Added Path Management Unit Tests & Fixed CWD-Dependent Test Suites**:
   - Created [`tests/test_paths.py`](../tests/test_paths.py) (8 unit tests covering `get_project_root()` from `/tmp`, relative vs. absolute path resolution, environment variable overrides, missing path validation errors, output directory creation, English/Telugu config/data/checkpoint resolution, and subprocess execution from `/tmp`).
   - Updated [`tests/test_duration_extraction.py`](../tests/test_duration_extraction.py), [`tests/test_fastspeech2_loss.py`](../tests/test_fastspeech2_loss.py), [`tests/test_telugu_mfa_alignment_pilot.py`](../tests/test_telugu_mfa_alignment_pilot.py), and [`tests/test_telugu_split_vocabulary_audit.py`](../tests/test_telugu_split_vocabulary_audit.py) to resolve project assets via `common.paths` instead of `os.getcwd()`.

---

## 2. Protected Directories & Deliverables Integrity

- **Zero files were moved, renamed, or deleted** in `data/`, `preprocessed_data/`, `checkpoints/`, `original_fastspeech2/hifigan/`, `Results/`, `Report/`, `output/`, or `scratch/`.
- **Zero empty directories were deleted** (preserving `common/audio/`, `documentation/Telugu_Training_Evidence/`, `research/scratch_audio/`, etc.).
- **All 62 mandatory assignment deliverables** in [`documentation/final_artifact_manifest.json`](final_artifact_manifest.json) remain 100% bit-identical (`0` SHA-256 changes).

