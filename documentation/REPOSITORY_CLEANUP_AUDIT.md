# FastSpeech 2 Project — Comprehensive Read-Only Repository Audit (`Phase 1–10`)

**Project Root**: `/Users/swaruppal/FastSpeech2_Project`  
**Platform**: macOS (Apple Silicon M2), Python 3.10.18, PyTorch 2.10.0, `torchaudio` 2.10.0 (MPS Active)  
**Audit Timestamp**: 2026-11-07  
**Audit Mode**: Strictly Read-Only (Zero files modified, moved, or deleted)

---

## 1. Executive Summary & Disk Usage Breakdown

The project currently occupies **16.89 GiB (18,134,726,564 bytes / 18.13 GB decimal)** on disk. A comprehensive SHA-256 and AST dependency inspection confirms that **99.7% of the disk footprint consists of raw datasets, preprocessed feature arrays, model checkpoints, deployment archive bundles, and the local virtual environment**, while the core source code, configurations, unit tests, technical report, and assignment deliverables occupy **56.41 MiB (405 files)**—or **81.80 MiB (527 files)** when tracking `original_fastspeech2/` reference source files directly.

### 1.1 Top-Level Directory & Root File Footprint

| Top-Level Path | Size (Bytes) | Size (MiB / GiB) | File Count | Category | Git Status (`.gitignore`) | Audit Classification |
| :--- | ---: | ---: | ---: | :--- | :---: | :---: |
| `data/` | 7,494,394,812 | 7,147.21 MiB (6.98 GiB) | 12,566 | Raw Datasets (Svarah & IndicTTS) | Ignored | **KEEP (PROTECTED)** |
| `checkpoints/` | 4,321,659,375 | 4,121.45 MiB (4.02 GiB) | 14 | Trained & Intermediate Checkpoints | Ignored | **KEEP BEST/LATEST; REVIEW INTERMEDIATE/MOCK** |
| `preprocessed_data/` | 1,685,455,952 | 1,607.37 MiB (1.57 GiB) | 30,287 | Extracted Mel/Pitch/Energy/Duration `.npy` | Ignored | **KEEP (PROTECTED)** |
| *Root Archive Bundles (`*.tar.gz`, `*.zip`)* | 1,520,350,608 | 1,449.92 MiB (1.42 GiB) | 8 | Colab Upload/Download Archives | Ignored | **BACKUP BEFORE DELETION** |
| `.venv/` | 1,132,899,498 | 1,080.42 MiB (1.06 GiB) | 23,604 | Python 3.10 Virtual Environment | Ignored | **KEEP LOCALLY (EXCLUDE FROM GITHUB)** |
| `output/` | 838,233,701 | 799.40 MiB (0.78 GiB) | 25 | Training Logs (1.24 MiB) + Duplicate Ckpts (798.16 MiB) | `output/ckpt/` Ignored | **KEEP LOGS; DELETE DUPLICATE CKPTS AFTER BACKUP** |
| `original_fastspeech2/` | 717,999,118 | 684.73 MiB (0.67 GiB) | 1,059 | Reference Implementation + `.git` + HiFi-GAN | Partial | **KEEP CODE & VOCODER; ARCHIVE `.git` & ZIPS** |
| `Telugu/` | 22,115,321 | 21.09 MiB | 46 | Telugu FastSpeech 2 Pipeline & Manifests | Tracked | **KEEP & TRACK IN GITHUB** |
| `Results/` | 13,294,102 | 12.68 MiB | 105 | Canonical Assignment Audio, Plots & Metadata | Tracked | **KEEP & TRACK IN GITHUB (PROTECTED)** |
| `English/` | 10,823,418 | 10.32 MiB | 38 | English FastSpeech 2 Pipeline & Splits | Tracked | **KEEP & TRACK IN GITHUB** |
| `Report/` | 7,235,814 | 6.90 MiB | 43 | `Technical_Report.pdf`, `.md` & 41 Figures | Tracked | **KEEP & TRACK IN GITHUB (PROTECTED)** |
| `documentation/` | 2,664,190 | 2.54 MiB | 29 | Manifests, Audits & Training Evidence | Tracked | **KEEP & TRACK IN GITHUB** |
| `research/` | 1,468,912 | 1.40 MiB | 67 | Deep-Dive Architecture Analyses & Pilot Audits | Tracked | **KEEP & TRACK IN GITHUB** |
| `tests/` | 1,048,576 | 1.00 MiB | 24 | Automated Unit & Integration Test Suite | Tracked | **KEEP & TRACK IN GITHUB (PROTECTED)** |
| `common/` | 724,120 | 0.69 MiB | 31 | Shared FastSpeech 2 Architecture & Trainer | Tracked | **KEEP & TRACK IN GITHUB** |
| `--output_path/` | 108,892 | 0.10 MiB | 3 | Accidental CLI Flag Directory (Pilot OOVs) | Ignored | **BACKUP BEFORE DELETION** |
| `scratch/` | 103,491 | 0.10 MiB | 4 | Report PDF Compiler & Manifest Generator Scripts | Ignored | **KEEP LOCALLY (EXCLUDE FROM GITHUB)** |
| `.pytest_cache/` | 17,472 | 0.02 MiB | 6 | Ephemeral Pytest Runner Cache | Ignored | **SAFE TO DELETE** |
| `config/` | 3,915 | < 0.01 MiB | 7 | Svarah & IndicTTS YAML Configurations | Tracked | **KEEP & TRACK IN GITHUB (PROTECTED)** |
| *Root Documentation & Notebooks* | 95,765 | 0.09 MiB | 6 | `README.md`, `requirements.txt`, `.gitignore`, `.ipynb` | Tracked | **KEEP & TRACK IN GITHUB** |

---

## 2. Large File Inventory (`>= 500 MB`, `>= 100 MB`, `>= 10 MB`)

### 2.1 Files Larger Than 500 MiB (4 Files — All Protected Raw Datasets)

| Relative Path | Size (Bytes) | Size (MiB) | SHA-256 Checksum | Purpose | Classification |
| :--- | ---: | ---: | :--- | :--- | :---: |
| `data/telugu_parquet/data/train-00002-of-00004.parquet` | 2,149,814,979 | 2,050.22 | `f42d61adf1efed810f9c318a6bbbf13b97f2024755671b1f2024755671b1f221` | IndicTTS Telugu raw audio/text shard 2 | **KEEP (PROTECTED)** / **EXCLUDE FROM GITHUB** |
| `data/telugu_parquet/data/train-00003-of-00004.parquet` | 1,727,697,226 | 1,647.66 | `a6e5ab6cdd539f0f570beca9126bd4b1e8fb7794b7fef10309da2afe54afb17d` | IndicTTS Telugu raw audio/text shard 3 | **KEEP (PROTECTED)** / **EXCLUDE FROM GITHUB** |
| `data/telugu_parquet/data/train-00001-of-00004.parquet` | 1,363,647,665 | 1,300.48 | `36941881bfb45e0e91c7aa3cdeb16b95d2e2431b870ae7decf59572a4e5c5c1f` | IndicTTS Telugu raw audio/text shard 1 | **KEEP (PROTECTED)** / **EXCLUDE FROM GITHUB** |
| `data/telugu_parquet/data/train-00000-of-00004.parquet` | 1,200,192,373 | 1,144.59 | `eaebe44956d2a70e1e5d533227bdb3c85b1a228775f16055ec0e37aba66590a6` | IndicTTS Telugu raw audio/text shard 0 | **KEEP (PROTECTED)** / **EXCLUDE FROM GITHUB** |

### 2.2 Files Between 100 MiB and 500 MiB (23 Files)

| Relative Path | Size (Bytes) | Size (MiB) | SHA-256 Checksum | Purpose | Classification |
| :--- | ---: | ---: | :--- | :--- | :---: |
| `Telugu_IndicTTS_Features.tar.gz` | 489,537,047 | 466.86 | `f435fc9f2d42de359f1c3a55106f39c3a67a962ea9ba6e320e36d805fdcb0cde` | Colab upload archive of `preprocessed_data/Telugu_Option_1B/` | **BACKUP BEFORE DELETION** |
| `FastSpeech2_Colab_Bundle_profile_stratified_v1.tar.gz` | 486,779,356 | 464.23 | `da1c419444ab29821966e16a12d45781db18db07673ad148247f03d299784869` | Colab upload archive of `preprocessed_data/Svarah/` + code | **BACKUP BEFORE DELETION** |
| `FastSpeech2_Colab_Bundle.tar.gz` | 486,296,901 | 463.77 | `fd4ee9922f6dcaeff281b862f2fd81cb78491ea3da7c5ca8ebd33efad9741913` | Superseded v0 English Colab upload bundle | **BACKUP BEFORE DELETION** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` | 418,467,535 | 399.08 | `6063dd284e32d7ec72b01280fc1254c9a4fc65065e2baaddfe0c31241b81cde3` | English Step 20,000 latest checkpoint (Epoch 100) | **KEEP LOCALLY** / **EXCLUDE FROM GITHUB** |
| `output/ckpt/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` | 418,467,535 | 399.08 | `6063dd284e32d7ec72b01280fc1254c9a4fc65065e2baaddfe0c31241b81cde3` | Exact byte-for-byte duplicate of English latest checkpoint | **BACKUP BEFORE DELETION** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` | 418,465,625 | 399.08 | `794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e` | **Authoritative English Best Checkpoint (Step 19,000)** | **KEEP (PROTECTED)** / **EXTERNAL HOST** |
| `output/ckpt/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` | 418,465,625 | 399.08 | `794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e` | Exact byte-for-byte duplicate of English best checkpoint | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_latest.pth.tar` | 417,324,751 | 397.99 | `8e46b77bedb23c80703ae0309a9c7fa01806e13db3b515ba88e9bb6005f4556f` | Telugu Step 20,000 latest checkpoint (Epoch 140) | **KEEP LOCALLY** / **EXCLUDE FROM GITHUB** |
| `checkpoints/IndicTTS/checkpoint_20000.pth.tar` | 417,323,860 | 397.99 | `c4da76b31b83c7d20b7b005305d0c84af35239dc6089cce8211a49e748355e72` | Telugu Step 20,000 periodic checkpoint (redundant with `latest`) | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_19000.pth.tar` | 417,321,364 | 397.99 | `7081bb5d653d3344cc2004476d01d8a07fe421c28371ef677be25fe593d81a2b` | Telugu Step 19,000 intermediate checkpoint (Epoch 133) | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_18000.pth.tar` | 417,321,236 | 397.99 | `3c16341856104dea05289c49a1c0d146f976c3af7b777cda7ae061c8888f2172` | Telugu Step 18,000 intermediate checkpoint (Epoch 126) | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_17000.pth.tar` | 417,321,108 | 397.99 | `933278f7e5bf45b7304a0166460822bf81b549f50bd4d8d45eafb1ea3b7d8fb3` | Telugu Step 17,000 intermediate checkpoint (Epoch 119) | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_16000.pth.tar` | 417,320,980 | 397.99 | `00e3de02fcc3ebfcd710338c4aa7c42bea417def8e4b1def3760eaf81d0d5783` | Telugu Step 16,000 intermediate checkpoint (Epoch 112) | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_best.pth.tar` | 417,319,705 | 397.99 | `0a6c443a7baea134f2c9c619d603ddcec8c4d1f86fdfd306680931f4a4cb55fd` | **Authoritative Telugu Best Checkpoint (Step 13,000)** | **KEEP (PROTECTED)** / **EXTERNAL HOST** |
| `original_fastspeech2/.git/objects/pack/tmp_pack_LAa061` | 331,603,967 | 316.24 | `eb9ed98f52523c60f3961a24e8acc3ac8eaad6ff7066d26a4f630504371908e3` | Stale temporary Git pack inside nested upstream `.git` | **BACKUP BEFORE DELETION** |
| `data/telugu_raw/train-00000-of-00004.parquet` | 240,132,096 | 229.01 | `a3c25f55dec1580c09219dd9d98babd75541de4f6b0696b76a753f2df85ee052` | Raw Telugu pilot Parquet shard | **KEEP (PROTECTED)** / **EXCLUDE FROM GITHUB** |
| `.venv/lib/python3.10/site-packages/torch/lib/libtorch_cpu.dylib` | 214,082,912 | 204.17 | `941f3e16a8e02b23b99bd2c8693caabaf8df859bfaa293f39a5d7dd8a8f5ca6e` | PyTorch C++ shared library inside `.venv/` | **KEEP LOCALLY** / **EXCLUDE FROM GITHUB** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar.mock_bak` | 140,864,483 | 134.34 | `6d9df4b4d63fe3659a4125f34000bfb04ce991bb57a155f722969259ba2b493a` | Obsolete Epoch 1 pilot checkpoint backup | **BACKUP BEFORE DELETION** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_5000.pth.tar` | 140,863,999 | 134.34 | `3babc8451d5cc9c04f314e1ae8260bf5c08abd3759578490a71c48604815c786` | Obsolete Epoch 1 pilot checkpoint (Val Loss 3.9105) | **BACKUP BEFORE DELETION** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar.mock_bak` | 140,863,999 | 134.34 | `24a4895f85783092856aa219fd24579e37ef6f4eaeb01fd9f7c520488239b463` | Obsolete Epoch 1 pilot checkpoint backup | **BACKUP BEFORE DELETION** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_best_diagnostic_mock.pth.tar.bak` | 140,863,999 | 134.34 | `24a4895f85783092856aa219fd24579e37ef6f4eaeb01fd9f7c520488239b463` | Byte-identical duplicate of `.mock_bak` above | **BACKUP BEFORE DELETION** |
| `.venv/lib/python3.10/site-packages/llvmlite/binding/libllvmlite.dylib` | 129,871,888 | 123.86 | `0dbd1ffab43f10f8ca3e27d10739bcab902d0bfddcb2c3762e1ea37dc5d85a5e` | Numba LLVM shared library inside `.venv/` | **KEEP LOCALLY** / **EXCLUDE FROM GITHUB** |
| `original_fastspeech2/.git/objects/pack/pack-ed04393442675b02e114b9afdbc7d9b28187e129.pack` | 123,553,238 | 117.83 | `8bd1e5c679a220731fa11157df99e7d931a14517c6a987111d796012c43c6ce2` | Upstream Git pack file inside nested `.git` | **BACKUP BEFORE DELETION** |

### 2.3 Files Between 10 MiB and 100 MiB (13 Files)

| Relative Path | Size (Bytes) | Size (MiB) | SHA-256 Checksum | Purpose | Classification |
| :--- | ---: | ---: | :--- | :--- | :---: |
| `data/telugu_mfa_model/Telugu_Acoustic_Model.zip` | 61,448,474 | 58.60 | `b6b1ec92f28cdc705f84ce37a1577b995c9b9f1eed6f154acfb5fc3146ad4c43` | Montreal Forced Aligner Telugu acoustic model | **KEEP (PROTECTED)** / **EXCLUDE FROM GITHUB** |
| `original_fastspeech2/hifigan/generator_universal.pth.tar` | 55,788,858 | 53.20 | `771eaf4876485a35e25577563d390c262e23c2421e4a8c929eacfde34a5b7a60` | **Active Universal HiFi-GAN Vocoder Weights** | **KEEP (PROTECTED)** / **EXCLUDE FROM GITHUB** |
| `English_inference_full_assets.zip` | 53,349,916 | 50.88 | `d2809b9b811cdea33c89131f326d0a748abb496d3dca63d5cdda0c62e7d25666` | Downloaded Colab inference bundle (unpacked in `Results/English/`) | **BACKUP BEFORE DELETION** |
| `original_fastspeech2/hifigan/generator_universal.pth.tar.zip` | 51,942,859 | 49.54 | `1fd93a04431b14296cf1d3f66a6fea89c2fdeb8f3b900a962e13ad21f67f9149` | Compressed duplicate of `generator_universal.pth.tar` | **BACKUP BEFORE DELETION** |
| `original_fastspeech2/hifigan/generator_LJSpeech.pth.tar.zip` | 51,838,074 | 49.44 | `ff0b41168d347edd43d7d6859d136221c1fd9ea4698cf8998d5f0228139f2d41` | Unused LJSpeech HiFi-GAN vocoder zip from upstream repo | **BACKUP BEFORE DELETION** |
| `original_fastspeech2/preprocessed_data/LibriTTS/train.txt` | 36,329,986 | 34.65 | `563991e1ad8a864b664ef318cd20915c6c714ba309430bb8c9041beebebd4cb7` | Upstream reference preprocessed manifest | **KEEP (PROTECTED)** / **EXCLUDE FROM GITHUB** |
| `original_fastspeech2/.git/objects/pack/pack-d833ae61de747dca45299d80f42bc83a25ceab21.pack` | 16,095,832 | 15.35 | `706f901a38d2a9d16b02697388a5b9a3290e4471b0c9b1177312f388d93192dc` | Upstream Git pack file inside nested `.git` | **BACKUP BEFORE DELETION** |
| `original_fastspeech2/preprocessed_data/AISHELL3/train.txt` | 13,286,353 | 12.67 | `97e54ba377858f36779b2747466dc9f048de140648ab65ecf86c490c5c2ee335` | Upstream reference preprocessed manifest | **KEEP (PROTECTED)** / **EXCLUDE FROM GITHUB** |
| `.venv/lib/python3.10/site-packages/...` (5 shared libraries) | 158,557,776 | 151.21 | *(5 `.dylib`/`.so` binaries in `pyarrow`, `duckdb`, `torch`)* | Installed package shared libraries inside `.venv/` | **KEEP LOCALLY** / **EXCLUDE FROM GITHUB** |

---

## 3. Strict Dataset & Preprocessing Protection Audit (`Phase 2`)

In accordance with **Phase 2**, the following directories and files are designated as **Strictly Protected** and will have **zero files deleted, moved, renamed, or modified**:

1. **`data/` (7,494,394,812 bytes / 12,566 files)**:
   - `data/svarah_train/`, `data/svarah_pilot/`, `data/mfa_pilot/`: Original Svarah WAV audio, transcripts, and MFA alignment outputs.
   - `data/telugu_parquet/`, `data/telugu_raw/`: Original IndicTTS Telugu Parquet shards.
   - `data/telugu_mfa_model/`, `data/telugu_mfa_pilot/`, `data/telugu_option1b_mfa/`: Telugu MFA acoustic model, dictionary, and aligned TextGrid files.
2. **`preprocessed_data/` (1,685,455,952 bytes / 30,287 files)**:
   - `preprocessed_data/Svarah/`: Extracted English `.npy` arrays (`mel/`, `pitch/`, `energy/`, `duration/`), `stats.json`, `speakers.json`, `train.txt`, `val.txt`.
   - `preprocessed_data/Telugu_Option_1B/` and `preprocessed_data/IndicTTS/`: Extracted Telugu `.npy` arrays, `stats.json`, `speakers.json`, `train.txt`, `val.txt`.
3. **`original_fastspeech2/preprocessed_data/` (51,996,753 bytes / 29 files)**:
   - Reference `AISHELL3/`, `LJSpeech/`, and `LibriTTS/` statistics and manifests.
4. **Protected Text & Metadata Files (Tracked in GitHub)**:
   - `Telugu/text/vocabulary.json`, `Telugu/text/normalizer.py`, `Telugu/metadata/**`, `Telugu/alignment/**`
   - `English/text/normalizer.py`, `English/metadata/**`, `English/alignment/**`

---

## 4. Source Code & AST Dependency Analysis (`Phase 3`)

A static Abstract Syntax Tree (AST) import scan across all `.py` files in `common/`, `English/`, `Telugu/`, `tests/`, and `original_fastspeech2/` verified all direct and indirect dependencies:

- **`common/`** (`common.config`, `common.model.fastspeech2`, `common.model.variance_adaptor`, `common.model.loss`, `common.model.optimizer`, `common.training.engine`, `common.training.checkpoint`, `common.transformer.*`, `common.utils.*`): Core shared architecture imported across `English/train.py`, `English/inference.py`, `Telugu/train.py`, `Telugu/inference.py`, and 18 unit test suites.
- **`original_fastspeech2/`**:
  - `original_fastspeech2/dataset.py`: Imported by `English/train.py`, `Telugu/train.py`, and `tests/test_telugu_option1b_features.py`.
  - `original_fastspeech2/text/symbols.py` and `original_fastspeech2/text/__init__.py`: Imported by `English/inference.py`, `English/alignment/duration.py`, and `English/monitoring.py`.
  - `original_fastspeech2/hifigan/`: Imported by `English/inference.py` and `Telugu/inference.py` for neural vocoding.
  - `original_fastspeech2/lexicon/librispeech-lexicon.txt`: Loaded by `English/inference.py` (`DEFAULT_LEXICON_PATH`) and `tests/test_english_inference.py`.
- **Conclusion**: **Zero source code files (`.py`), YAML configs (`.yaml`), lexicons (`.txt`), or unit tests (`tests/*.py`) are proposed for removal.**

---

## 5. Complete Checkpoint Audit (`Phase 4`)

Every `.pth.tar`, `.bak`, and `.mock_bak` checkpoint in the workspace was loaded via `torch.load(..., map_location="cpu")` and cryptographically hashed:

| Relative Path | Size (Bytes) | SHA-256 Checksum | Epoch | Global Step | Val Loss at Step | Best Val Loss | Status & Compatibility | Recommendation |
| :--- | ---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` | 418,465,625 | `794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e` | 95 | **19,000** | **2.348900** | 2.348900 | **Authoritative English Best** (235 tensors, 88 speakers) | **KEEP (PROTECTED)** / **EXTERNAL HOST** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` | 418,467,535 | `6063dd284e32d7ec72b01280fc1254c9a4fc65065e2baaddfe0c31241b81cde3` | 100 | 20,000 | 2.368902 | 2.348900 | Authoritative English Latest (Final Step 20k resume state) | **KEEP LOCALLY** / **EXCLUDE FROM GITHUB** |
| `output/ckpt/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` | 418,465,625 | `794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e` | 95 | 19,000 | 2.348900 | 2.348900 | Exact byte-for-byte duplicate of `checkpoints/.../checkpoint_best.pth.tar` | **BACKUP BEFORE DELETION** |
| `output/ckpt/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` | 418,467,535 | `6063dd284e32d7ec72b01280fc1254c9a4fc65065e2baaddfe0c31241b81cde3` | 100 | 20,000 | 2.368902 | 2.348900 | Exact byte-for-byte duplicate of `checkpoints/.../checkpoint_latest.pth.tar` | **BACKUP BEFORE DELETION** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_5000.pth.tar` | 140,863,999 | `3babc8451d5cc9c04f314e1ae8260bf5c08abd3759578490a71c48604815c786` | 1 | 5,000 | 3.910500 | 3.910500 | Superseded pre-Colab pilot checkpoint | **BACKUP BEFORE DELETION** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar.mock_bak` | 140,863,999 | `24a4895f85783092856aa219fd24579e37ef6f4eaeb01fd9f7c520488239b463` | 1 | 5,000 | 3.910500 | 3.910500 | Superseded pre-Colab diagnostic mock backup | **BACKUP BEFORE DELETION** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_best_diagnostic_mock.pth.tar.bak` | 140,863,999 | `24a4895f85783092856aa219fd24579e37ef6f4eaeb01fd9f7c520488239b463` | 1 | 5,000 | 3.910500 | 3.910500 | Byte-identical duplicate of `checkpoint_best.pth.tar.mock_bak` | **BACKUP BEFORE DELETION** |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar.mock_bak` | 140,864,483 | `6d9df4b4d63fe3659a4125f34000bfb04ce991bb57a155f722969259ba2b493a` | 1 | 5,000 | 3.910500 | 3.910500 | Superseded pre-Colab diagnostic mock latest backup | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_best.pth.tar` | 417,319,705 | `0a6c443a7baea134f2c9c619d603ddcec8c4d1f86fdfd306680931f4a4cb55fd` | 91 | **13,000** | **2.086877** | 2.086877 | **Authoritative Telugu Best** (235 tensors, 101 vocab, 2 speakers) | **KEEP (PROTECTED)** / **EXTERNAL HOST** |
| `checkpoints/IndicTTS/checkpoint_latest.pth.tar` | 417,324,751 | `8e46b77bedb23c80703ae0309a9c7fa01806e13db3b515ba88e9bb6005f4556f` | 140 | 20,000 | 2.118075 | 2.086877 | Authoritative Telugu Latest (Final Step 20k resume state) | **KEEP LOCALLY** / **EXCLUDE FROM GITHUB** |
| `checkpoints/IndicTTS/checkpoint_16000.pth.tar` | 417,320,980 | `00e3de02fcc3ebfcd710338c4aa7c42bea417def8e4b1def3760eaf81d0d5783` | 112 | 16,000 | 2.102739 | 2.086877 | Intermediate periodic checkpoint (post-13k pitch drift phase) | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_17000.pth.tar` | 417,321,108 | `933278f7e5bf45b7304a0166460822bf81b549f50bd4d8d45eafb1ea3b7d8fb3` | 119 | 17,000 | 2.094598 | 2.086877 | Intermediate periodic checkpoint | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_18000.pth.tar` | 417,321,236 | `3c16341856104dea05289c49a1c0d146f976c3af7b777cda7ae061c8888f2172` | 126 | 18,000 | 2.107811 | 2.086877 | Intermediate periodic checkpoint | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_19000.pth.tar` | 417,321,364 | `7081bb5d653d3344cc2004476d01d8a07fe421c28371ef677be25fe593d81a2b` | 133 | 19,000 | 2.095243 | 2.086877 | Intermediate periodic checkpoint | **BACKUP BEFORE DELETION** |
| `checkpoints/IndicTTS/checkpoint_20000.pth.tar` | 417,323,860 | `c4da76b31b83c7d20b7b005305d0c84af35239dc6089cce8211a49e748355e72` | 140 | 20,000 | 2.118075 | 2.086877 | Periodic Step 20k copy (redundant with `checkpoint_latest.pth.tar`) | **BACKUP BEFORE DELETION** |

---

## 6. HiFi-GAN Vocoder & Runtime Dependencies (`Phase 5`)

Inspection of `original_fastspeech2/hifigan/`:

| Relative Path | Size (Bytes) | SHA-256 Checksum | Role in Inference | Recommendation |
| :--- | ---: | :--- | :--- | :---: |
| `original_fastspeech2/hifigan/__init__.py` | 179 | `acbdcb18c64a94abe9b7d3d14fb1c4f98f6e0f5ed85e822cb8a24c36f6da79d6` | Exports `Generator` and `AttrDict` | **KEEP & TRACK IN GITHUB** |
| `original_fastspeech2/hifigan/models.py` | 5,515 | `a93fc3e80f5db5d928d94dbd7711d5185fd5d0ca6505b31232a2e442faa38f55` | HiFi-GAN neural architecture | **KEEP & TRACK IN GITHUB** |
| `original_fastspeech2/hifigan/config.json` | 799 | `c1bd98e99062ddbced38729a5252dc2aa772328d16d70097ac139dab2f269dc9` | Universal vocoder hyperparameters (`DEFAULT_HIFIGAN_CONFIG`) | **KEEP & TRACK IN GITHUB** |
| `original_fastspeech2/hifigan/LICENSE` | 1,067 | `5eec017ce442989e01a46cd9cd6aeb13cd0aa3b3943f681577bda00475afaa23` | MIT License for HiFi-GAN code | **KEEP & TRACK IN GITHUB** |
| `original_fastspeech2/hifigan/generator_universal.pth.tar` | 55,788,858 | `771eaf4876485a35e25577563d390c262e23c2421e4a8c929eacfde34a5b7a60` | **Primary unpacked vocoder checkpoint (`DEFAULT_HIFIGAN_WEIGHTS`)** | **KEEP LOCALLY (PROTECTED)** / **EXTERNAL HOST** |
| `original_fastspeech2/hifigan/generator_universal.pth.tar.zip` | 51,942,859 | `1fd93a04431b14296cf1d3f66a6fea89c2fdeb8f3b900a962e13ad21f67f9149` | Compressed archive of `generator_universal.pth.tar` | **BACKUP BEFORE DELETION** |
| `original_fastspeech2/hifigan/generator_LJSpeech.pth.tar.zip` | 51,838,074 | `ff0b41168d347edd43d7d6859d136221c1fd9ea4698cf8998d5f0228139f2d41` | Unused LJSpeech-only vocoder zip | **BACKUP BEFORE DELETION** |

---

## 7. SHA-256 Duplicate Detection Analysis (`Phase 7`)

An exhaustive SHA-256 scan across all non-data files identified **85 duplicate file groups**:

1. **High-Impact Checkpoint Duplicates (932.50 MiB recoverable)**:
   - `output/ckpt/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` == `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` (`418,465,625 B`, SHA-256 `794b5123...`).
   - `output/ckpt/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` == `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` (`418,467,535 B`, SHA-256 `6063dd28...`).
   - `checkpoints/Svarah_profile_stratified_v1/checkpoint_best_diagnostic_mock.pth.tar.bak` == `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar.mock_bak` (`140,863,999 B`, SHA-256 `24a4895f...`).
2. **Intentional Dual-Hierarchy Assignment & Report Copies (~6.4 MiB total — KEEP ALL)**:
   - `Results/English/test_generated_audios/*.wav` & `Results/English/audio/*.wav`
   - `Results/English/variance_adaptor_plots/{duration,pitch,energy}/*.png`, `Results/English/plots/*.png`, & `Report/figures/E*.png`
   - `Results/Telugu/test_generated_audios/*.wav`, `Results/Telugu/audio/*.wav`, & `Results/Telugu/Corrected_Assignment/audio/*.wav`
   - `Results/Telugu/variance_adaptor_plots/{duration,pitch,energy}/*.png`, `Results/Telugu/plots/*.png`, `Results/Telugu/Corrected_Assignment/plots/*.png`, & `Report/figures/T*.png`
   - `documentation/English_Training_Evidence/**` mirrors of `output/logs/Svarah_profile_stratified_v1/**` and `English/metadata/splits/profile_stratified_v1/**`.
   - **Justification for Keeping**: These lightweight copies (~6.4 MiB combined) ensure simultaneous compatibility with (a) the exact folder names mandated in `FastSpeech2_Assignment(3).pdf`, (b) default CLI output paths in `English/inference.py` and `Telugu/inference.py`, (c) internal paths recorded inside `Results/Telugu/metadata/*.json`, and (d) inline Markdown image links in `Report/Technical_Report.md` and `README.md`.

---

## 8. Virtual Environment & Dependency Verification (`Phase 9`)

- **Virtual Environment Path**: `/Users/swaruppal/FastSpeech2_Project/.venv` (`1.06 GiB`)
- **Python Version**: `3.10.18 (main, Jun 3 2025, 18:23:41) [Clang 17.0.0]`
- **Core ML Stack**: `torch==2.10.0`, `torchaudio==2.10.0` (Apple Silicon `mps` backend built and available: `True`)
- **Audio & Alignment Stack**: `numpy==2.2.6`, `scipy==1.14.1`, `librosa==0.11.0`, `soundfile==0.14.0`, `pyworld==0.3.5`, `tgt==1.5`
- **Text & Phonetics Stack**: `g2p-en==2.1.0`, `inflect==4.1.0`, `Unidecode==1.3.8`, `nltk==3.10.4`
- **Data Ingestion & Reporting Stack**: `PyYAML==6.0.3`, `matplotlib==3.10.9`, `scikit-learn==1.7.2`, `pandas==2.3.3`, `pillow==12.1.0`, `reportlab==4.4.10`, `pyarrow==25.0.1`, `duckdb==1.5.6`, `datasets==5.1.0`, `huggingface_hub==2.4.0`
- **Policy**: `.venv/` is **retained locally** so all CLI commands and unit tests run immediately without network downloads, and is **excluded from GitHub** via `.gitignore`.

