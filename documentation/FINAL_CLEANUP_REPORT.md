# FastSpeech 2 Project — Final Staged Cleanup & Verification Report (`Phases 13–16`)

**Project Root**: `/Users/swaruppal/FastSpeech2_Project`  
**External Backup Root**: `/Users/swaruppal/Downloads/FastSpeech2_Cleanup_Backup/`  
**Backup Manifest**: `/Users/swaruppal/Downloads/FastSpeech2_Cleanup_Backup/BACKUP_MANIFEST.json`  
**Completion Timestamp**: 2026-11-07  
**Verification Status**: **PASSED (244 / 244 Tests Passed, 62 / 62 Deliverables Matched, 0 Checksum Mismatches)**

---

## 1. Before-and-After Project Storage Metrics (`Phase 16`)

| Metric | Before Cleanup | After Approved Staged Cleanup | Net Change |
| :--- | ---: | ---: | ---: |
| **Total Local Project Size** | **16.89 GiB** (`18,134,726,564 B`) | **12.14 GiB** (`13,033,704,574 B`) | **-4.75 GiB (`-5,101,021,990 B` / `-28.1%`)** |
| **Proposed Git-Tracked Size** | **56.41 MiB** (`59,153,087 B`) *(submodule warning)* | **81.85 MiB** (`85,828,396 B`) *(zero submodule warnings)* | **+25.44 MiB** *(tracks `original_fastspeech2/` source directly)* |
| **Number of Git-Tracked Files** | 405 files | **530 files** | +125 files *(123 `original_fastspeech2/` files + docs)* |
| **Raw Dataset Size (`data/`)** | **7.32 GiB** (`7,858,435,715 B`, `15,085 files`) | **7.32 GiB** (`7,858,435,715 B`, `15,085 files`) | **0 B (100% Untouched & Protected)** |
| **Preprocessed Data Size (`preprocessed_data/` + `original_fastspeech2/preprocessed_data/`)** | **1.62 GiB** (`1,739,558,654 B`, `55,916 files`) | **1.62 GiB** (`1,739,558,654 B`, `55,916 files`) | **0 B (100% Untouched & Protected)** |
| **Checkpoint & Vocoder Storage (`checkpoints/` + `output/ckpt/` + `hifigan/`)** | **4.86 GiB** (`5,221,956,646 B`, 17 files) | **1.61 GiB** (`1,727,366,474 B`, 5 files) | **-3.25 GiB** *(4 authoritative ckpts + HiFi-GAN kept)* |
| **Archive Storage (`*.tar.gz`, `*.zip` outside `data/`)** | **1.51 GiB** (`1,624,131,541 B`, 10 archives) | **466.86 MiB** (`489,537,047 B`, 1 archive) | **-1.06 GiB** *(Retained `Telugu_IndicTTS_Features.tar.gz`)* |
| **Nested Upstream `.git` (`original_fastspeech2/.git`)** | **455.50 MiB** (`477,626,085 B`, 193 files) | **0 B** | **-455.50 MiB** *(Moved to external backup)* |
| **Virtual Environment Size (`.venv/`)** | **1.06 GiB** (`1,132,907,658 B`) | **1.06 GiB** (`1,132,907,658 B`) | **0 B (100% Preserved)** |
| **Redundant Duplicate Checkpoint Copies** | 3 files (`932.50 MiB`) | **0 files (`0 B`)** | **-932.50 MiB** |
| **Source Code, Configs & Tests** | 100% present | **100% present** | **0 files deleted** |
| **Mandatory Assignment Deliverables** | 62 verified files | **62 verified files** | **0 files modified or deleted** |

---

## 2. Detailed Log of Executed Actions (`Phase 13`)

### 2.1 Files Backed Up with SHA-256 Verification & Removed from Workspace (`23 Files + 1 Nested Git Repo = 4.75 GiB`)

Every file listed below was copied to `/Users/swaruppal/Downloads/FastSpeech2_Cleanup_Backup/`, cryptographically verified via SHA-256 (`src_sha256 == dst_sha256`), and recorded in `BACKUP_MANIFEST.json` prior to workspace removal:

| Tier | Relative Workspace Path | Size (MiB) | Verified SHA-256 Checksum | Backup Location |
| :---: | :--- | ---: | :--- | :--- |
| **1** | `output/ckpt/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` | 399.08 | `794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e` | `~/Downloads/FastSpeech2_Cleanup_Backup/output/ckpt/...` *(+ kept in `checkpoints/`)* |
| **1** | `output/ckpt/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` | 399.08 | `6063dd284e32d7ec72b01280fc1254c9a4fc65065e2baaddfe0c31241b81cde3` | `~/Downloads/FastSpeech2_Cleanup_Backup/output/ckpt/...` *(+ kept in `checkpoints/`)* |
| **1** | `--output_path/oov_counts_pilot_dictionary.txt` | 0.01 | `7cd41226c9c98748082e0bdb2cf8546f0d75099fec6fe34d51409611e1bf4ec0` | `~/Downloads/FastSpeech2_Cleanup_Backup/--output_path/...` |
| **1** | `--output_path/oovs_found_pilot_dictionary.txt` | 0.01 | `c17c4703fe7e1f355ae0ae05e0baa633169c5c334c5b71a9bf453a9041a603bd` | `~/Downloads/FastSpeech2_Cleanup_Backup/--output_path/...` |
| **1** | `--output_path/utterance_oovs.txt` | 0.09 | `16b5fcdf042a0988398534ada8b06fe7a00b66405403fb6bfffd3da5f874bf42` | `~/Downloads/FastSpeech2_Cleanup_Backup/--output_path/...` |
| **2** | `checkpoints/Svarah_profile_stratified_v1/checkpoint_5000.pth.tar` | 134.34 | `3babc8451d5cc9c04f314e1ae8260bf5c08abd3759578490a71c48604815c786` | `~/Downloads/FastSpeech2_Cleanup_Backup/checkpoints/...` |
| **2** | `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar.mock_bak` | 134.34 | `24a4895f85783092856aa219fd24579e37ef6f4eaeb01fd9f7c520488239b463` | `~/Downloads/FastSpeech2_Cleanup_Backup/checkpoints/...` |
| **2** | `checkpoints/Svarah_profile_stratified_v1/checkpoint_best_diagnostic_mock.pth.tar.bak` | 134.34 | `24a4895f85783092856aa219fd24579e37ef6f4eaeb01fd9f7c520488239b463` | `~/Downloads/FastSpeech2_Cleanup_Backup/checkpoints/...` |
| **2** | `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar.mock_bak` | 134.34 | `6d9df4b4d63fe3659a4125f34000bfb04ce991bb57a155f722969259ba2b493a` | `~/Downloads/FastSpeech2_Cleanup_Backup/checkpoints/...` |
| **3** | `FastSpeech2_Colab_Bundle_profile_stratified_v1.tar.gz` | 464.23 | `da1c419444ab29821966e16a12d45781db18db07673ad148247f03d299784869` | `~/Downloads/FastSpeech2_Cleanup_Backup/...` |
| **3** | `FastSpeech2_Colab_Bundle.tar.gz` | 463.77 | `fd4ee9922f6dcaeff281b862f2fd81cb78491ea3da7c5ca8ebd33efad9741913` | `~/Downloads/FastSpeech2_Cleanup_Backup/...` |
| **3** | `English_inference_full_assets.zip` | 50.88 | `d2809b9b811cdea33c89131f326d0a748abb496d3dca63d5cdda0c62e7d25666` | `~/Downloads/FastSpeech2_Cleanup_Backup/...` |
| **3** | `original_fastspeech2/hifigan/generator_universal.pth.tar.zip` | 49.54 | `1fd93a04431b14296cf1d3f66a6fea89c2fdeb8f3b900a962e13ad21f67f9149` | `~/Downloads/FastSpeech2_Cleanup_Backup/original_fastspeech2/hifigan/...` |
| **3** | `original_fastspeech2/hifigan/generator_LJSpeech.pth.tar.zip` | 49.44 | `ff0b41168d347edd43d7d6859d136221c1fd9ea4698cf8998d5f0228139f2d41` | `~/Downloads/FastSpeech2_Cleanup_Backup/original_fastspeech2/hifigan/...` |
| **3** | `FastSpeech2_Source_Code.zip` | 3.42 | `b593aeb2c3468c10355ec9b09d3a32832e5c809d7a4e3bcfdc911e884e274d4e` | `~/Downloads/FastSpeech2_Cleanup_Backup/...` |
| **3** | `FastSpeech2_Colab_Code_Update_profile_stratified_v1.tar.gz` | 0.55 | `6cb62019e0846b77a797245a365d0278f8cebace69f7a180555e0e346296e45e` | `~/Downloads/FastSpeech2_Cleanup_Backup/...` |
| **3** | `Telugu_Colab_Bundle.tar.gz` | 0.21 | `ed33d61bd8c1397425e2dc16c7bf4962c017a0b23f74c2fe69b0e8d33011915e` | `~/Downloads/FastSpeech2_Cleanup_Backup/...` |
| **3** | `English_inference_only.zip` | 0.01 | `1b1aa42f9aa95495781c551cd7fcd91d29e5c4ae5f3753363d50addf480667d5` | `~/Downloads/FastSpeech2_Cleanup_Backup/...` |
| **4** | `checkpoints/IndicTTS/checkpoint_16000.pth.tar` | 397.99 | `00e3de02fcc3ebfcd710338c4aa7c42bea417def8e4b1def3760eaf81d0d5783` | `~/Downloads/FastSpeech2_Cleanup_Backup/checkpoints/IndicTTS/...` |
| **4** | `checkpoints/IndicTTS/checkpoint_17000.pth.tar` | 397.99 | `933278f7e5bf45b7304a0166460822bf81b549f50bd4d8d45eafb1ea3b7d8fb3` | `~/Downloads/FastSpeech2_Cleanup_Backup/checkpoints/IndicTTS/...` |
| **4** | `checkpoints/IndicTTS/checkpoint_18000.pth.tar` | 397.99 | `3c16341856104dea05289c49a1c0d146f976c3af7b777cda7ae061c8888f2172` | `~/Downloads/FastSpeech2_Cleanup_Backup/checkpoints/IndicTTS/...` |
| **4** | `checkpoints/IndicTTS/checkpoint_19000.pth.tar` | 397.99 | `7081bb5d653d3344cc2004476d01d8a07fe421c28371ef677be25fe593d81a2b` | `~/Downloads/FastSpeech2_Cleanup_Backup/checkpoints/IndicTTS/...` |
| **4** | `checkpoints/IndicTTS/checkpoint_20000.pth.tar` | 397.99 | `c4da76b31b83c7d20b7b005305d0c84af35239dc6089cce8211a49e748355e72` | `~/Downloads/FastSpeech2_Cleanup_Backup/checkpoints/IndicTTS/...` |
| **5** | `original_fastspeech2/.git/` (193 files) | 455.50 | *(All 193 files verified via SHA-256)* | `~/Downloads/FastSpeech2_Cleanup_Backup/original_fastspeech2_git.bak` |

### 2.2 Ephemeral Caches Removed (`1.62 MiB`)
- 24 `__pycache__/` directories (146 `.pyc` files) outside protected directories.
- `.pytest_cache/` directory (6 files).
- 12 `.DS_Store` files outside `data/` and `preprocessed_data/`.

### 2.3 Explicitly Retained Files (Per User Conditions)
- `Results/English/audio/diagnostic_test.wav` (`16,428 B`, SHA-256 `ae91bf02112db37e...`) — **Retained in workspace**.
- `SOURCE_ARCHIVE_MANIFEST.txt` (`16,474 B`, SHA-256 `cc892535348d7e9e...`) — **Retained in workspace**.
- `Telugu_IndicTTS_Features.tar.gz` (`489,537,047 B`, SHA-256 `f435fc9f2d42de35...`) — **Retained in workspace** alongside verified unpacked `preprocessed_data/Telugu_Option_1B/`.
- `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` (Step 19,000) & `checkpoint_latest.pth.tar` (Step 20,000) — **Retained in workspace**.
- `checkpoints/IndicTTS/checkpoint_best.pth.tar` (Step 13,000) & `checkpoint_latest.pth.tar` (Step 20,000) — **Retained in workspace**.
- `original_fastspeech2/hifigan/generator_universal.pth.tar` (`53.20 MiB`) — **Retained in workspace**.

---

## 3. Post-Cleanup Verification Results (`Phase 14`)

1. **Model & Vocoder Loading Verification**:
   - `English.inference.load_english_fastspeech2`: Loaded `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` (`global_step=19000`, `epoch=95`, `best_val_loss=2.348900`).
   - `Telugu.inference.load_telugu_fastspeech2`: Loaded `checkpoints/IndicTTS/checkpoint_best.pth.tar` (`global_step=13000`, `epoch=91`, `best_val_loss=2.086877`).
   - `load_hifigan_vocoder`: Loaded `original_fastspeech2/hifigan/generator_universal.pth.tar` cleanly in both English and Telugu pipelines.
2. **Deliverables & Manifest Verification**:
   - All **62 mandatory artifacts** in `documentation/final_artifact_manifest.json` and all **85 cataloged entries** in `documentation/FINAL_FILE_MANIFEST.json` passed SHA-256 and byte-size checks with **0 mismatches**.
   - All **41 PNG figures** in `Report/figures/` verified present and non-empty.
   - `Results/Telugu/official_samples.json` T1 verified exact: `న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.`.
3. **Full Regression Test Suite (`unittest discover -s tests -p "test_*.py" -v`)**:
   - **Total Tests Executed**: **244 tests** across all **22 test modules** (`test_colab_monitoring_and_notebook.py`, `test_config.py`, `test_duration_extraction.py`, `test_english_inference.py`, `test_english_normalizer.py`, `test_english_training_audit.py`, `test_fastspeech2.py`, `test_fastspeech2_loss.py`, `test_models.py`, `test_postnet.py`, `test_svarah_manifests.py`, `test_svarah_preprocessing_qc.py`, `test_telugu_data_preparation.py`, `test_telugu_inference.py`, `test_telugu_mfa_alignment_pilot.py`, `test_telugu_option1b_features.py`, `test_telugu_split_vocabulary_audit.py`, `test_training_engine.py`, `test_transformer.py`, `test_utils.py`, `test_variance_adaptor.py`, `test_variance_components.py`).
   - **Passed**: **244 / 244 (`100%`)**
   - **Failed**: **0**
   - **Errors / Skipped**: **0**
   - *(Specifically, `test_telugu_option1b_features.py` independently verified all 2,721 extracted Telugu mel, pitch, energy, and duration `.npy` files in `preprocessed_data/Telugu_Option_1B/`, and `test_colab_monitoring_and_notebook.py` verified all 15,488 extracted Svarah `.npy` files in `preprocessed_data/Svarah/`).*

