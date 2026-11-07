# FastSpeech 2 Project — Safe Cleanup & Backup Plan (`Phase 11`)

**Project Root**: `/Users/swaruppal/FastSpeech2_Project`  
**External Backup Root**: `~/Downloads/FastSpeech2_Cleanup_Backup/` (`/Users/swaruppal/Downloads/FastSpeech2_Cleanup_Backup/`)  
**Execution Status**: **PENDING USER APPROVAL (Zero files modified or deleted)**

---

## 1. Guiding Principles & Protected Assets

1. **Zero Data or Preprocessing Loss**: `data/` (7.32 GiB), `preprocessed_data/` (1.57 GiB), and `original_fastspeech2/preprocessed_data/` (52.00 MiB) are **100% untouched**.
2. **Authoritative Checkpoints Preserved in Place**:
   - English Best (Step 19,000): `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` (`418,465,625 B`, SHA-256 `794b5123...`) — **KEPT IN PLACE**.
   - English Latest (Step 20,000): `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` (`418,467,535 B`, SHA-256 `6063dd28...`) — **KEPT IN PLACE**.
   - Telugu Best (Step 13,000): `checkpoints/IndicTTS/checkpoint_best.pth.tar` (`417,319,705 B`, SHA-256 `0a6c443a...`) — **KEPT IN PLACE**.
   - Telugu Latest (Step 20,000): `checkpoints/IndicTTS/checkpoint_latest.pth.tar` (`417,324,751 B`, SHA-256 `8e46b77b...`) — **KEPT IN PLACE**.
   - Universal HiFi-GAN Vocoder: `original_fastspeech2/hifigan/generator_universal.pth.tar` (`55,788,858 B`, SHA-256 `771eaf48...`) — **KEPT IN PLACE**.
3. **Zero Source Code, Test, Config, or Deliverable Deletion**: Every `.py` module, `.yaml` config, unit test (`tests/`), report (`Report/`), and assignment output (`Results/`) is preserved.
4. **Virtual Environment Preserved**: `.venv/` (1.06 GiB) is retained locally and excluded from GitHub via `.gitignore`.

---

## 2. Proposed Cleanup Candidates by Tier (`5,331.60 MiB / 5.21 GiB` Total Recoverable)

### Tier 1: Byte-Identical Duplicate Checkpoints & Ephemeral Caches (`799.92 MiB / 838,769,285 Bytes`)

| Exact path | Size | SHA-256 | Purpose | Dependencies | Risk | Recommendation | Recoverability | Backup | Space saved |
| :--- | ---: | :--- | :--- | :--- | :---: | :---: | :--- | :---: | ---: |
| `output/ckpt/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` | 418,465,625 B (399.08 MiB) | `794b5123a25f7052c2479ae6bde8f700f952d699e6148f66587e0eb8cf54245e` | Duplicate copy of English best checkpoint | None (`English/inference.py` resolves `checkpoints/Svarah_profile_stratified_v1/`) | Low | Delete duplicate | Identical copy kept at `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar` | Verified in `checkpoints/` | 399.08 MiB |
| `output/ckpt/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` | 418,467,535 B (399.08 MiB) | `6063dd284e32d7ec72b01280fc1254c9a4fc65065e2baaddfe0c31241b81cde3` | Duplicate copy of English latest checkpoint | None | Low | Delete duplicate | Identical copy kept at `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar` | Verified in `checkpoints/` | 399.08 MiB |
| `--output_path/oov_counts_pilot_dictionary.txt` | 8,918 B (8.71 KiB) | `7cd41226c9c98748082e0bdb2cf8546f0d75099fec6fe34d51409611e1bf4ec0` | Stray pilot OOV count created by mistyped `--output_path` flag | None | Low | Backup & Delete | Backed up to `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 8.71 KiB |
| `--output_path/oovs_found_pilot_dictionary.txt` | 7,159 B (6.99 KiB) | `c17c4703fe7e1f355ae0ae05e0baa633169c5c334c5b71a9bf453a9041a603bd` | Stray pilot OOV word list | None | Low | Backup & Delete | Backed up to `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 6.99 KiB |
| `--output_path/utterance_oovs.txt` | 92,815 B (90.64 KiB) | `16b5fcdf042a0988398534ada8b06fe7a00b66405403fb6bfffd3da5f874bf42` | Stray pilot utterance OOV dump | None | Low | Backup & Delete | Backed up to `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 90.64 KiB |
| `Results/English/audio/diagnostic_test.wav` | 16,428 B (16.04 KiB) | `ae91bf02112db37e4b38c2b7bbc0120f2dbf316078d9052c8049eacc08c2d50b` | Early vocoder smoke-test waveform (not an assignment sample) | None (0 references in code, tests, or report) | Low | Backup & Delete | Backed up to `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 16.04 KiB |
| `SOURCE_ARCHIVE_MANIFEST.txt` | 16,474 B (16.09 KiB) | `cc892535348d7e9ea38840afd23dd59b0ee6ace271e7854c2628bfb38a5f5b8b` | Superseded text listing for `FastSpeech2_Source_Code.zip` | None (superseded by `documentation/FINAL_ARTIFACT_MANIFEST.md`) | Low | Backup & Delete | Backed up to `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 16.09 KiB |
| `research/scratch_audio/` | 0 B | N/A (Empty directory) | Empty temporary audio folder | None | Low | Delete empty dir | Instantly recreatable (`mkdir`) | Not required | 0 B |
| `__pycache__/` (23 directories, 146 `.pyc` files) | 1,586,699 B (1.51 MiB) | *(146 bytecode files)* | Python interpreter bytecode caches | None (regenerated automatically by Python) | Low | Delete cache | Automatically regenerated on import | Not required | 1.51 MiB |
| `.pytest_cache/` (6 cache files) | 17,472 B (17.06 KiB) | *(6 pytest cache files)* | Pytest runner state cache | None | Low | Delete cache | Automatically regenerated by `pytest` | Not required | 17.06 KiB |
| `.DS_Store` (12 files outside `data/` & `preprocessed_data/`) | 90,116 B (88.00 KiB) | *(12 macOS Finder metadata files)* | macOS Finder folder view metadata | None | Low | Delete OS metadata | OS regenerates as needed | Not required | 88.00 KiB |

---

### Tier 2: Superseded Pre-Colab Pilot / Mock Checkpoint Backups (`537.35 MiB / 563,456,480 Bytes`)

| Exact path | Size | SHA-256 | Purpose | Dependencies | Risk | Recommendation | Recoverability | Backup | Space saved |
| :--- | ---: | :--- | :--- | :--- | :---: | :---: | :--- | :---: | ---: |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_5000.pth.tar` | 140,863,999 B (134.34 MiB) | `3babc8451d5cc9c04f314e1ae8260bf5c08abd3759578490a71c48604815c786` | Early Epoch 1 pilot checkpoint (Val Loss 3.9105) | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 134.34 MiB |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar.mock_bak` | 140,863,999 B (134.34 MiB) | `24a4895f85783092856aa219fd24579e37ef6f4eaeb01fd9f7c520488239b463` | Diagnostic mock backup from Epoch 1 pilot | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 134.34 MiB |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_best_diagnostic_mock.pth.tar.bak` | 140,863,999 B (134.34 MiB) | `24a4895f85783092856aa219fd24579e37ef6f4eaeb01fd9f7c520488239b463` | Byte-identical duplicate of `.mock_bak` above | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 134.34 MiB |
| `checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar.mock_bak` | 140,864,483 B (134.34 MiB) | `6d9df4b4d63fe3659a4125f34000bfb04ce991bb57a155f722969259ba2b493a` | Diagnostic mock latest backup from Epoch 1 pilot | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 134.34 MiB |

---

### Tier 3: Root Colab Deployment Archive Bundles & Redundant HiFi-GAN Zips (`1,548.89 MiB / 1,624,131,541 Bytes`)

| Exact path | Size | SHA-256 | Purpose | Dependencies | Risk | Recommendation | Recoverability | Backup | Space saved |
| :--- | ---: | :--- | :--- | :--- | :---: | :---: | :--- | :---: | ---: |
| `Telugu_IndicTTS_Features.tar.gz` | 489,537,047 B (466.86 MiB) | `f435fc9f2d42de359f1c3a55106f39c3a67a962ea9ba6e320e36d805fdcb0cde` | Compressed upload bundle of `preprocessed_data/Telugu_Option_1B/` | None (unpacked features remain in `preprocessed_data/`) | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` + unpacked in `preprocessed_data/` | Required | 466.86 MiB |
| `FastSpeech2_Colab_Bundle_profile_stratified_v1.tar.gz` | 486,779,356 B (464.23 MiB) | `da1c419444ab29821966e16a12d45781db18db07673ad148247f03d299784869` | Compressed upload bundle of English features + code | None (unpacked features remain in `preprocessed_data/Svarah/`) | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` + regenerable via `English/prepare_colab_bundle.py` | Required | 464.23 MiB |
| `FastSpeech2_Colab_Bundle.tar.gz` | 486,296,901 B (463.77 MiB) | `fd4ee9922f6dcaeff281b862f2fd81cb78491ea3da7c5ca8ebd33efad9741913` | Superseded v0 English Colab bundle | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 463.77 MiB |
| `English_inference_full_assets.zip` | 53,349,916 B (50.88 MiB) | `d2809b9b811cdea33c89131f326d0a748abb496d3dca63d5cdda0c62e7d25666` | Downloaded Colab inference zip (unpacked in `Results/English/`) | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 50.88 MiB |
| `original_fastspeech2/hifigan/generator_universal.pth.tar.zip` | 51,942,859 B (49.54 MiB) | `1fd93a04431b14296cf1d3f66a6fea89c2fdeb8f3b900a962e13ad21f67f9149` | Compressed archive of `generator_universal.pth.tar` | Unpacked `generator_universal.pth.tar` (53.20 MiB) is kept in place and loaded first | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` + unpacked `.pth.tar` kept | Required | 49.54 MiB |
| `original_fastspeech2/hifigan/generator_LJSpeech.pth.tar.zip` | 51,838,074 B (49.44 MiB) | `ff0b41168d347edd43d7d6859d136221c1fd9ea4698cf8998d5f0228139f2d41` | Unused LJSpeech-only vocoder zip from upstream repo | None (project uses `generator_universal.pth.tar`) | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 49.44 MiB |
| `FastSpeech2_Source_Code.zip` | 3,588,558 B (3.42 MiB) | `b593aeb2c3468c10355ec9b09d3a32832e5c809d7a4e3bcfdc911e884e274d4e` | Snapshot zip of project source code | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 3.42 MiB |
| `FastSpeech2_Colab_Code_Update_profile_stratified_v1.tar.gz` | 575,668 B (0.55 MiB) | `6cb62019e0846b77a797245a365d0278f8cebace69f7a180555e0e346296e45e` | Incremental Colab code patch archive | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 0.55 MiB |
| `Telugu_Colab_Bundle.tar.gz` | 215,056 B (0.21 MiB) | `ed33d61bd8c1397425e2dc16c7bf4962c017a0b23f74c2fe69b0e8d33011915e` | Telugu code upload archive for Colab | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 0.21 MiB |
| `English_inference_only.zip` | 8,106 B (0.01 MiB) | `1b1aa42f9aa95495781c551cd7fcd91d29e5c4ae5f3753363d50addf480667d5` | Small English inference script zip | None | Low | Archive to Backup & Delete | Preserved in `~/Downloads/FastSpeech2_Cleanup_Backup/` | Required | 0.01 MiB |

---

### Tier 4: Telugu Intermediate Periodic Checkpoints Steps 16,000–20,000 (`1,989.94 MiB / 2,086,608,548 Bytes`)

> **Note**: Both `checkpoints/IndicTTS/checkpoint_best.pth.tar` (Step 13,000, Epoch 91, Val Loss 2.0869) and `checkpoints/IndicTTS/checkpoint_latest.pth.tar` (Step 20,000, Epoch 140, final training state) are **KEPT IN PLACE** inside `checkpoints/IndicTTS/`. Only the 5 non-best periodic intermediate snapshots below are moved to external backup.

| Exact path | Size | SHA-256 | Purpose | Dependencies | Risk | Recommendation | Recoverability | Backup | Space saved |
| :--- | ---: | :--- | :--- | :--- | :---: | :---: | :--- | :---: | ---: |
| `checkpoints/IndicTTS/checkpoint_16000.pth.tar` | 417,320,980 B (397.99 MiB) | `00e3de02fcc3ebfcd710338c4aa7c42bea417def8e4b1def3760eaf81d0d5783` | Step 16,000 periodic snapshot (Val Loss 2.1027) | None | Low | Archive to `~/Downloads/FastSpeech2_Cleanup_Backup/` | 100% preserved in external backup | Required | 397.99 MiB |
| `checkpoints/IndicTTS/checkpoint_17000.pth.tar` | 417,321,108 B (397.99 MiB) | `933278f7e5bf45b7304a0166460822bf81b549f50bd4d8d45eafb1ea3b7d8fb3` | Step 17,000 periodic snapshot (Val Loss 2.0946) | None | Low | Archive to `~/Downloads/FastSpeech2_Cleanup_Backup/` | 100% preserved in external backup | Required | 397.99 MiB |
| `checkpoints/IndicTTS/checkpoint_18000.pth.tar` | 417,321,236 B (397.99 MiB) | `3c16341856104dea05289c49a1c0d146f976c3af7b777cda7ae061c8888f2172` | Step 18,000 periodic snapshot (Val Loss 2.1078) | None | Low | Archive to `~/Downloads/FastSpeech2_Cleanup_Backup/` | 100% preserved in external backup | Required | 397.99 MiB |
| `checkpoints/IndicTTS/checkpoint_19000.pth.tar` | 417,321,364 B (397.99 MiB) | `7081bb5d653d3344cc2004476d01d8a07fe421c28371ef677be25fe593d81a2b` | Step 19,000 periodic snapshot (Val Loss 2.0952) | None | Low | Archive to `~/Downloads/FastSpeech2_Cleanup_Backup/` | 100% preserved in external backup | Required | 397.99 MiB |
| `checkpoints/IndicTTS/checkpoint_20000.pth.tar` | 417,323,860 B (397.99 MiB) | `c4da76b31b83c7d20b7b005305d0c84af35239dc6089cce8211a49e748355e72` | Step 20,000 periodic snapshot (redundant with `checkpoint_latest.pth.tar`) | None | Low | Archive to `~/Downloads/FastSpeech2_Cleanup_Backup/` | 100% preserved in external backup + `checkpoint_latest.pth.tar` kept | Required | 397.99 MiB |

---

### Tier 5: Nested Upstream Git Metadata (`original_fastspeech2/.git` — `455.50 MiB / 477,626,085 Bytes`)

| Exact path | Size | SHA-256 | Purpose | Dependencies | Risk | Recommendation | Recoverability | Backup | Space saved |
| :--- | ---: | :--- | :--- | :--- | :---: | :---: | :--- | :---: | ---: |
| `original_fastspeech2/.git/` (32 files, including 316 MiB stale `tmp_pack_LAa061`) | 477,626,085 B (455.50 MiB) | *(Directory of 32 Git pack/index files)* | Upstream `ming024/FastSpeech2` Git history & interrupted pack | Causes `warning: adding embedded git repository` if left inside `original_fastspeech2/` | Low | Move to `~/Downloads/FastSpeech2_Cleanup_Backup/original_fastspeech2_git.bak` | 100% preserved in external backup | Required | 455.50 MiB |

---

## 3. Summary of Space Saved & Before/After Projection

| Metric | Before Cleanup | After Cleanup (Tiers 1–5 Approved) | Net Change |
| :--- | ---: | ---: | ---: |
| **Total Local Workspace Size** | **16.89 GiB** (18,134,726,564 B) | **11.68 GiB** (12,544,134,625 B) | **-5.21 GiB (-5,590,591,939 B / -30.8%)** |
| **Raw Dataset Storage (`data/`)** | 6.98 GiB (7,494,394,812 B) | 6.98 GiB (7,494,394,812 B) | **0 B (100% Preserved)** |
| **Preprocessed Features (`preprocessed_data/` + `original_fastspeech2/preprocessed_data/`)** | 1.62 GiB (1,737,452,705 B) | 1.62 GiB (1,737,452,705 B) | **0 B (100% Preserved)** |
| **Checkpoint Storage (`checkpoints/` + `output/ckpt/` + HiFi-GAN)** | 4.86 GiB (5,221,956,646 B) | **1.61 GiB** (1,727,366,474 B) | **-3.25 GiB (4 authoritative + HiFi-GAN kept)** |
| **Archive Bundle Storage (`*.tar.gz`, `*.zip` outside `data/`)** | 1.51 GiB (1,624,131,541 B) | **0 B** | **-1.51 GiB (Backed up externally)** |
| **Nested Upstream `.git` (`original_fastspeech2/.git`)** | 455.50 MiB (477,626,085 B) | **0 B** | **-455.50 MiB (Backed up externally)** |
| **Virtual Environment (`.venv/`)** | 1.06 GiB (1,132,899,498 B) | 1.06 GiB (1,132,899,498 B) | **0 B (100% Preserved)** |
| **Proposed Git-Tracked Repository Size** | 56.41 MiB (405 files, submodule warning) | **81.80 MiB (527 files, zero submodule warnings)** | **Clean, self-contained GitHub repo (<100 MiB)** |

---

## 4. Backup Protocol (`~/Downloads/FastSpeech2_Cleanup_Backup/`)

Before any file in Tiers 1–5 is removed from `/Users/swaruppal/FastSpeech2_Project`:
1. Create `/Users/swaruppal/Downloads/FastSpeech2_Cleanup_Backup/` preserving relative directory structure.
2. Copy/archive every unique candidate file and compute its SHA-256 checksum at the backup destination.
3. Verify that the backup SHA-256 matches the source SHA-256 in Section 2 above before unlinking the workspace file.
