# English Svarah Speaker-Profile-Stratified Split Implementation Report (Plan B-Lite: `profile_stratified_v1`)

**Date:** 2026-11-05  
**Split Version:** `profile_stratified_v1`  
**Output Directory:** [`English/metadata/splits/profile_stratified_v1/`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/splits/profile_stratified_v1)  
**Training Config:** [`config/Svarah/train_profile_stratified_v1.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/train_profile_stratified_v1.yaml)  
**Generator Script:** [`English/metadata/generate_profile_stratified_v1_split.py`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/generate_profile_stratified_v1_split.py)

---

## 1. Executive Summary

Plan B-Lite has been implemented and verified. Using the existing `3,872` preprocessed English Svarah utterances (`15,488` `.npy` feature arrays across `mel/`, `pitch/`, `energy/`, and `duration/`), we generated a deterministic (`seed = 20261105`), `user_prefix`-disjoint, `recording_id`-disjoint, and transcript-disjoint stratified train/validation split across all **`88` active demographic speaker profiles**:

| Metric | Baseline Split (`preprocessed_data/Svarah/`) | New Stratified Split (`profile_stratified_v1`) | Improvement / Guarantee |
| :--- | ---: | ---: | :--- |
| **Training Utterances** | `3,208` (`4.5546 h`) | **`3,208` (`4.5023 h`)** | Exact match (`200` steps/epoch at `batch_size=16`) |
| **Validation Utterances** | `664` (`0.9168 h`) | **`664` (`0.9784 h`)** | Exact match (`664` validation utterances) |
| **Total Preprocessed Utterances** | `3,872` | **`3,872`** | `100%` preserved (`0` dropped) |
| **Distinct Profiles in `train.txt`** | `66 / 88` (`75.0%`) | **`88 / 88` (`100.0%`)** | All `88` active profiles trained in `speaker_emb` |
| **Distinct Profiles in `val.txt`** | `22 / 88` (`25.0%`) | **`88 / 88` (`100.0%`)** | Full demographic profile coverage in validation |
| **Seen Validation Profiles** | **`0 / 22` (`0.00%`)** | **`88 / 88` (`100.00%`)** | Eliminates untrained random $\mathcal{N}(0, I_{256})$ embeddings at validation |
| **Train Gender Balance (F / M)** | `1,680` (`52.37%`) / `1,528` (`47.63%`) | **`1,589` (`49.53%`) / `1,619` (`50.47%`)** | Balanced (`~50/50`) |
| **Validation Gender Balance (F / M)** | `241` (`36.30%`) / `423` (`63.70%`) | **`332` (`50.00%`) / `332` (`50.00%`)** | Exact **`50.00%` Female / `50.00%` Male** balance |
| **Train / Val Mean Pitch ($z$)** | `+0.0230` / **`-0.2463`** | **`-0.0166` / `-0.0549`** | Eliminates the `-0.27` $\sigma$ male register skew in validation |
| **`user_prefix` Overlap (`train ∩ val`)** | `0` | **`0` (`0.00%`)** | Zero contributor/session prefix leakage |
| **`recording_id` Overlap (`train ∩ val`)** | `0` | **`0` (`0.00%`)** | Zero source-recording leakage |
| **Normalized Transcript Overlap (`train ∩ val`)** | `7` (`8` val utts) | **`0` (`0.00%`)** | Eliminates the `7` leaked transcripts in the baseline split |
| **Validation Pitch MSE Floor (Predicting Train Profile Mean)** | **`1.2468`** | **`0.4582`** | **`-63.25%` (`-0.7886` MSE) reduction in irreducible pitch floor** |

---

## 2. Generated Artifacts & Checksums

All versioned split artifacts are stored under [`English/metadata/splits/profile_stratified_v1/`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/splits/profile_stratified_v1):

| Artifact | Path | Description & Verification |
| :--- | :--- | :--- |
| **`train.txt`** | [`English/metadata/splits/profile_stratified_v1/train.txt`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/splits/profile_stratified_v1/train.txt) | `3,208` lines (`1,396,077` mel frames, `158,660` phoneme tokens, `88` profiles, `1,316` `user_prefix`es, `1,444` `recording_id`s). SHA-256: `1fcd48ec9957153318f4a8626fa2e8a3d16ff37841b75fcde877edfa3c4a6116` |
| **`val.txt`** | [`English/metadata/splits/profile_stratified_v1/val.txt`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/splits/profile_stratified_v1/val.txt) | `664` lines (`303,371` mel frames, `33,996` phoneme tokens, `88` profiles, `155` `user_prefix`es, `171` `recording_id`s). SHA-256: `aa969d5b9b069bf1b4288c4f1b60f800f7fc7de7d511938ff379207cc658aab2` |
| **`split_summary.json`** | [`English/metadata/splits/profile_stratified_v1/split_summary.json`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/splits/profile_stratified_v1/split_summary.json) | Aggregate counts, hours, gender balance, pitch MSE floor comparison, and SHA-256 hashes |
| **`split_manifest.csv`** | [`English/metadata/splits/profile_stratified_v1/split_manifest.csv`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/splits/profile_stratified_v1/split_manifest.csv) | 17-column utterance-level manifest (`3,872` rows) tracking `new_split`, `original_split`, `user_prefix`, `recording_id`, demographics, frame count, and duration |
| **`split_audit.json`** | [`English/metadata/splits/profile_stratified_v1/split_audit.json`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/splits/profile_stratified_v1/split_audit.json) | Complete boolean verification checklist, zero-overlap audit, frozen-file SHA-256 verification, and per-profile (`88` profiles) train/val breakdown |

---

## 3. Separate Step-0 Training Configuration & Pipeline Support

1. **New Training Config ([`config/Svarah/train_profile_stratified_v1.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/train_profile_stratified_v1.yaml)):**
   - Points to isolated output directories so existing baseline checkpoints and logs are never overwritten:
     - `ckpt_path: "./output/ckpt/Svarah_profile_stratified_v1"`
     - `log_path: "./output/log/Svarah_profile_stratified_v1"`
     - `result_path: "./output/result/Svarah_profile_stratified_v1"`
   - Points to the versioned split manifests:
     - `split_dir: "English/metadata/splits/profile_stratified_v1"`
     - `train_manifest: "English/metadata/splits/profile_stratified_v1/train.txt"`
     - `val_manifest: "English/metadata/splits/profile_stratified_v1/val.txt"`
   - Keeps all optimizer (`batch_size: 16`, `warm_up_step: 4000`, `weight_decay: 0.0`, `init_lr: 0.0625`, `grad_clip_thresh: 1.0`) and step parameters identical to [`config/Svarah/train.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/train.yaml).
2. **Manifest Resolution in [`English/train.py`](file:///Users/swaruppal/FastSpeech2_Project/English/train.py#L79-L97) and [`English/monitoring.py`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py#L159-L197):**
   - `English/train.py` supports `--split_dir`, `--train_manifest`, and `--val_manifest` (via CLI or `train_config.path`), passing the resolved manifest paths to `Dataset(...)` and `validate_svarah_dataset(...)` while defaulting to `preprocessed_data/Svarah/{train.txt,val.txt}` when using the original `config/Svarah/train.yaml`.
   - [`English/prepare_colab_bundle.py`](file:///Users/swaruppal/FastSpeech2_Project/English/prepare_colab_bundle.py#L48-L53) includes `English/metadata/splits/profile_stratified_v1` in `project_items` so even a lightweight `--code_only` bundle (`~250 KB`) ships the new manifests and config to Google Colab without re-uploading any `.npy` files.

---

## 4. Verification of Frozen Baseline Files (SHA-256)

All existing manifests, configurations, and feature files were verified unchanged before and after implementation:

| Frozen File | Verified SHA-256 | Status |
| :--- | :--- | :--- |
| [`English/metadata/train.txt`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/train.txt) | `4c34430c082bf7ec7055e5dd0283c1b61296f0a1e6381235bbc0bcafa4a9239b` | **UNCHANGED** |
| [`English/metadata/val.txt`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/val.txt) | `5c18f223e3b5b604c3dba4264d5bb4778cc2e57f3ef6160a5a65296dc69d11a2` | **UNCHANGED** |
| [`English/metadata/test.txt`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/test.txt) | `b265543e1a4d8531a227609f4356e9494c9de420e3945da1bde2dc0d48542fef` | **UNCHANGED** |
| [`preprocessed_data/Svarah/train.txt`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/train.txt) | `047bb252fffec303dab03fd53cba7ae4dcb4d5487fac17d09553d0c9438f896e` | **UNCHANGED** |
| [`preprocessed_data/Svarah/val.txt`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/val.txt) | `b2bb3967e26ee044b7f75a11341f58913caf879cec133cb5575af5fe89ba558d` | **UNCHANGED** |
| [`preprocessed_data/Svarah/stats.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/stats.json) | `1c2e9ff77e49de744f68cd7944c93992fdda317bbe374a21d0fdf47c671f1f79` | **UNCHANGED** |
| [`preprocessed_data/Svarah/speakers.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/speakers.json) | `1a5313130a7a2e0b9c7196ee303cb6a7817d0257791bd25775b27293592587de` | **UNCHANGED** |
| [`config/Svarah/preprocess.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/preprocess.yaml) | `b6db335b38b285ef40b68831eed19aedc1e511da6e9d6cb1f4acc0b23b6b35de` | **UNCHANGED** |
| [`config/Svarah/model.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/model.yaml) | `bf5a5e60905419aa19c8c9305994d45a69ff69ca182f716b9a7ff54d5aca70c4` | **UNCHANGED** |
| [`config/Svarah/train.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/train.yaml) | `71bcea7b8e96f30e2f83865e4a6f8b72ccb6449cf1ef2015fb752d2cdd8d1657` | **UNCHANGED** |

---

## 5. Regression Test Suite Verification

Executed `tests/test_colab_monitoring_and_notebook.py` (`8/8` tests passed):
- `test_01_notebook_structure_and_syntax`: **OK**
- `test_02_persistent_logger_and_resume_deduplication`: **OK**
- `test_03_dashboard_five_graphs_and_summary`: **OK**
- `test_04_verified_checkpoint_saving_and_best_guard`: **OK**
- `test_05_colab_code_only_bundle_creation`: **OK**
- `test_06_empty_tail_batch_group_and_epoch_transition`: **OK**
- `test_07_validate_svarah_dataset_timings_and_regression_guarantees`: **OK**
- `test_08_profile_stratified_v1_split_and_fresh_training_config`: **OK** (`15,488` files verified in `0.108s`, `88/88` profiles in train & val, `0` `user_prefix`/`recording_id`/transcript overlap, deterministic SHA-256 reproduction, and frozen file SHA-256 verification)


