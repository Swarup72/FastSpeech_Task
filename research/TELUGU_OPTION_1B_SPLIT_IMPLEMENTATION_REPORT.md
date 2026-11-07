# Telugu Option 1B Dataset Split Implementation & Verification Report

**Date:** 2026-11-05  
**Dataset:** `SPRINGLab/IndicTTS_Telugu` (`8,576` source utterances, `36.7123` decoded WAV hours)  
**Generator Script:** [`Telugu/metadata/generate_option_1b_split.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/generate_option_1b_split.py)  
**Output Directory:** [`Telugu/metadata/splits/option_1b/`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/)  
**Audit Decision:** **PASSED — READY FOR FULL TELUGU MFA ALIGNMENT**

---

## 1. Executive Summary

Following approval of **Option 1B** from [`research/TELUGU_DATASET_SPLIT_REAUDIT.md`](file:///Users/swaruppal/FastSpeech2_Project/research/TELUGU_DATASET_SPLIT_REAUDIT.md), we implemented an independent, deterministic split generator ([`Telugu/metadata/generate_option_1b_split.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/generate_option_1b_split.py)) that constructs the Option 1B dataset split directly from the **actual decoded WAV headers** across all 8,576 recordings in `data/telugu_parquet/data/train-*-of-00004.parquet`.

All existing files in `Telugu/metadata/` ([`partition_dataset.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/partition_dataset.py), [`train.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/train.txt), [`val.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/val.txt), [`test.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/test.txt), [`split_summary.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/split_summary.json), and [`text_manifest.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/text_manifest.json)) remain **100% untouched**.

---

## 2. Generated Option 1B Artifacts

All six required files have been generated under [`Telugu/metadata/splits/option_1b/`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/):

1. [`Telugu/metadata/splits/option_1b/train.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/train.txt) — `2,302` utterances (`5.328891` hours; SHA-256: `779367804fe3dece651d89e8b6019feffee647eeaa090bab3ecc478a651682d6`)
2. [`Telugu/metadata/splits/option_1b/val.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/val.txt) — `422` utterances (`1.000000` hour; SHA-256: `20b9d0bcad5fc923e949f7f6c303de8c26355f8ab1333d8bed82fba3b84fb85d`)
3. [`Telugu/metadata/splits/option_1b/test.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/test.txt) — `422` utterances (`1.000000` hour; SHA-256: `721acdd19961ac3016dedf0480581b8361704fda062d1797c726b6258039f0f8`)
4. [`Telugu/metadata/splits/option_1b/split_summary.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/split_summary.json) — Complete duration, gender balance, exclusion, and checksum summary.
5. [`Telugu/metadata/splits/option_1b/split_manifest.csv`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/split_manifest.csv) — Full 8,576-row utterance manifest recording `row_idx`, `uid`, `path`, `speaker_id`, `gender`, `split`, `exclusion_reason`, `actual_duration_sec`, `sample_rate`, `num_frames`, `char_len`, and `word_len`.
6. [`Telugu/metadata/splits/option_1b/split_audit.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/split_audit.json) — Automated verification audit results (`"audit_status": "PASSED"`).

---

## 3. Exact Split Statistics & Gender Balance Verification

| Split | Total Utterances | Total Decoded Duration (Sec / Hours) | Share of Active Pool | Female Utterances | Female Duration (Sec / Hours / %) | Male Utterances | Male Duration (Sec / Hours / %) | Duration Range `[Min, Median, Mean, P95, Max]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train.txt`** | **2,302** | `19,184.01s` / **`5.328891 h`** | **72.71%** | **1,055** | `9,592.00s` / **`2.664445 h`** (**50.00%**) | **1,247** | `9,592.00s` / **`2.664445 h`** (**50.00%**) | `[1.0291s, 8.1236s, 8.3336s, 13.9404s, 14.9563s]` |
| **`val.txt`** | **422** | `3,600.00s` / **`1.000000 h`** | **13.64%** | **192** | `1,800.00s` / **`0.500000 h`** (**50.00%**) | **230** | `1,800.00s` / **`0.500000 h`** (**50.00%**) | `[1.1287s, 8.4291s, 8.5308s, 13.9519s, 14.9620s]` |
| **`test.txt`** | **422** | `3,600.00s` / **`1.000000 h`** | **13.64%** | **188** | `1,800.00s` / **`0.500000 h`** (**50.00%**) | **234** | `1,800.00s` / **`0.500000 h`** (**50.00%**) | `[1.6744s, 8.4752s, 8.5308s, 13.9101s, 14.9712s]` |
| **Total Active** | **3,146** | `26,384.01s` / **`7.328891 h`** | **100.00%** | **1,435** | `13,192.00s` / **`3.664445 h`** (**50.00%**) | **1,711** | `13,192.00s` / **`3.664445 h`** (**50.00%**) | **`[1.0291s .. 14.9712s]` (`0` outliers)** |

---

## 4. Full 8,576-Utterance Corpus Accounting

Every single one of the `8,576` source recordings (`36.712271` hours) is accounted for in [`split_manifest.csv`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/split_manifest.csv) and [`split_summary.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/split_summary.json):

| Category / Disposition | Utterances | Decoded Duration (Hours) | Description |
| :--- | :---: | :---: | :--- |
| **Active `train` Split** | 2,302 | 5.328891 h | `1,055` Female (`2.664445h`) + `1,247` Male (`2.664445h`) |
| **Active `val` Split** | 422 | 1.000000 h | `192` Female (`0.500000h`) + `230` Male (`0.500000h`) |
| **Active `test` Split** | 422 | 1.000000 h | `188` Female (`0.500000h`) + `234` Male (`0.500000h`) |
| **`unused_male_reserve`** | 2,007 | 4.368657 h | Clean `[1.0s, 15.0s]` Male utterances held in reserve to preserve 50/50 gender balance across Train, Val, and Test |
| **`too_long_super_15s_actual`** | 3,219 | 24.824352 h | True decoded WAV duration `> 15.0s` (multi-sentence paragraphs up to `151.60s`) |
| **`cross_gender_duplicate_title`** | 198 | 0.187223 h | `90` shared story titles spoken by both Female (`0`) and Male (`1`), quarantined to prevent cross-split leakage |
| **`intra_gender_duplicate_transcript`** | 5 | 0.002906 h | Duplicate single-speaker transcripts (first occurrence kept, 5 duplicates excluded) |
| **`too_short_sub_1s_actual`** | 1 | 0.000264 h | True decoded WAV duration `< 1.0s` (`0.95s`) |
| **Total Source Corpus** | **8,576** | **36.712271 h** | **100.0% reconciled** |

---

## 5. Verification Checklist & Unit Test Results

All verification checks in [`Telugu/metadata/splits/option_1b/split_audit.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/split_audit.json) and [`tests/test_telugu_data_preparation.py`](file:///Users/swaruppal/FastSpeech2_Project/tests/test_telugu_data_preparation.py#L218-L331) (`TestTeluguOption1BSplit`) passed:

1. **Actual WAV Duration Window (`[1.0s, 15.0s]`)**:
   - `count_below_1s_in_active_splits`: **`0`** (min = `1.0291s`)
   - `count_above_15s_in_active_splits`: **`0`** (max = `14.9712s`)
   - All `148` previously hidden `> 15.0s` outliers (including the `42.70s` training outlier and `34.33s` validation outlier) have been eliminated.
2. **Source-ID & Normalized-Transcript Leakage**:
   - `train_val_id_overlap`: **`0`** | `train_test_id_overlap`: **`0`** | `val_test_id_overlap`: **`0`** | `active_reserve_id_overlap`: **`0`**
   - `train_val_text_overlap`: **`0`** | `train_test_text_overlap`: **`0`** | `val_test_text_overlap`: **`0`**
3. **Gender Balance & Utterance Counts**:
   - `train`: `2,302` utterances (`1,055` F / `1,247` M; **`50.00%` / `50.00%` duration**)
   - `val`: `422` utterances (`192` F / `230` M; **`50.00%` / `50.00%` duration**)
   - `test`: `422` utterances (`188` F / `234` M; **`50.00%` / `50.00%` duration**)
4. **Deterministic Reproduction (`seed=42`)**:
   - Verified by `test_04_deterministic_reproducibility`: re-running `generate_option_1b_split(seed=42)` in an isolated directory produces identical SHA-256 digests for `train.txt`, `val.txt`, and `test.txt`.
5. **Legacy Files Preserved Unchanged**:
   - Verified by `test_01_all_six_artifacts_exist_and_legacy_files_preserved`: `Telugu/metadata/train.txt` (`2,330`), `val.txt` (`457`), `test.txt` (`2,366`), `split_summary.json`, and `partition_dataset.py` remain untouched.

