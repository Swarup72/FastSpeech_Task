# Svarah Metadata Manifest Construction & Data Integrity Report

**Stage**: Stage 2 — Step 1.4J-B2  
**Date**: November 4, 2026  
**Auditor**: Senior Speech Processing & ML Systems Research Engineer  
**Status**: **COMPLETED & VERIFIED**  
**Target Deadline**: 7 November 2026  
**Deliverable**: Comprehensive Metadata Manifest, Rejection Accounting, and Data Integrity Audit  

---

## 1. Executive Summary

This report documents the construction and automated validation of the complete Svarah metadata manifest. All 6,656 utterances from the authenticated Svarah corpus (`ai4bharat/Svarah`) were audited, normalized, validated against duration constraints, and linked to deterministic demographic profile identifiers.

### Key Verification Milestones

| Metric / Audit Check | Value | Verification Status |
| :--- | :--- | :---: |
| **Total Corpus Utterances** | **6,656** | **100% Accounted For** |
| **Total Audio Duration** | **9.614 hours** ($34,612.0 \text{ s}$) | **Verified from Raw Metadata** |
| **Unique Utterance IDs** | **6,656** (zero collisions) | **VERIFIED** |
| **Duplicate Audio Paths** | **0** | **VERIFIED** |
| **Empty / Missing Transcripts** | **0** | **VERIFIED** |
| **Duration Filter Limits** | $1.00 \text{ s} \le \text{duration} \le 15.00 \text{ s}$ | **Approved Decision 2** |
| **Valid Utterances** | **5,180 utterances** (**7.966 hours**) | **VERIFIED** |
| **Rejected Utterances (< 1.0s)** | **1,215 utterances** (**0.197 hours** / $11.8 \text{ min}$) | **Cataloged in rejected_utterances.json** |
| **Rejected Utterances (> 15.0s)** | **261 utterances** (**1.451 hours** / $87.1 \text{ min}$) | **Cataloged in rejected_utterances.json** |
| **Total Rejections** | **1,476 utterances** (**1.648 hours**) | **Zero Data Lost (Auditable JSON)** |
| **Unique Demographic Profiles** | **115 profiles** (all present in valid pool) | **VERIFIED** |
| **Contiguous Profile IDs** | `speaker_profile_id`: **0 to 114** | **Deterministic & Stable** |

---

## 2. Dataset Schema & Inferred Fields

The source dataset is hosted as three Parquet shards (`data/test-0000{0,1,2}-of-00003.parquet`) under the single split name `test`. The schema consists of 11 source attributes, which were ingested and enriched:

```text
Source Field              Data Type   Description & Content
----------------------------------------------------------------------------------------------------------
audio_filepath.path       string      Unique audio filename (e.g., 281474976884635_f3269_chunk_0.wav)
duration                  float64     Audio duration in seconds (matches decoded WAV frames exactly)
text                      string      Original crowd-sourced transcript from Karya mobile app
gender                    string      Speaker gender ('Female', 'Male')
age-group                 string      Age bucket ('18-30', '30-45', '45-60', '60+')
primary_language          string      Native L1 Indian language (19 distinct languages)
native_place_state        string      Origin state/UT (e.g. 'Karnataka', 'Assam', 'Jammu Kashmir')
native_place_district     string      District of origin (e.g. 'Hassan', 'Darbhanga', 'Kalimpong')
highest_qualification     string      Educational level ('Graduate', 'Post Graduate', 'Doctoral', etc.)
job_category              string      Employment category ('Full Time', 'Part Time', 'Student', etc.)
occupation_domain         string      Work sector ('Technology and Services', 'Education and Research', etc.)
```

### Inferred / Derived Attributes

1. **`utterance_id`**: Stem of `audio_filepath.path` (e.g. `281474976884635_f3269_chunk_0`). 100% unique across all 6,656 rows.
2. **`normalized_text`**: Text produced by the new deterministic English text normalizer (`English/text/normalizer.py`), expanding numbers, ordinals, symbols, and letter-spaced initialisms.
3. **`demo_key`**: Pipe-delimited string concatenation of all 8 demographic fields:
   `gender|age-group|primary_language|native_place_state|native_place_district|highest_qualification|job_category|occupation_domain`
4. **`speaker_profile_id`**: Contiguous integer ID ($0 \le \text{ID} \le 114$) assigned by lexicographically sorting unique `demo_key` values.
5. **`status`**: `'valid'` ($1.0 \le \text{duration} \le 15.0$) or `'rejected'`.
6. **`rejection_reason`**: `'duration_under_1s'`, `'duration_over_15s'`, or `null` for valid utterances.

---

## 3. Duration Filtering & Rejection Accounting

In accordance with **Approved Decision 2**, audio duration filtering was applied:
- **Minimum duration**: 1.000 second.
- **Maximum duration**: 15.000 seconds.

### 3.1 Rejection Statistics

```text
Category                      Count   Duration (s)   Duration (h)   Fraction of Corpus
--------------------------------------------------------------------------------------
Under 1.0 second (< 1.0s)     1,215       709.5 s        0.197 h            18.25 %
Over 15.0 seconds (> 15.0s)     261     5,224.0 s        1.451 h             3.92 %
Total Rejected                1,476     5,933.5 s        1.648 h            22.17 %
Total Retained (Valid)        5,180    28,678.5 s        7.966 h            77.83 %
--------------------------------------------------------------------------------------
Corpus Total                  6,656    34,612.0 s        9.614 h           100.00 %
```

### 3.2 Acoustic Justification for Duration Limits
1. **Under 1.0 Second Rejections**:
   - The majority of sub-1s recordings are single-word interjections (*"Wow"*, *"Off"*, *"Yes"*, *"No"*) or truncated breath sounds lasting 0.09s to 0.5s.
   - In FastSpeech 2, utterances with very few phonemes (1–3 tokens) create degenerate attention alignments and boundary instability in convolutional variance predictors.
   - Total lost audio is merely **11.8 minutes** across 1,215 files, removing substantial acoustic noise at negligible data cost.
2. **Over 15.0 Seconds Rejections**:
   - Utterances exceeding 15 seconds (up to 29.87s) require $T_{\text{mel}} > 1,300$ frames at 86.13 fps.
   - FastSpeech 2's positional embeddings are configured with `max_seq_len: 1000` ($11.6 \text{ s}$ speech at 22.05 kHz). Feeding unsegmented 25s–30s audio causes positional out-of-bounds exceptions or OOM errors during GPU self-attention.
   - Excluding these 261 long recordings ensures strict memory stability and avoids truncated spectrogram targets during training.

### 3.3 Zero Data Loss Guarantee
No raw audio files or transcripts were deleted. All 1,476 rejected utterances are fully preserved in `English/metadata/rejected_utterances.json` with their original paths, durations, transcripts, demographic profile keys, and explicit rejection reasons.

---

## 4. Manifest File Specifications

The metadata manifest is serialized in two complementary formats:

### 4.1 Full JSON Manifest (`English/metadata/full_manifest.json`)
Contains 6,656 structured JSON records. Schema:

```json
{
  "utterance_id": "281474976884635_f3269_chunk_0",
  "audio_path": "281474976884635_f3269_chunk_0.wav",
  "raw_text": "Our district Hasana was the land of the great Hoysala dynasty who ruled over this region from 11th to 14th century CE.",
  "normalized_text": "Our district Hasana was the land of the great Hoysala dynasty who ruled over this region from eleventh to fourteenth century C E.",
  "duration": 8.7595,
  "sample_rate": 16000,
  "speaker_profile_id": 9,
  "demo_key": "Female|18-30|Kannada|Karnataka|Hassan|Post Graduate|Full Time|Technology and Services",
  "demographics": {
    "gender": "Female",
    "age-group": "18-30",
    "primary_language": "Kannada",
    "native_place_state": "Karnataka",
    "native_place_district": "Hassan",
    "highest_qualification": "Post Graduate",
    "job_category": "Full Time",
    "occupation_domain": "Technology and Services"
  },
  "source": {
    "dataset": "ai4bharat/Svarah",
    "shard": "test-00000-of-00003.parquet",
    "row_in_shard": 0
  },
  "status": "valid",
  "rejection_reason": null,
  "split": "train"
}
```

### 4.2 Rejected Manifest (`English/metadata/rejected_utterances.json`)
Contains 1,476 structured entries documenting excluded samples.

### 4.3 Training, Validation, and Test Manifests
- `English/metadata/train.txt`: 3,215 valid training utterances (4.9980 hours).
- `English/metadata/val.txt`: 665 valid validation utterances (0.9965 hours).
- `English/metadata/test.txt`: 1,300 valid held-out test utterances (1.9718 hours).

Format (pipe-delimited, single-line guaranteed):
```text
utterance_id|speaker_profile_id|raw_text|normalized_text|duration_s|gender|primary_language
```

---

## 5. Automated Integrity Verification

The manifest was tested using automated unit and integration checks in `tests/test_svarah_manifests.py`:

```bash
.venv/bin/python -m unittest tests/test_svarah_manifests.py
```
- `test_speaker_profiles_integrity`: PASSED (115 contiguous unique IDs 0..114).
- `test_full_manifest_integrity`: PASSED (6,656 unique IDs, non-empty text, valid fields).
- `test_duration_filtering_limits`: PASSED (All 5,180 valid utterances strictly in [1.0s, 15.0s]).
- `test_rejected_manifest_consistency`: PASSED (1,215 under 1s, 261 over 15s).
- `test_profile_leakage_and_split_durations`: PASSED (Zero profile leakage, train ~5.0h, val ~1.0h).

