# Svarah Ingestion Smoke Test (Step 1.4J-B1)

**Date**: 4 November 2026
**Scope**: Authenticated access, full metadata inspection (no audio), download of a 25-utterance sample, audio checks, a 16 → 22.05 kHz resampling test, and a speaker-mapping test.
**Not done**: full preprocessing, MFA alignment, feature extraction, training.
**Code changes**: None to `common/` or `original_fastspeech2/`. Regression suite: **131/131 pass**.

All numbers here come from scripts run on real Svarah data unless marked otherwise. Scripts and outputs are in `data/svarah_smoke_test/` (excluded from Git).

> [!WARNING]
> **This report corrects several statements in the earlier Svarah reports.** Those reports were written before authenticated access and relied partly on estimates that turned out to be wrong. See §10 for the full list. In particular, the dataset **has no speaker-ID column and no speech-category column**.

---

## 1. Authentication result

| Check | Result |
| :--- | :--- |
| `HF_TOKEN` in the agent's shell | Not exported (it is set only in your interactive Terminal session) |
| Cached `hf auth login` token | Present; used implicitly by `huggingface_hub` |
| `HfApi().whoami()` | User account authenticated (username not repeated here) |
| `dataset_info("ai4bharat/Svarah")` | Accessible, `gated: auto`, revision `ebbf7777fe771490696a3f7b007097606fa8c924` |
| Token handling | Never read, printed, logged or written by any script |

All downloads are pinned to the revision above for reproducibility.

## 2. Actual dataset schema and splits

- **Configs / splits**: one config (`default`) with one split, **`test`**: **6,656 rows**. There is no official train/val split, so we must create our own.
- **Files**: 3 Parquet shards (`data/test-0000{0,1,2}-of-00003.parquet`), 1.095 GB in total, each with 23 row groups of about 96 rows.
- **Columns**:

| Column | Type | Notes |
| :--- | :--- | :--- |
| `audio_filepath` | struct{`bytes`, `path`} | Embedded audio; `path` looks like `281474976884635_f3269_chunk_0.wav` |
| `duration` | float64 | Seconds; matches decoded length exactly on all 25 samples |
| `text` | string | Original transcript (mixed case, punctuation, digits) |
| `gender`, `age-group`, `primary_language`, `native_place_state`, `native_place_district`, `highest_qualification`, `job_category`, `occupation_domain` | string | Speaker demographics |

**Not present**: speaker ID and speech category (read / extempore / use-case).

**Metadata quality** (all 6,656 rows, read without downloading audio bytes):
- No missing values in any column. No empty transcripts. No duplicate audio paths.
- Total duration: **9.614 h** (34,610 s). Mean utterance length is 5.20 s, median 4.21 s, range 0.098–29.87 s.
- 1,215 utterances are shorter than 1 s, and 261 are longer than 15 s (110 longer than 20 s). Keeping only 1–15 s utterances leaves **5,180 utterances / 7.966 h**.
- 3,447 utterances have `chunk_N` with N > 0, meaning they are segments of longer recordings. 1,429 have two words or fewer, so many are sentence fragments.
- Gender by utterance: 3,579 Female / 3,077 Male.

### 2.1 Speaker identity: what can actually be derived

| Candidate key | Unique values | Verdict |
| :--- | :---: | :--- |
| Filename prefix 1 (`281474976884635`) | 2,938 | Recording/session ID, not a speaker |
| Filename prefix 2 (`f3269`) | 3,232 | Recording/file ID, not a speaker |
| **Full 8-field demographic profile** | **115** | **Best available proxy** (the paper reports 117 speakers) |

Every filename ID maps to exactly one demographic profile, so the profile key is internally consistent. It is still a **proxy**. 115 ≠ 117 implies at least two pairs of real speakers have identical demographic profiles and get merged. The effect is small (≤ 2 merged pairs) but must be documented.

### 2.2 Speaker-wise durations (proxy speakers, full metadata)

- **115 proxy speakers**: 62 Female, 53 Male.
- Minutes per speaker: mean 5.02, median 3.78, min 0.30, max 24.53, std 4.05. The distribution is strongly skewed: 20 speakers have under 2 min and 11 have over 10 min.
- Utterances per speaker: mean 57.9, median 47, min 3, max 184.
- 19 L1 languages. Hours by language: Nepali 1.163, Maithili 0.762, Kannada 0.711, Malayalam 0.675, Urdu 0.636, Bodo 0.626, Odia 0.611, Konkani 0.537, Telugu 0.504, Tamil 0.444, Dogri 0.438, Kashmiri 0.399, Hindi 0.398, Gujarati 0.369, Bengali 0.334, Marathi 0.297, Punjabi 0.270, Assamese 0.256, Sindhi 0.183.
- Full table: `data/svarah_smoke_test/proxy_speaker_durations.csv`.

### 2.3 Split feasibility on actual durations (probe, not final)

A deterministic probe (seeded) built a speaker-disjoint split:
- **Validation**: 12 speakers, **1.000 h**, 6 F / 6 M, 12 different L1s.
- **Training**: 65 speakers, **5.013 h**, chosen round-robin by language from the remaining speakers.
- **Unused**: 38 speakers, 3.602 h (available as a held-out test set).
- Train/val speaker overlap: **0**.

**Conclusion**: a speaker-independent ~5 h / ~1 h split is feasible with the proxy speaker key. The earlier "62/12/43" figure was an estimate. The real counts depend on which speakers are chosen, because per-speaker duration varies by 80×. The probe has not yet applied the 1–15 s duration filter. With the filter, the usable pool falls to 7.97 h, which is still enough for 5 h + 1 h, but there is less spare.

**Speech category** cannot be stratified because it is not in the metadata. A heuristic (e.g. "ends with `?`" or imperative verbs → use-case) could be added later but would be unverified.

## 3. Sample utterance metadata

- **25 utterances**, read from **only 2 row groups** of shard 0 (200 rows, about 32 MB transferred). Stored audio: 5.1 MB. Total directory: 10 MB.
- Selection was deterministic: alternating genders, one utterance per proxy speaker, extra utterances from one speaker (for mapping repeats), plus the shortest and longest rows as edge cases.
- Each row keeps the original transcript, all demographic fields, repo/revision/shard/row index, original filename and SHA-256. Files: `sample_manifest.csv`, `PROVENANCE.json`.
- Composition: **14 F / 11 M**, **21 proxy speakers**, **15 L1 languages**, durations 0.17–25.49 s.
- Speech category is unknown (see §2.3). Judging by the content, the sample clearly mixes use-case commands ("Stop processing the payment for Amazon", "Show my DTH bill readings from the last 6 months."), read/Wikipedia-style sentences, and extempore fragments ("The main food of", "normal competitions what we have.").

Transcript issues seen in the sample that the frontend will need to handle (full-corpus counts in brackets):
- Digits and ordinals: "15-17 years", "11th to 14th century", "100%", "4 October" [769 utterances contain digits; 38 contain `%`, `₹`, `$` or `&`].
- Acronyms: DTH, RSS, CE, COVAXIN [433 utterances contain an all-caps token].
- Indian proper nouns and loanwords: Hoysala, Chaitra Dasai, vandalaga, dodapathre, thambuli, shakhas.
- Fragments and single words: "Off" (0.17 s), "The main food of".

## 4. Speaker information (sample)

21 proxy speakers. One Kashmiri female speaker contributes 2 utterances. The deterministic mapping (§7) assigns IDs 0–20. Demographic fields are kept per utterance in `sample_manifest.csv`.

## 5. Audio inspection (all 25 decoded with `soundfile`)

| Property | Result |
| :--- | :--- |
| Container / encoding | **WAV, PCM_16** (all 25), embedded as bytes in Parquet |
| Sample rate | **16,000 Hz** (all 25) |
| Channels | **1 (mono)** (all 25) |
| Decoded vs metadata duration | Identical (difference 0.0000 s for all 25) |
| Corrupted / undecodable | 0 |
| NaN/Inf | 0 |
| Silent (peak < 1e-4) | 0 |
| Clipping (abs ≥ 0.999) | 1 file (`281474976894291_f2897_chunk_0`, 4 samples, peak 1.0) |
| Level | RMS from −14.9 to −37.0 dBFS; peak 0.12–1.00 (about 22 dB loudness spread) |

Per-file table: `audio_inspection.csv`. Edge cases worth noting: a 0.17 s single-word clip and a 25.5 s clip. Both would be removed by a 1–15 s filter.

## 6. Resampling results (5 files, 16 kHz → 22.05 kHz)

Method: `torchaudio.functional.resample`, windowed-sinc with a Kaiser window (`lowpass_filter_width=64`, `rolloff≈0.9476`, `beta≈14.77`, the "kaiser_best" setting). Output written as 16-bit WAV.

| Check | Result |
| :--- | :--- |
| Output sample rate | 22,050 Hz (all 5) |
| Output length | Exactly `ceil(N·22050/16000)` samples (all 5) |
| Duration preserved | Within 0.03 ms (all 5) |
| Energy above 8 kHz in output | −59 to −88 dB relative to total, so no spurious content is added |
| Round-trip (16 k → 22.05 k → 16 k) SNR | 25.97–55.81 dB |
| Clipping in output | 0 in the tested 5 files |

**Interpreting the round-trip SNR**: the spread does not indicate corruption. For every file, the SNR matches the ceiling predicted by the source's own energy above 7.58 kHz (the low-pass rolloff point): predicted 56.3 / 32.0 / 26.1 dB vs measured 55.8 / 31.1 / 26.0 dB. The resampler removes the 7.58–8 kHz band, and that band is what the round trip loses. This is negligible for TTS because the HiFi-GAN mel range already ends at `fmax=8000`.

**Overshoot finding**: the already-clipped file #5 reaches a float peak of **1.0039** after resampling (4 samples > 1.0), which is normal sinc ringing. **Preprocessing must peak-normalise (e.g. to 0.95) or clamp before 16-bit writing / STFT.** The original repo already normalises with `wav / max(|wav|) * max_wav_value`.

**Limitation**: resampling only changes the sample grid. The 22.05 kHz audio still contains **no information above 8 kHz**, so nothing is restored. This is acceptable only because the vocoder's mel filterbank stops at 8 kHz. The 8–11.025 kHz band of the STFT will be empty, unlike LJSpeech-style training data.

## 7. Speaker mapping verification (sample only)

- Key: the full demographic profile string. Mapping: `sorted(unique_keys) → 0..N-1`.
- Result: **21 speakers → IDs 0..20**. Unique ✔, contiguous ✔, and the same after shuffling input order ✔ (asserted in the script).
- Saved as `speakers_smoke_test.json` (sample only; **not** the final mapping).
- The final 115-speaker mapping should be built from the full metadata with the same rule. Note that `n_speaker` would then be **115**, not 117.

## 8. Remaining blockers and open decisions

1. **Speaker identity is a proxy** (115 profiles vs 117 speakers). You need to decide whether to accept this (recommended: small effect, document it) or look for an external speaker key.
2. **No speech-category labels**: we cannot stratify by category or report category hours from the data itself. The 1.4 / 6.4 / 1.7 h figures come from the paper only.
3. **Duration filter decision**: 1–15 s gives 7.97 h usable. The impact of fragments (≤ 2 words) on alignment quality is not yet measured.
4. **Text normalisation** is needed for numbers, ordinals, symbols and acronyms before G2P/MFA.
5. **MFA is not installed locally** (needs conda-forge; plan to use Colab or a local conda env). No Svarah alignment has been attempted yet.
6. **Loudness variation (~22 dB RMS spread)**: plan peak normalisation; consider loudness normalisation.
7. `HF_TOKEN` is not exported to non-interactive shells. The cached login works, so this is a blocker only for environments without the cache (e.g. Colab needs `HF_TOKEN` set as a secret).

## 9. Recommended next step (Step 1.4J-B2)

Implement a **metadata + split builder only** (still no audio feature extraction):
1. Build the 115-proxy-speaker table and deterministic `speakers.json` from full metadata.
2. Apply the agreed duration filter, then produce a seeded, speaker-disjoint, gender- and L1-balanced split with a ~1 h val set and ~5 h train set, and export `train.txt` / `val.txt` manifests with provenance.
3. Write a text-normalisation module (numbers, ordinals, %, ₹, acronyms) with unit tests on real Svarah transcripts.
4. Then run MFA on a ~100-utterance subset (Colab/conda) to measure OOV rate and alignment failure rate before scaling up.

## 10. Corrections to earlier Svarah reports

| Earlier claim | Verified fact |
| :--- | :--- |
| 9.66 h total | **9.614 h** (sum of `duration`) |
| 117 speakers with speaker IDs | **No speaker-ID column**; 115 distinguishable demographic profiles |
| Audio is 16 kHz FLAC | **16 kHz mono WAV PCM_16** embedded in Parquet |
| Accent table incl. Sanskrit / Santali with per-speaker estimates | **Not supported by the metadata**. Actual L1s include Urdu and Kashmiri; there is no Sanskrit or Santali. Hours per language are in §2.2. |
| Speech-category split derivable | Category is **not in metadata** |
| 62 / 12 / 43 split | Estimate only; the probe gives 65 / 12 / 38 (proxy speakers) |
| Colab T4 epoch times | **Extrapolations, not measurements**. Only the local MPS/CPU synthetic numbers (1.25 s/step at BS=16 on MPS) were measured. |

