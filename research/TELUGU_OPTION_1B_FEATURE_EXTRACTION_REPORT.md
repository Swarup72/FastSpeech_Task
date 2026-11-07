# Telugu FastSpeech 2 — Option 1B Acoustic Feature Extraction Report

**Date:** 2026-11-06  
**Dataset:** AI4Bharat IndicTTS Telugu (`Telugu_Option_1B`)  
**Implementation Module:** [`Telugu/alignment/preprocess_features.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/preprocess_features.py)  
**Output Directory:** [`preprocessed_data/Telugu_Option_1B/`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Telugu_Option_1B)  
**Summary Artifact:** [`preprocessed_data/Telugu_Option_1B/extraction_summary.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Telugu_Option_1B/extraction_summary.json)

---

## 1. Executive Summary & Utterance Accounting

We implemented and executed a deterministic, crash-safe, resumable acoustic feature extraction pipeline for the approved **Telugu Option 1B** dataset ([`Telugu/metadata/splits/option_1b/`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b)) using the verified IndicMFA TextGrid alignments ([`data/telugu_mfa_option_1b/TextGrid/`](file:///Users/swaruppal/FastSpeech2_Project/data/telugu_mfa_option_1b/TextGrid)).

### 1.1 Split & Speaker Breakdown

| Split | Option 1B Manifest Utterances | Excluded Failed Alignments | Successfully Extracted Utterances | Female (`spk_000`, ID `0`) | Male (`spk_001`, ID `1`) | Raw Resampled Audio Duration | Silence-Trimmed Active Speech Duration | Total Mel Frames (`T_mel`) | Total Aligned Grapheme/Pause Tokens (`L_phone`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** (`train.txt`) | `2,302` | `2` | **`2,300`** (`99.91%`) | `1,053` | `1,247` | **`19,162.08 s`** (`5.322801 h`) | **`16,495.65 s`** (`4.582124 h`) | `1,421,968` | `202,117` |
| **Validation** (`val.txt`) | `422` | `1` | **`421`** (`99.76%`) | `192` | `229` | **`3,597.18 s`** (`0.999218 h`) | **`3,103.61 s`** (`0.862115 h`) | `267,540` | `38,535` |
| **Combined (`Train + Val`)** | **`2,724`** | **`3`** | **`2,721`** (`99.89%`) | **`1,245`** | **`1,476`** | **`22,759.27 s`** (`6.322019 h`) | **`19,599.26 s`** (`5.444239 h`) | **`1,689,508`** | **`240,652`** |
| **Held-Out Test** (`test.txt`) | `422` | — | **`0` (Strictly Untouched)** | — | — | Untouched | Untouched | Untouched | Untouched |

- **Total `.npy` Feature Arrays Generated:** `16,326` files (`2,721` utterances $\times$ `6` feature directories: `mel/`, `pitch/`, `energy/`, `duration/`, `pitch_phoneme/`, `energy_phoneme/`).
- **Wall-Clock Execution Time:** `158.655 seconds` (`2m 38.7s`) for all `2,721` utterances.
- **Held-Out Test Isolation:** `test_split_accessed = False`. Neither [`Telugu/metadata/splits/option_1b/test.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/test.txt) nor its audio was accessed, read, or processed at any stage.

---

## 2. Acoustic Feature Extraction Configuration & Signal Processing Methodology

All acoustic features were extracted using parameters matched exactly to the project's FastSpeech 2 and HiFi-GAN vocoder configuration:

| Parameter | Value | Implementation Details |
| :--- | :--- | :--- |
| **Target Sampling Rate (`sampling_rate`)** | `22,050 Hz` | Resampled from `16,000 Hz` (Female) and `48,000 Hz` (Male) via Kaiser-windowed sinc interpolation (`beta=14.769656459379492`, `rolloff=0.9475937167399596`, `width=24`) in [`Telugu/alignment/resample.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/resample.py), followed by peak normalization when $\max(|x|) > 1.0$. |
| **Silence Trimming** | `[start_time, end_time]` | Waveform sliced to `[int(start_time * 22050) : int(end_time * 22050)]` using the first and last non-silence phone boundaries from the utterance's IndicMFA TextGrid. |
| **FFT Size (`n_fft`)** | `1024` | Center-padded (`padding = (n_fft - hop_length) // 2 = 384`, `mode="reflect"`) Hann-windowed STFT (`center=False` after manual reflect pad). |
| **Window Length (`win_length`)** | `1024` (`46.44 ms`) | Hann window (`torch.hann_window(1024)`). |
| **Hop Length (`hop_length`)** | `256` (`11.61 ms`) | Frame rate $\approx 86.13\text{ frames/s}$. |
| **Mel Channels (`n_mel_channels`)** | `80` | Slaney-normalized librosa mel filterbank (`fmin=0.0 Hz`, `fmax=8000.0 Hz`) with dynamic range compression $\log(\max(\text{Mel} \cdot |X|, 10^{-5}))$, saved in shape `[T_mel, 80]` (`float32`). |
| **Pitch ($F_0$) Extraction** | `PyWorld DIO + Stonemask` | Extracted at `frame_period = 256 / 22050 * 1000.0 ms` (`11.61 ms`) on `float64` trimmed audio via `pyworld.dio` followed by `pyworld.stonemask` refinement. |
| **Unvoiced ($F_0 = 0$) Frame Handling** | Linear Interpolation | Unvoiced frames where raw $F_0 = 0$ are linearly interpolated across contiguous unvoiced spans between adjacent voiced anchor frames (`scipy.interpolate.interp1d` with constant edge fill `(f0[nonzero[0]], f0[nonzero[-1]])`). This prevents discontinuous zero-drops in the continuous frame-level pitch contour before z-score normalization. |
| **Energy Extraction** | STFT Magnitude $L_2$ Norm | Frame-level energy is computed as the Euclidean ($L_2$) norm across the `513` linear STFT magnitude bins per frame: $E_t = \|\|X_t\|\|_2 = \sqrt{\sum_{k=0}^{512} |X_{k,t}|^2}$. |
| **Phoneme-Level Pitch & Energy** | Duration-Span Mean | Averaged over each token's reconciled mel-frame span `[pos : pos + d]` and stored in `pitch_phoneme/` and `energy_phoneme/` (`[L_phone]`, `float32`). |
| **Duration Extraction & Reconciliation** | Telescoping + Remainder Priority | Interval boundaries are converted to frame indices via `round((t - start_time) * sr / hop_length)` with a strict minimum of `1` frame per token (`min_phone_duration=1`). Any boundary rounding difference between `sum(durations)` and `T_mel` (`-10` to `+1` frames) is reconciled via [`reconcile_durations_with_mel`](file:///Users/swaruppal/FastSpeech2_Project/English/alignment/duration.py#L140-L234) so `sum(duration) == T_mel` holds identically for 100% of utterances while keeping every token duration $\ge 1$. |

---

## 3. Feature Normalization Statistics (`stats.json`)

To prevent any validation or test distribution leakage, all normalization parameters (`mean` and `std`) were fit **strictly on the `2,300` training utterances** after interquartile-range (IQR) outlier filtering ([`remove_outliers`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/preprocess_features.py#L111-L122): keeping values in $[Q_{25} - 1.5 \cdot \text{IQR}, Q_{75} + 1.5 \cdot \text{IQR}]$). Validation features (`421` utterances) were transformed using the frozen training `mean` and `std`.

### 3.1 Saved Parameters in [`preprocessed_data/Telugu_Option_1B/stats.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Telugu_Option_1B/stats.json)

| Feature Domain | Level | Normalized Train Min | Normalized Train Max | Training Fit Mean ($\mu_{\text{train}}$) | Training Fit Std ($\sigma_{\text{train}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Pitch ($F_0$)** | **Frame-Level** (`stats["pitch"]`) | `-2.940123` | `11.886672` | **`195.619853 Hz`** | **`46.419601 Hz`** |
| **Energy ($L_2$ Norm)** | **Frame-Level** (`stats["energy"]`) | `-1.054235` | `8.676959` | **`28.107102`** | **`26.661125`** |
| **Pitch ($F_0$)** | **Phoneme-Level** (`stats["phoneme_level"]["pitch"]`) | — | — | **`197.741307 Hz`** | **`43.766460 Hz`** |
| **Energy ($L_2$ Norm)** | **Phoneme-Level** (`stats["phoneme_level"]["energy"]`) | — | — | **`33.322263`** | **`23.154543`** |

---

## 4. Excluded Recordings & Data Hygiene

### 4.1 Strictly Excluded Failed Alignments (3 Utterances)

Per task requirements and our alignment audit ([`research/TELUGU_OPTION_1B_FULL_ALIGNMENT_REPORT.md`](file:///Users/swaruppal/FastSpeech2_Project/research/TELUGU_OPTION_1B_FULL_ALIGNMENT_REPORT.md)), the 3 recordings with truncated upstream IndicTTS audio were strictly excluded from feature extraction and logged in [`preprocessed_data/Telugu_Option_1B/extraction_failures.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Telugu_Option_1B/extraction_failures.json):

| Utterance ID | Split | Speaker | Exclusion Reason | Upstream Diagnostic |
| :--- | :---: | :---: | :--- | :--- |
| **`train_telugufemale_00930`** | `train` | Female (`0`) | `strictly_excluded_failed_alignment_upstream_truncated_audio` | `13.91s` WAV cut off mid-paragraph (`386` graphemes, `27.7` graphemes/s). |
| **`train_telugufemale_00928`** | `train` | Female (`0`) | `strictly_excluded_failed_alignment_upstream_truncated_audio` | `8.01s` WAV cut off mid-paragraph (`280` graphemes, `34.9` graphemes/s). |
| **`train_telugumale_02390`** | `val` | Male (`1`) | `strictly_excluded_failed_alignment_upstream_truncated_audio` | `2.82s` WAV cut off after first clause (`202` graphemes, `71.7` graphemes/s). |

### 4.2 Context-Sensitive Latin-`o` Normalization Verification

Both Option 1B training utterances containing the upstream Latin-`o` OCR typo (`train_telugumale_01966`: `నడుస్తున్నదేoటి` $\to$ `నడుస్తున్నదేంటి` and `train_telugumale_01981`: `ఊరుకోoడి` $\to$ `ఊరుకోండి`) were verified in [`preprocessed_data/Telugu_Option_1B/train.txt`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Telugu_Option_1B/train.txt) to contain the Telugu Anusvara (`ం`, token ID `70`) in both their curly-brace aligned token sequences and normalized text columns, with zero Latin `o`/`O` characters.

---

## 5. Output Directory Structure & Validation Results

### 5.1 Output Layout under [`preprocessed_data/Telugu_Option_1B/`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Telugu_Option_1B)

```text
preprocessed_data/Telugu_Option_1B/
├── mel/                        # 2,721 files: {spk_str}-mel-{uid}.npy         ([T_mel, 80], float32)
├── pitch/                      # 2,721 files: {spk_str}-pitch-{uid}.npy       ([T_mel], float32, z-scored)
├── energy/                     # 2,721 files: {spk_str}-energy-{uid}.npy      ([T_mel], float32, z-scored)
├── duration/                   # 2,721 files: {spk_str}-duration-{uid}.npy    ([L_phone], int32, >= 1)
├── pitch_phoneme/              # 2,721 files: {spk_str}-pitch-{uid}.npy       ([L_phone], float32, z-scored)
├── energy_phoneme/             # 2,721 files: {spk_str}-energy-{uid}.npy      ([L_phone], float32, z-scored)
├── speakers.json               # {"spk_000": 0, "spk_001": 1}
├── stats.json                  # Training-only pitch & energy normalization statistics
├── train.txt                   # 2,300 lines: uid|spk_str|{tokens}|normalized_text
├── val.txt                     # 421 lines:   uid|spk_str|{tokens}|normalized_text
├── extraction_failures.json    # 3 excluded upstream truncated-audio records
└── extraction_summary.json     # Full split/gender/duration/frame accounting & metadata
```

### 5.2 Per-Utterance Invariant Verification

Every single generated utterance (`2,721 / 2,721`, `100%`) passed [`validate_utterance_feature_files`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/preprocess_features.py#L434-L508) both immediately upon creation and in full post-extraction regression testing:
1. **Zero `NaN` or `Inf` values** across all `16,326` `.npy` arrays (`np.isfinite(arr).all() == True`).
2. **Exact Frame Alignment:** `int(np.sum(duration)) == mel.shape[0] == len(pitch) == len(energy)` for all `2,721` utterances.
3. **Exact Token-Duration Alignment:** `len(duration) == len(text_to_sequence(fastspeech2_text)) == len(pitch_phoneme) == len(energy_phoneme)` for all `2,721` utterances.
4. **Strictly Positive Durations:** `(duration >= 1).all()` across all `240,652` aligned phoneme/pause intervals.
5. **Valid Vocabulary IDs:** Every token in `train.txt` and `val.txt` maps to a valid non-padding, non-`<unk>` symbol in [`Telugu/text/symbols.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/text/symbols.py).

---

## 6. Test Suite Verification

We created [`tests/test_telugu_option1b_features.py`](file:///Users/swaruppal/FastSpeech2_Project/tests/test_telugu_option1b_features.py) with 9 comprehensive tests covering all required dimensions:

| Test Case | Verification Scope | Status |
| :--- | :--- | :---: |
| `test_01_mel_feature_shapes_and_dtypes_all_2721_utterances` | Verifies `[T_mel, 80]` shape, `float32` dtype, `T_mel > 0`, and finiteness across all `2,721` mel files. | **PASS** |
| `test_02_pitch_and_energy_validity_and_unvoiced_interpolation` | Verifies frame/phoneme pitch & energy shapes, finiteness, non-zero variance, and linear interpolation across `f0 == 0` frames. | **PASS** |
| `test_03_duration_consistency_across_all_2721_utterances` | Verifies `sum(duration) == T_mel == len(pitch) == len(energy)` and `len(duration) == len(phoneme_seq)` for all `2,721` utterances, plus reconciliation unit tests. | **PASS** |
| `test_04_nan_inf_and_corruption_detection` | Injects `NaN`, `Inf`, zero durations, and frame mismatches into temporary `.npy` files and confirms [`validate_utterance_feature_files`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/preprocess_features.py#L434-L508) rejects each one. | **PASS** |
| `test_05_manifest_consistency_and_fastspeech2_dataset_compatibility` | Verifies `train.txt` (`2,300`) and `val.txt` (`421`), disjoint IDs, Latin-`o` normalization, and end-to-end batch loading via [`original_fastspeech2/dataset.py`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/dataset.py) (`Dataset`, `TextDataset`, and `DataLoader`). | **PASS** |
| `test_06_normalization_statistics_strictly_from_training_split` | Verifies `stats.json` min/max/mean/std match the `2,300` training utterances (`mean ≈ 0`, `std ≈ 1` after IQR outlier filtering) with zero validation leakage. | **PASS** |
| `test_07_resumable_execution_preserves_valid_files_and_recovers_missing_file` | Verifies re-running `run_telugu_option_1b_preprocessing` skips valid files (`st_mtime_ns` unchanged) and selectively re-extracts a deleted `.npy` file. | **PASS** |
| `test_08_strict_exclusion_of_failed_alignments` | Verifies `train_telugufemale_00930`, `train_telugufemale_00928`, and `train_telugumale_02390` are absent from all manifests/features and logged in `extraction_failures.json`. | **PASS** |
| `test_09_strict_exclusion_of_test_data` | Verifies zero overlap with `test.txt` (`422` IDs) and confirms guardrail exceptions when attempting to load `test.txt`. | **PASS** |

### 6.1 Full Project Regression Suite Summary

- **Telugu Option 1B Feature Extraction Suite** ([`tests/test_telugu_option1b_features.py`](file:///Users/swaruppal/FastSpeech2_Project/tests/test_telugu_option1b_features.py)): **`9 / 9` passed (`5.987s`)**
- **Complete Telugu Test Suite** (`test_telugu*.py`): **`41 / 41` passed (`6.863s`)**
- **Complete Project Regression Suite** (all `224` unit & integration tests across English, Telugu, FastSpeech 2 Model, Variance Adaptor, Loss, Training Engine, and Colab Monitoring): **`224 / 224` passed (`100%` pass rate, zero regressions)**


