# Telugu Full IndicMFA Forced Alignment Report — Option 1B

**Date:** 2026-11-05  
**Dataset Split:** Option 1B ([`Telugu/metadata/splits/option_1b/`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/))  
**Alignment Pipeline Script:** [`Telugu/alignment/align_option_1b.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/alignment/align_option_1b.py)  
**TextGrid Output Directory:** `data/telugu_mfa_option_1b/TextGrid/{train,val}/`  
**Alignment & Duration Metadata Directory:** [`Telugu/metadata/splits/option_1b/alignment/`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/alignment/)  
**Status:** **COMPLETED (`2,721 / 2,724` = `99.8899%` Success) — READY FOR FEATURE EXTRACTION APPROVAL**

---

## 1. Executive Summary

Full IndicMFA forced alignment was executed across the entire approved **Option 1B** training (`2,302` utterances) and validation (`422` utterances) splits (`2,724` utterances total, `6.3289` hours of audio). In strict compliance with project constraints, the held-out `test.txt` split (`422` utterances) was **never aligned, inspected acoustically, or staged**. All existing 50-utterance pilot outputs (`data/telugu_mfa_pilot/`) were preserved untouched.

- **Total Attempted Utterances:** `2,724` (`2,302` Train + `422` Validation)
- **Total Successful Alignments:** **`2,721 / 2,724` (`99.8899%`)**
  - **Training (`train.txt`):** **`2,300 / 2,302` (`99.9131%`)** — `1,053 / 1,055` Female (`99.81%`), **`1,247 / 1,247` Male (`100.00%`)**
  - **Validation (`val.txt`):** **`421 / 422` (`99.7630%`)** — **`192 / 192` Female (`100.00%`)**, `229 / 230` Male (`99.57%`)
- **Total Failed Alignments:** **`3 / 2,724` (`0.1101%`)** (`2` Train, `1` Validation), all recorded with full diagnostics in [`alignment_failures.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/alignment/alignment_failures.json).
- **OOV & Unknown Phonemes:** **`0`** dictionary OOVs, **`0`** 101-symbol vocabulary OOVs, and **`0`** `spn`/`<unk>` intervals across all `2,721` aligned utterances.
- **Mel-Frame Duration Consistency (`sr=22050, hop=256`):** **`100.00%` (`2,721 / 2,721`)** of aligned utterances achieve **zero frame discrepancy (`0` frames)** after telescoping boundary reconciliation (`mean_abs_frame_discrepancy_before_reconciliation = 0.57` frames on Train and `0.66` frames on Validation).

---

## 2. Exact Commands, Configuration & Resource Telemetry

### Commands Executed
```bash
# Pass 1: Full 6-batch alignment across 2,302 train + 422 val utterances (default beam: 10, retry_beam: 40)
.venv/bin/python -m Telugu.alignment.align_option_1b --batch_size 461 --num_jobs 6

# Pass 2: Crash-safe resume pass with wide-beam MFA config (beam: 200, retry_beam: 800) on the 25 initial dropouts
.venv/bin/python -m Telugu.alignment.align_option_1b --batch_size 461 --num_jobs 4
```

### Alignment Configuration
| Parameter | Value |
| :--- | :--- |
| **MFA Binary** | `/opt/anaconda3/envs/mfa_env/bin/mfa` (Montreal Forced Aligner v3.x) |
| **Acoustic Model** | `data/telugu_mfa_model/Telugu_Acoustic_Model.zip` (AI4Bharat IndicMFA Telugu) |
| **Pronunciation Dictionary** | [`research/Telugu_Dictionary_g2g.txt`](file:///Users/swaruppal/FastSpeech2_Project/research/Telugu_Dictionary_g2g.txt) (Letterwise Grapheme-to-Grapheme) |
| **Audio Resampling** | Kaiser-windowed sinc (`16,000 Hz` Female & `48,000 Hz` Male $\to$ `22,050 Hz` 16-bit PCM mono) |
| **MFA CLI Flags** | `mfa align --clean --use_mp -j 6 --single_speaker --no_textgrid_cleanup` |
| **Wide-Beam Retry Config** | `beam: 200`, `retry_beam: 800` via `--config_path wide_beam_config.yaml` |
| **Batch Size** | `461` utterances/batch (`5` Train batches + `1` Validation batch) |
| **Pass 1 Runtime (`2,724` utts)** | `363.05s` (`6.05` min wall-clock; `279.99s` MFA engine; `~0.13s/utt`) |
| **Pass 2 Resume Runtime (`25` utts)** | `191.59s` (`3.19` min wall-clock; `61.75s` MFA wide-beam retry; skipped `2,699` completed utts) |
| **Peak Process RAM Usage** | `1,260.97 MB` (`~1.23 GB`) |

---

## 3. Separate Training and Validation Alignment Results

| Metric | Training Split (`train.txt`) | Validation Split (`val.txt`) | Combined (`train + val`) | Held-Out Test Split (`test.txt`) |
| :--- | :---: | :---: | :---: | :---: |
| **Attempted Utterances** | **2,302** | **422** | **2,724** | **0 (Strictly Excluded)** |
| **Successful Alignments** | **2,300** | **421** | **2,721** | — |
| **Failed Alignments** | **2** | **1** | **3** | — |
| **Alignment Success Rate** | **99.9131%** | **99.7630%** | **99.8899%** | — |
| **Female Succeeded / Attempted** | `1,053 / 1,055` (**99.81%**) | `192 / 192` (**100.00%**) | `1,245 / 1,247` (**99.84%**) | — |
| **Male Succeeded / Attempted** | `1,247 / 1,247` (**100.00%**) | `229 / 230` (**99.57%**) | `1,476 / 1,477` (**99.93%**) | — |
| **Total Audio Duration** | `5.328904 h` | `1.000003 h` | `6.328907 h` | — |
| **Aligned Speech Duration** | `3.460333 h` | `0.648893 h` | `4.109226 h` | — |
| **Aligned Silence Duration** | `1.862481 h` | `0.350328 h` | `2.212809 h` | — |
| **Dictionary OOV Tokens** | **`0`** | **`0`** | **`0`** | — |
| **Vocabulary (101-Symbol) OOV Tokens** | **`0`** | **`0`** | **`0`** | — |
| **Unknown (`spn` / `<unk>`) Intervals** | **`0`** | **`0`** | **`0`** | — |
| **Mean Pre-Reconciliation Frame Diff** | `0.5709` frames | `0.6556` frames | `0.5840` frames | — |
| **Max Post-Reconciliation Frame Diff** | **`0` frames** (`2,300/2,300`) | **`0` frames** (`421/421`) | **`0` frames** (`2,721/2,721`) | — |
| **Min / Max Speech Token Frames** | `1` / `40` frames | `1` / `77` frames | `1` / `77` frames | — |

---

## 4. Failure Diagnosis & Wide-Beam Recovery Status

1. **Initial Pass (`beam: 10, retry_beam: 40`)**:
   - `2,699 / 2,724` utterances (`99.08%`) aligned on the first pass; `25` utterances (`17` Train, `8` Val) exceeded the default narrow beam width (`10`/`40`).
2. **Wide-Beam Retry Recovery (`beam: 200, retry_beam: 800`)**:
   - **`22 / 25` (`88.0%`)** of the initial beam dropouts recovered cleanly with valid, monotonic TextGrids and zero OOVs (`15` in Train, `7` in Val).
3. **Root Cause of the Remaining 3 Unrecoverable Utterances (Upstream Corpus Truncated Audio)**:
   - All 3 remaining failures are **upstream IndicTTS corpus recording truncations** where the transcript contains a full multi-sentence paragraph (`202`–`386` graphemes) while the WAV audio recording was cut off mid-sentence during studio recording:

| Utterance ID | Split | Speaker | WAV Duration | Transcript Graphemes | Implied Grapheme Rate | Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`train_telugufemale_00930`** | `train` | Female (`0`) | `13.91s` | **386 graphemes** (`449` chars) | **`27.7` graphemes/s** (`3.1×` normal Female rate of `8.8/s`) | Upstream WAV cut off mid-paragraph; second half of transcript is absent from audio. |
| **`train_telugufemale_00928`** | `train` | Female (`0`) | `8.01s` | **280 graphemes** (`322` chars) | **`34.9` graphemes/s** (`4.0×` normal Female rate of `8.8/s`) | Upstream WAV cut off at `8.01s`; ~20 seconds of paragraph text missing from audio. |
| **`train_telugumale_02390`** | `val` | Male (`1`) | `2.82s` | **202 graphemes** (`242` chars) | **`71.7` graphemes/s** (`4.7×` normal Male rate of `15.2/s`) | Upstream WAV contains only the first clause (`2.82s`) of a 242-character paragraph (`202` phones require $\ge 6.06\text{s}$ minimum 3-state HMM traversal). |

Because Kaldi/MFA enforces a minimum 3-frame (`30 ms`) HMM state traversal per phoneme, IndicMFA rightly refused to force-fit `202`–`386` graphemes into truncated audio. All 3 records are preserved in [`alignment_failures.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/alignment/alignment_failures.json) and excluded from feature extraction so corrupt audio-text pairs never contaminate training.

---

## 5. Manual Quality Inspection of Representative Aligned Samples

We inspected 6 representative utterances across both splits and speakers (including the 2 context-sensitive Latin-`o` $\to$ Anusvara `ం` corrected utterances in Option 1B `train`):

| Sample Category | Utterance ID (`Split`, `Gender`) | Audio Dur | Leading / Trailing Silence | Speech / Pause Tokens | Expected Trimmed Mel Frames | Raw Frame Sum $\to$ Reconciled Sum | First 10 Extracted `(Token, Mel Frames)` Pairs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Latin-`o` Corrected #1** (`నడుస్తున్నదేoటి` $\to$ `నడుస్తున్నదేంటి`) | `train_telugumale_01966` (`train`, `male`) | `5.963s` | `0.490s` / `0.333s` | `72` / `7` | `443` | `443` $\to$ **`443`** (`diff=0`) | `[('మ', 7), ('స', 9), ('ీ', 6), ('ద', 4), ('ు', 2), ('ల', 6), ('ో', 7), ('ఎ', 10), ('వ', 8), ('ర', 3)]` |
| **Latin-`o` Corrected #2** (`ఊరుకోoడి` $\to$ `ఊరుకోండి`) | `train_telugumale_01981` (`train`, `male`) | `14.670s` | `0.480s` / `0.480s` | `164` / `14` | `1181` | `1185` $\to$ **`1181`** (`diff=-4` $\to$ `0`) | `[('స', 10), ('హ', 11), ('ద', 6), ('ే', 7), ('వ', 6), ('ు', 2), ('డ', 5), ('ు', 2), ('అ', 8), ('న', 4)]` |
| **Train Female Representative** | `train_telugufemale_01230` (`train`, `female`) | `5.297s` | `0.630s` / `0.737s` | `45` / `7` | `339` | `339` $\to$ **`339`** (`diff=0`) | `[('న', 6), ('ీ', 8), ('క', 8), ('ు', 2), ('గ', 7), ('@sp', 5), ('ు', 2), ('డ', 12), ('్', 2), ('ల', 5)]` |
| **Train Male Representative** | `train_telugumale_03090` (`train`, `male`) | `6.421s` | `0.950s` / `1.051s` | `67` / `3` | `381` | `381` $\to$ **`381`** (`diff=0`) | `[('మ', 7), ('ు', 2), ('ఖ', 12), ('్', 1), ('య', 4), ('మ', 6), ('ై', 6), ('న', 7), ('స', 6), ('్', 2)]` |
| **Val Female Representative** | `train_telugufemale_00905` (`val`, `female`) | `6.783s` | `0.490s` / `0.973s` | `57` / `9` | `459` | `459` $\to$ **`459`** (`diff=0`) | `[('మ', 5), ('ి', 3), ('గ', 1), ('ి', 3), ('ల', 4), ('ి', 13), ('న', 8), ('వ', 9), ('ా', 7), ('ళ', 6)]` |
| **Val Male Representative** | `train_telugumale_00123` (`val`, `male`) | `5.744s` | `0.660s` / `0.604s` | `60` / `6` | `386` | `386` $\to$ **`386`** (`diff=0`) | `[('స', 9), ('@sp', 2), ('ి', 4), ('ద', 1), ('్', 6), ('ధ', 5), ('ా', 8), ('ం', 5), ('త', 10), ('ం', 15)]` |

---

## 6. Versioned Output Files & Regression Test Results

### Generated Versioned Option 1B Alignment Artifacts
- `data/telugu_mfa_option_1b/TextGrid/train/*.TextGrid` (`2,300` verified TextGrid files)
- `data/telugu_mfa_option_1b/TextGrid/val/*.TextGrid` (`421` verified TextGrid files)
- [`Telugu/metadata/splits/option_1b/alignment/alignment_summary.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/alignment/alignment_summary.json)
- [`Telugu/metadata/splits/option_1b/alignment/alignment_progress.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/alignment/alignment_progress.json)
- [`Telugu/metadata/splits/option_1b/alignment/alignment_failures.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/alignment/alignment_failures.json) (`3` upstream truncated-audio records)
- [`Telugu/metadata/splits/option_1b/alignment/duration_metadata_train.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/alignment/duration_metadata_train.json) (`2,302` records)
- [`Telugu/metadata/splits/option_1b/alignment/duration_metadata_val.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/splits/option_1b/alignment/duration_metadata_val.json) (`422` records)

### Telugu Regression Suite Results
Executed `.venv/bin/python -m unittest discover -s tests -p "test_telugu*.py" -v`:
- **`32 / 32` tests passed (`Ran 32 tests in 0.986s — OK`)**, including `TestTeluguOption1BFullAlignment` (`test_01_alignment_manifests_and_test_split_isolation`, `test_02_train_and_val_alignment_counts_and_failure_accounting`, `test_03_duration_frame_consistency_across_all_2721_aligned_utterances`).

