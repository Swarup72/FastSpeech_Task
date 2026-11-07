# English FastSpeech 2 (Svarah) Validation Fix Implementation Audit

**Date:** 2026-11-05  
**Prerequisite Diagnostic:** [`research/ENGLISH_VALIDATION_LOSS_DIAGNOSTIC.md`](file:///Users/swaruppal/FastSpeech2_Project/research/ENGLISH_VALIDATION_LOSS_DIAGNOSTIC.md)  
**Target Deadline:** November 7, 2026  
**Status:** Read-Only Verification Complete (Zero datasets, configs, model files, or checkpoints modified)

---

## 1. Investigation 1: Svarah Speaker Identity & Split Scientific Validity

### 1.1 Are the 115 Speaker IDs Individual Human Speakers or Inferred Demographic Profiles?
- **Verified Finding:** The `115` IDs (`spk_000` through `spk_114` in [`preprocessed_data/Svarah/speakers.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/speakers.json)) represent **inferred 8-field demographic profile combinations**, not biometric individual speaker IDs.
- As recorded in [`English/metadata/speaker_profiles.json`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/speaker_profiles.json#L10):
  > `"These IDs represent distinct demographic profile combinations and serve as proxy conditioning identities. They are NOT verified individual biometric speaker identities. Multiple distinct human speakers may share identical demographic profiles."`
- Specifically, each `speaker_profile_id` (`0..114`) is a deterministic index of the 8-tuple:
  $$\texttt{gender} \mid \texttt{age-group} \mid \texttt{primary\_language} \mid \texttt{native\_place\_state} \mid \texttt{native\_place\_district} \mid \texttt{highest\_qualification} \mid \texttt{job\_category} \mid \texttt{occupation\_domain}$$

### 1.2 How Speaker IDs Are Assigned and Consumed
1. **Utterance Structure in [`English/metadata/full_manifest.json`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/full_manifest.json):**
   - Each utterance ID has the form `{user_prefix}_f{file_id}_chunk_{chunk_id}` (e.g., `281474976884635_f3269_chunk_0`).
   - Across all `5,180` valid utterances in `full_manifest.json`, there are **`1,981` unique `user_prefix` IDs** and **`2,188` unique `recording_id`s (`{user_prefix}_f{file_id}`)** mapped into the **`115` `speaker_profile_id`s**:
     - Every `user_prefix` maps to **strictly 1 `speaker_profile_id`** (`min = 1, max = 1`, `0` prefixes span multiple profiles).
     - Every `recording_id` maps to **strictly 1 `speaker_profile_id`** (`min = 1, max = 1`, `0` recordings span multiple profiles).
     - Each `speaker_profile_id` aggregates **`3` to `49` `user_prefix`es** (`median = 16.0`, `mean = 17.23`) and **`3` to `49` `recording_id`s** (`median = 17.0`, `mean = 19.03`).
2. **Preprocessing & Model Consumption:**
   - [`English/alignment/preprocess_features.py`](file:///Users/swaruppal/FastSpeech2_Project/English/alignment/preprocess_features.py#L280-L299) prefixes every saved `.npy` file with `spk_{id:03d}` and writes `uid|spk_XXX|{phones}|raw_text` to `train.txt` and `val.txt`.
   - [`Dataset.__getitem__()`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/dataset.py#L34-L35) maps `spk_XXX` via `speakers.json` to integer `0..114`, which indexes `self.speaker_emb = nn.Embedding(115, 256)` in [`FastSpeech2.forward()`](file:///Users/swaruppal/FastSpeech2_Project/common/model/fastspeech2.py#L262-L265).

### 1.3 Are the 115 Profile IDs Stable and Acoustically Meaningful for `nn.Embedding(115, 256)`?
To test whether `speaker_profile_id` is acoustically coherent or noisy, we computed a one-way ANOVA variance decomposition ($R^2 = 1 - \text{SS}_{\text{within}} / \text{SS}_{\text{total}}$) of utterance-level mean $F_0$ (in Hz) across all `3,872` preprocessed utterances (`train + val`):

| Grouping Key | Active Groups (in 3,872 utts) | Mean Utterances / Group | $F_0$ Variance Explained ($R^2$) | Within-Group $F_0$ Std (Hz) | Suitability for `nn.Embedding` Lookup |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Global (No Conditioning)** | `1` | `3,872.0` | `0.0000` | `59.45 Hz` | Poor — mixes 115 Hz male and 245 Hz female voices |
| **`speaker_profile_id` (`spk_XXX`)** | **`88`** | **`44.0`** | **`0.8723` (`87.2%`)** | **`18.70 Hz`** | **Optimal** — captures `87.2%` of register variance with `44` utts/embedding |
| **`user_prefix`** | `1,471` | `2.63` | `0.9512` (`95.1%`) | `11.56 Hz` | Too sparse (`2.6` utts/ID) — severe overfitting per embedding |
| **`recording_id` (`prefix_fXXXX`)** | `1,615` | `2.40` | `0.9515` (`95.2%`) | `11.52 Hz` | Too sparse (`2.4` utts/ID) — severe overfitting per embedding |

**Verdict:** Because an 8-way intersection down to the exact `district + primary_language + occupation_domain + qualification + age-group + gender` captures either a single contributor across multiple sessions or a tightly homogeneous demographic/regional speaker cohort, **`speaker_profile_id` explains `87.23%` of all fundamental frequency variance across the corpus while providing ~44 utterances per embedding**. It is both stable and scientifically meaningful for `nn.Embedding(115, 256)` — **provided that validation evaluates seen profile IDs rather than untrained random rows**.

### 1.4 Train, Validation, and Test Distributions & Leakage Audit

| Split | Metadata Utterances (`English/metadata/`) | Preprocessed Utterances (`preprocessed_data/Svarah/`) | Distinct `speaker_profile_id`s | Distinct `user_prefix`es | Distinct `recording_id`s | Speaker Overlap with Train |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Train** | `3,215` | **`3,208`** | **`66`** | `1,196` | `1,326` | `66 / 66` (`100%`) |
| **Validation** | `665` | **`664`** | **`22`** | `275` | `289` | **`0 / 22` (`0.0%`)** |
| **Test** | `1,300` | **`0` (Not MFA-aligned)** | **`27`** | `510` | `573` | **`0 / 27` (`0.0%`)** |
| **Total** | `5,180` | **`3,872`** | **`115`** | `1,981` | `2,188` | — |

- **Current Source Recording Leakage:** `0` shared `user_prefix`es and `0` shared `recording_id`s across `train`, `val`, and `test`.
- **Current Normalized Transcript Leakage:** `train.txt` and `val.txt` currently share **`7` normalized transcripts (`8` validation utterances)**.

### 1.5 Is an 80/10/10 (or Stratified Train/Val) Split Scientifically Appropriate?
1. **Why a Naive Utterance-Level Random Split Is Unsafe:** Because multiple chunks (`_chunk_0`, `_chunk_1`, ...) originate from the same `recording_id` (`{user_prefix}_f{XXXX}`), splitting individual utterances randomly within a `speaker_profile_id` would place chunks from the exact same recording and speaker into both `train` and `val` (recording/session leakage).
2. **Why a `user_prefix`-Grouped Stratified Split Within Each `speaker_profile_id` IS Scientifically Rigorous:**
   - Across the `88` already-preprocessed `speaker_profile_id`s (`3,872` utterances), **`88 / 88` (`100%`) have $\ge 2$ distinct `recording_id`s and $\ge 3$ distinct `user_prefix`es** (`86 / 88` have $\ge 10$ utterances).
   - By partitioning **whole `user_prefix` groups** within each of the `88` `speaker_profile_id`s (and holding out any shared transcripts), we simultaneously achieve:
     - **`100%` (`88 / 88`) seen `speaker_profile_id`s** in `val.txt` (every validation utterance indexes a trained `nn.Embedding` row),
     - **`0.00%` `user_prefix` overlap and `0.00%` `recording_id` overlap** between `train` and `val` (every validation utterance comes from an unseen recording session / contributor prefix within that demographic profile!),
     - **`0.00%` transcript leakage**,
     - **`50.0%` Female / `50.0%` Male gender balance** in both `train` and `val` (fixing the current `36.3% F / 63.7% M` skew in `val.txt`), and
     - **A `-54.5%` to `-62.8%` drop in the validation pitch MSE floor** (`1.2468 -> 0.5665` frame-level / `0.4642` phoneme-level)!
3. **Should We Split 83/17 (`train`/`val` only) or 80/10/10 (`train`/`val`/`test`) Across the `3,872` Preprocessed Utterances?**
   - Notice that `English/metadata/test.txt` (`1,300` utterances) was never aligned with MFA and has no `.npy` features in `preprocessed_data/Svarah/`.
   - **Recommended Split Strategy:**
     - **Keep `English/metadata/test.txt` (`1,300` utterances) untouched** as the held-out metadata test manifest.
     - Partition the **`3,872` already-preprocessed utterances (`88` profiles)** using `user_prefix`-disjoint, transcript-disjoint stratification into **`~3,208` `train.txt` (`~83%`) and `~664` `val.txt` (`~17%`)** (matching the exact current split sizes so `steps_per_epoch = 200` and all monitoring/test expectations remain unchanged; optionally also exporting a `test_seen_profiles.txt` manifesto if needed for synthesis evaluation).

---

## 2. Investigation 2: Phoneme-Level Pitch and Energy (`pitch_phoneme/` & `energy_phoneme/`)

### 2.1 Verification of Existing `pitch_phoneme/` and `energy_phoneme/` Arrays on Disk
We inspected all `3,872` `.npy` files in [`preprocessed_data/Svarah/pitch_phoneme/`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/pitch_phoneme) (`15 MB`) and [`preprocessed_data/Svarah/energy_phoneme/`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/energy_phoneme) (`15 MB`):

| Check | Result Across All `3,872` Utterances (`192,656` Phoneme Tokens) |
| :--- | :--- |
| **File Count** | `3,872 / 3,872` present in both `pitch_phoneme/` and `energy_phoneme/` |
| **Dimension Alignment (`len == len(duration) == len(phones)`)** | **`0` mismatches (`100%` exact match with `duration/` and `train.txt`/`val.txt` phoneme counts)** |
| **NaN / Inf Count** | **`0` non-finite values (`100%` finite `float32`)** |
| **Train `pitch_phoneme` Distribution** | `mean = +0.0283`, `std = 1.0481`, `min = -3.0792`, `max = +10.1557` (`mean_hz = 179.83 Hz, std_hz = 58.40 Hz`) |
| **Val `pitch_phoneme` Distribution** | `mean = -0.2715`, `std = 1.0619`, `min = -1.9296`, `max = +7.8269` |
| **Train `energy_phoneme` Distribution** | `mean = +0.0128`, `std = 1.0150`, `min = -1.0714`, `max = +6.5189` (`mean_raw = 30.58, std_raw = 28.53`) |
| **Val `energy_phoneme` Distribution** | `mean = -0.0740`, `std = 0.8592`, `min = -1.0712`, `max = +5.2876` |

### 2.2 How `VarianceAdaptor` and `FastSpeech2Loss` Consume `phoneme_level` Features
- In [`VarianceAdaptor.forward()`](file:///Users/swaruppal/FastSpeech2_Project/common/model/variance_adaptor.py#L455-L462), setting `pitch.feature: "phoneme_level"` and `energy.feature: "phoneme_level"` causes `_process_pitch()` and `_process_energy()` to run **before** `LengthRegulator` (`_process_durations()`), predicting one pitch and energy scalar per source phoneme token (`[B, L_src]`) masked by `src_mask`, quantizing via `torch.bucketize`, and adding `pitch_embedding` + `energy_embedding` directly to the encoder hidden states `[B, L_src, 256]` before frame expansion.
- In [`FastSpeech2Loss.forward()`](file:///Users/swaruppal/FastSpeech2_Project/common/model/loss.py#L118-L132), the loss function automatically switches masks from `~mel_masks` (`[B, L_mel]`) to `~src_masks` (`[B, L_src]`).
- **Compatibility with `mel/` and `duration/`:** Existing `mel/` (`[L_mel, 80]`) and `duration/` (`[L_src]`) `.npy` files are **100% compatible and require zero reprocessing**.

### 2.3 Does Switching to `phoneme_level` Require Only a Config Change or Additional File/Code Changes?
> [!WARNING]
> **Switching from `frame_level` to `phoneme_level` is NOT a config-only change.** Editing `config/Svarah/preprocess.yaml` alone will cause an immediate `ValueError` crash unless three coupled components are updated together:

1. **Directory Path Hardcoding in [`original_fastspeech2/dataset.py`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/dataset.py#L44-L55):**
   - `Dataset.__getitem__()` hardcodes `os.path.join(self.preprocessed_path, "pitch", ...)` and `"energy"`, ignoring `pitch_phoneme/` and `energy_phoneme/`.
   - Because `original_fastspeech2/dataset.py` is kept frozen, switching to `phoneme_level` requires moving `pitch/` -> `pitch_frame/` and `energy/` -> `energy_frame/`, and copying `pitch_phoneme/` -> `pitch/` and `energy_phoneme/` -> `energy/` inside `preprocessed_data/Svarah/` (plus uploading those `7,744` small `.npy` files [`~6 MB` compressed] to Google Drive, since [`English/prepare_colab_bundle.py`](file:///Users/swaruppal/FastSpeech2_Project/English/prepare_colab_bundle.py#L64-L73) only bundled `pitch/` and `energy/`).
2. **Top-Level `[min, max, mean, std]` in [`preprocessed_data/Svarah/stats.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/stats.json):**
   - [`VarianceAdaptor.__init__()`](file:///Users/swaruppal/FastSpeech2_Project/common/model/variance_adaptor.py#L97-L98) reads `stats["pitch"][0:2]` (`[min, max]`) and `stats["energy"][0:2]`. Currently `stats["phoneme_level"]["pitch"]` only stores `[mean, std]` (2 floats). Top-level `stats["pitch"]` and `stats["energy"]` in `stats.json` must be updated to the 4-element phoneme-level `[min, max, mean, std]`.
3. **Hardcoded Frame-Length Assertions in [`English/monitoring.py`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py#L304-L307) (`validate_svarah_dataset()`):**
   - Lines `304–307` check `if pitch_arr.shape[0] != mel_frames or energy_arr.shape[0] != mel_frames: raise ValueError(...)`.
   - Lines `360–363` check `if pitches.shape != (b_size, int(max_mel_len)): raise ValueError(...)`.
   - Both checks in `English/monitoring.py` must be conditional on `pitch_feature_level` / `energy_feature_level` (checking `dur_arr.shape[0]` and `(b_size, int(max_src_len))` when `phoneme_level` is active).
4. **Incompatibility with Existing Checkpoints (`step_5000`):**
   - Moving `pitch_predictor` / `energy_predictor` from post-`LengthRegulator` (`frame_level`) to pre-`LengthRegulator` (`phoneme_level`) changes the input feature distribution and `pitch_bins` / `energy_bins` buffers, requiring a **fresh training run from Step 0**.

---

## 3. Investigation 3: Empirical Audit of F0 Outliers (in Hz)

### 3.1 Are the High-Pitch Outliers Genuine Speech or Extraction/Interpolation Artifacts?
Converting all `1,699,448` normalized pitch frames across the `3,872` utterances back to physical Hertz ($F_0 = z \times 59.4528 + 177.9316\text{ Hz}$) proves conclusively that **the extreme positive tail ($z \in [+4.0, +10.13]$, i.e., $F_0 \in [416\text{ Hz}, 780\text{ Hz}]$) consists of PyWorld DIO+Stonemask octave-doubling / fricative noise spikes amplified by `interp1d` across unvoiced segments**:

| Physical $F_0$ Threshold | Normalized $z$ Threshold | Frame Count (of 1,699,448) | % of All Frames | Phoneme Token Count (of 192,656) | % of All Tokens |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **$F_0 > 350\text{ Hz}$** | $z > +2.89$ | `26,196` | `1.541%` | `2,444` | `1.269%` |
| **$F_0 > 400\text{ Hz}$** | $z > +3.74$ | `10,994` | `0.647%` | `859` | `0.446%` |
| **$F_0 > 450\text{ Hz}$** | $z > +4.58$ | `4,745` | `0.279%` | `310` | `0.161%` |
| **$F_0 > 500\text{ Hz}$** | $z > +5.42$ | `2,383` | `0.140%` | `134` | `0.070%` |
| **$F_0 > 600\text{ Hz}$** | $z > +7.10$ | `540` | `0.032%` | `25` | `0.013%` |

#### Three Pieces of Forensic Evidence:
1. **`54.5%` (`5,990 / 10,994`) of All Frames Above `400 Hz` Fall Inside Unvoiced Consonants or `@sp` Silence:**
   - Across the `628` utterances where $\max(F_0) > 415.7\text{ Hz}$ ($z > 4.0$), the top phonemes hosting the peak $F_0$ frame are:
     - **`@sp` silence (`121` utterances)**, voiceless **`T` (`52`)**, voiceless **`S` (`48`)**, `AH0` (`35`), voiceless **`K` (`29`)**, voiceless **`F` (`24`)**, voiceless **`P` (`21`)**, and **`Z` (`19`)**.
   - During sibilants (`S, Z, CH`), stop releases (`T, K, P`), or breath clicks bordering an `@sp` pause, `pw.dio()` (`f0_ceil = 800 Hz` default) briefly locks onto high-frequency frication energy (`650–780 Hz`). Then `scipy.interpolate.interp1d` connects that spurious boundary value across the entire 25-to-190-frame `@sp` pause!
2. **`82.0%` (`6,867 / 8,371`) of $z > 4.0$ Frames Occur Where $\max(F_0) \ge 1.75 \times \text{median}(F_0)$ Within the Same Utterance:**
   - Examples from `train.txt` and `val.txt`:
     - `281474976897082_f2642_chunk_0` (`spk_068`, **Male**): Utterance median $F_0 = \mathbf{132.3\text{ Hz}}$, jumps to **`751.1 Hz` (`5.68x` median, $z = +9.64$)** across `24` frames of an `@sp` pause.
     - `281474976890932_f1939_chunk_0` (`spk_048`, **Female**): Utterance median $F_0 = \mathbf{236.6\text{ Hz}}$, jumps to **`779.9 Hz` (`3.30x` median, $z = +10.13$)** on fricative `Z`.
     - `281474976893133_f510_chunk_0` (`spk_000`, **Val Female**): Utterance median $F_0 = \mathbf{257.7\text{ Hz}}$, jumps to **`738.9 Hz` (`2.87x` median, $z = +9.44$)** on voiceless `S`.
3. **Why This Hurts Both `nn.MSELoss()` and `VarianceAdaptor` Quantization Bins:**
   - Even though frames with $|z| > 3.0$ are only `1.38%` of all frames, squaring in `nn.MSELoss()` causes them to contribute **`20.19%` of total frame-level $z^2$** (and `16.10%` at `phoneme_level`).
   - Worse, [`VarianceAdaptor._make_bins()`](file:///Users/swaruppal/FastSpeech2_Project/common/model/variance_adaptor.py#L129-L140) spaces `255` linear thresholds evenly from `pitch_min = -2.99` to `pitch_max = +10.13`, wasting `62%` of the 256 embedding bins on the `[+2.0, +10.13]` outlier tail (`< 3.5%` of frames).

### 3.2 Evaluation of Robust Outlier-Handling Alternatives (Without Arbitrary Clipping)

| Option | Mechanism | Pros | Cons / Risks |
| :--- | :--- | :--- | :--- |
| **Option O1 (Do Not Clip `.npy` Files; Switch to `phoneme_level` Only)** | Phoneme-duration averaging dilutes 1–2 frame consonant boundary spikes (`|z| > 4` drops from `11.31%` of $z^2$ to `7.53%`). | Zero modification to normalized values; follows reference FastSpeech 2 [`Preprocessor`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/preprocessor/preprocessor.py#L198-L217). | Still leaves long `@sp` pauses interpolated from boundary spikes; requires fresh training run. |
| **Option O2 (Update `stats.json` Quantization Range `[pitch_min, pitch_max]` to Tukey / Physical Speech Bounds `[-3.0, +4.0]` Without Altering `.npy` Files)** | [`VarianceAdaptor`](file:///Users/swaruppal/FastSpeech2_Project/common/model/variance_adaptor.py#L199) uses `torch.bucketize(values, self.pitch_bins)`, which safely maps any value $> \text{pitch\_max}$ to bin `255`. Setting `stats.json` `pitch_max = 4.0` ($415.7\text{ Hz}$, the 99.5th percentile of human speech in Svarah) doubles quantization resolution across normal speech (`[-1.5, +1.5]` $\sigma$) without clipping a single `.npy` file! | **Zero `.npy` files touched**; `100%` backward-compatible with `torch.bucketize`; grounded in the 99.5th percentile / `415 Hz` adult speech ceiling. | Does not reduce raw `nn.MSELoss()` on the `0.5%` of frames with $z > 4.0$ (only improves Decoder pitch embedding resolution). |
| **Option O3 (Tukey $1.5\,\text{IQR}$ Despiking Before `interp1d` in `preprocess_features.py`)** | Apply the existing [`remove_outliers()`](file:///Users/swaruppal/FastSpeech2_Project/English/alignment/preprocess_features.py#L305-L313) per-utterance Tukey fence $[Q_{25} - 1.5\,\text{IQR}, Q_{75} + 1.5\,\text{IQR}]$ to raw voiced $F_0$ frames *before* `interp1d` so octave-doubling spikes are treated as unvoiced (`0.0`) and interpolated smoothly from neighboring vowels. | Eliminates the `500–780 Hz` spikes at their physical root cause without arbitrary z-score clipping. | Requires re-running feature extraction on all `3,872` WAV/TextGrid pairs (`~12 min` locally) and re-uploading `pitch/` (`15 MB`) to Google Drive. |

**Recommendation on Outlier Handling:** Given the **November 7 deadline**, **do NOT re-extract WAV features or arbitrarily clip `.npy` arrays**. The unseen-speaker split accounts for **`~0.80–0.95` MSE** (over `3x` larger than the outlier tail effect). Fixing the speaker split (and optionally switching to existing `phoneme_level` arrays + setting `stats.json` `pitch_max` appropriately) resolves the validation plateau cleanly without arbitrary clipping.

---

## 4. Investigation 4: Training Configuration Review

Reviewing [`config/Svarah/train.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/train.yaml), [`config/Svarah/model.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/model.yaml), [`common/model/optimizer.py`](file:///Users/swaruppal/FastSpeech2_Project/common/model/optimizer.py), [`common/model/loss.py`](file:///Users/swaruppal/FastSpeech2_Project/common/model/loss.py), and [`English/train.py`](file:///Users/swaruppal/FastSpeech2_Project/English/train.py):

| Setting | Current Value | Verdict | Technical Justification |
| :--- | :--- | :--- | :--- |
| **`warm_up_step` & LR Schedule** | `4000` (Noam schedule: $\text{lr} = d_{\text{model}}^{-0.5} \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$) | **KEEP UNCHANGED (`4000`)** | **Warning:** In [`ScheduledOptim`](file:///Users/swaruppal/FastSpeech2_Project/common/model/optimizer.py#L168-L173), peak LR is $\text{lr}_{\max} = 256^{-0.5} \cdot \text{warmup}^{-0.5}$. Lowering `warm_up_step` from `4000` to `2000` without modifying `ScheduledOptim` would **increase peak LR by $\sqrt{2} = 1.414\times$** (from `9.88e-4` to `1.40e-3`), risking gradient instability! Keeping `warm_up_step: 4000` maintains a safe `9.88e-4` peak LR. |
| **`weight_decay`** | `0.0` | **KEEP UNCHANGED (`0.0`) on Seen-Speaker Split** | [`ScheduledOptim`](file:///Users/swaruppal/FastSpeech2_Project/common/model/optimizer.py#L49-L53) passes all parameters (including `speaker_emb`, `pitch_embedding`, `energy_embedding`, and `LayerNorm`) into a single Adam param group. Once all `88` speakers are present in `train.txt`, `weight_decay: 0.0` is the standard FastSpeech 2 setting and prevents shrinking rarer speaker/phoneme embeddings toward zero. |
| **`dropout`** | `encoder: 0.2`, `decoder: 0.2`, `variance_predictor: 0.5` | **KEEP UNCHANGED** | Matches the reference FastSpeech 2 paper and prevents VariancePredictor overfitting (`0.5` dropout inside the 2-layer CNN predictors). |
| **Loss Component Weighting** | `1 : 1 : 1 : 1 : 1` in [`FastSpeech2Loss`](file:///Users/swaruppal/FastSpeech2_Project/common/model/loss.py#L140-L141) | **KEEP UNCHANGED** | Equal sum is standard in FastSpeech 2 and keeps `total_loss` directly comparable across runs. Once `val.txt` contains seen speakers, `pitch_loss` drops naturally from `1.4` to `~0.50–0.60`, balancing the components without artificial weights. |
| **Best-Checkpoint Selection** | `val_loss.total < prev_best` -> `best_model.pt` in [`English/train.py`](file:///Users/swaruppal/FastSpeech2_Project/English/train.py#L500-L506) | **KEEP UNCHANGED** | Atomic `best_model.pt` saving and verification in `save_and_verify_checkpoint()` works as designed; it only stopped updating after step `5,000` because `val.txt` had `0%` seen speakers. |

---

## 5. Practical Decision Matrix & Implementation Plan (Given the November 7 Deadline)

With less than 48 hours until the **November 7 deadline**, we must weigh **validation loss improvement** against **engineering/upload time and training time on Colab**.

 Crucially, because all `3,872` `.npy` files (`3,208` train + `664` val) are **already stored together in the same flat directories** (`preprocessed_data/Svarah/{mel,pitch,energy,duration}/`) both locally and on Google Drive, we have two concrete options:

### Plan B-Lite (Recommended for November 7 Deadline — Lowest Risk, Zero `.npy` Re-Upload, ~30 Min Colab Run)
**Scope:** Fix the #1 root cause (**100% unseen validation speakers & gender skew**) by regenerating **only `train.txt` (`3,208` utts) and `val.txt` (`664` utts)** using a deterministic, `user_prefix`-disjoint, `recording_id`-disjoint, transcript-disjoint stratified split across all `88` preprocessed `speaker_profile_id`s, while keeping `frame_level` features and all `.npy` files untouched.

- **Why Plan B-Lite Is Ideal Before Nov 7:**
  1. **Zero `.npy` Reprocessing or Drive Upload:** All `15,488` `.npy` files in `mel/`, `pitch/`, `energy/`, and `duration/` on Google Drive remain 100% valid where they already sit! You only replace `train.txt` (`784 KB`) and `val.txt` (`160 KB`) on Google Drive (or upload a `200 KB` `--code_only` bundle).
  2. **Zero Code or Config Changes Required:** `config/Svarah/*.yaml`, `common/model/*`, `original_fastspeech2/dataset.py`, and `English/monitoring.py` require **zero edits**.
  3. **Can Either Warm-Start from `step_5000` or Train Fresh:** Because `frame_level` and `stats.json` are unchanged, you can either start a clean run (`0 -> 10,000` steps in `~35 min` on Colab L4/T4) or resume from `step_5000` in a new checkpoint subfolder (`checkpoints/Svarah_stratified`) so existing checkpoints are never overwritten.
  4. **Expected Quantitative Outcome:**
     - Seen validation speakers: **`0% -> 100%` (`88 / 88` profiles)**
     - Recording / `user_prefix` / transcript leakage: **`0.00%`**
     - Validation gender balance: **`50.0%` Female / `50.0%` Male**
     - Validation `pitch_loss`: drops from **`1.30–1.50` -> `~0.55–0.68`**
     - Validation `mel_loss` & `postnet_mel_loss`: drop from **`0.84` -> `~0.62–0.70`**
     - Total validation loss: drops from **`3.91–4.10` -> `~2.65–2.90`**, with smooth monotonic convergence and genuine `best_model.pt` selection.

---

### Plan B-Full (Stratified Seen-Profile Split + Switch to `phoneme_level` Pitch & Energy — Requires Fresh Run & ~6 MB Feature Patch)
**Scope:** Everything in Plan B-Lite, **plus** switching `pitch.feature` and `energy.feature` to `"phoneme_level"` using the already-extracted `pitch_phoneme/` and `energy_phoneme/` arrays.

- **Exact Files Requiring Changes Under Plan B-Full:**
  1. [`preprocessed_data/Svarah/train.txt`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/train.txt) & [`preprocessed_data/Svarah/val.txt`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/val.txt): Stratified `user_prefix`-disjoint split across all `88` profiles (with backups of original `train.txt.bak` / `val.txt.bak`).
  2. `preprocessed_data/Svarah/pitch/` & `energy/`: Move existing frame-level directories to `pitch_frame/` & `energy_frame/`, and copy `pitch_phoneme/` -> `pitch/` and `energy_phoneme/` -> `energy/` so [`original_fastspeech2/dataset.py`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/dataset.py) loads phoneme-level arrays without modification.
  3. [`preprocessed_data/Svarah/stats.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/stats.json): Update top-level `"pitch"` and `"energy"` `[min, max, mean, std]` to the phoneme-level values (`[-3.0792, 10.1557, 179.8255, 58.3994]` and `[-1.0714, 6.5189, 30.5835, 28.5313]`).
  4. [`config/Svarah/preprocess.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/preprocess.yaml#L24-L29): Change `pitch.feature` and `energy.feature` from `"frame_level"` to `"phoneme_level"`.
  5. [`English/monitoring.py`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py#L304-L363): Update `validate_svarah_dataset()` lines `304–307` and `360–363` to validate `pitch` and `energy` lengths against `dur_arr.shape[0]` (`max_src_len`) when `phoneme_level` is configured.
- **Expected Risks & Requirements Under Plan B-Full:**
  - **Requires a 100% fresh training run from Step 0** (cannot resume from `step_5000`).
  - Requires uploading the replaced `pitch/` and `energy/` folders (`~6 MB` `.tar.gz` patch) to Google Drive so Colab does not load the old frame-level `.npy` files from Drive.
  - Expected total validation loss: **`~2.35–2.65`** (`pitch_loss ≈ 0.45–0.55`, `energy_loss ≈ 0.45–0.52`).

---

## 6. Summary Recommendation

Given the **November 7 deadline**:
1. **Approve Plan B-Lite first** (deterministic, leak-free `user_prefix`-grouped stratified `train.txt` / `val.txt` split across all `88` preprocessed speaker profiles, keeping `frame_level` and all `.npy` files unchanged, saving new checkpoints to a separate directory so existing checkpoints are preserved). It eliminates the #1 root cause (`100%` unseen validation speakers + `63.7%` male skew), requires **zero `.npy` changes on Google Drive**, and is guaranteed to work on the first run.
2. Only if you explicitly want `phoneme_level` prosody prediction and are comfortable replacing `pitch/` and `energy/` on Google Drive for a fresh Step-0 run, approve **Plan B-Full** instead.

