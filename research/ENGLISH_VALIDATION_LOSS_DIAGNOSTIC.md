# FastSpeech 2 English (Svarah) Validation Loss Diagnostic Report

**Date:** 2026-11-05  
**Dataset:** `ai4bharat/Svarah` (English) — `3,208` training utterances, `664` validation utterances  
**Observed Training Behavior:**
- Total training loss falls below `< 1.0` after ~5,000+ steps (`25+` epochs).
- Best validation total loss is **`3.9105` at global step `5,000` (`Epoch 25`)**, after which validation total loss plateaus and fluctuates around **`4.0–4.1`**.
- Component breakdown at plateau:
  - Validation pitch loss (MSE): **`~1.30–1.50`** (`~35%` of total validation loss)
  - Validation mel loss (L1/MAE): **`~0.84`** (`~21%` of total validation loss)
  - Validation PostNet mel loss (L1/MAE): **`~0.84`** (`~21%` of total validation loss)
  - Validation energy loss (MSE): **`~0.59`** (`~15%` of total validation loss)
  - Validation duration loss (log-MSE): **`~0.32`** (`~8%` of total validation loss)
  - Sum of components: $1.40 + 0.84 + 0.84 + 0.59 + 0.32 = \mathbf{3.99}$.

---

## 1. Executive Summary & Ranked Root Causes

A comprehensive quantitative audit of all `3,208` training utterances (`1,415,088` mel frames; `160,109` phoneme tokens) and `664` validation utterances (`284,360` mel frames; `32,547` phoneme tokens) across [`preprocessed_data/Svarah/`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah), [`English/metadata/`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata), [`English/alignment/preprocess_features.py`](file:///Users/swaruppal/FastSpeech2_Project/English/alignment/preprocess_features.py), and [`common/`](file:///Users/swaruppal/FastSpeech2_Project/common) reveals that **the validation loss plateau around `3.91–4.1` is NOT a training loop bug or numerical failure**. Instead, it is the mathematically inevitable result of **five interlocking factors**, dominated by an **unseen-speaker validation split combined with learned speaker lookup embeddings**:

| Rank | Root Cause | Quantitative Impact on Validation Loss | Primary Affected Components |
| :--- | :--- | :--- | :--- |
| **1 (Primary)** | **100% Unseen Speakers in `val.txt` (`22 / 22` validation speakers have `0` utterances in `train.txt`) + Untrained Random `nn.Embedding(115, 256)` Vectors** | Accounts for **`~0.80–0.95` MSE** of the pitch loss gap and **`~0.25–0.35` L1** of each mel loss term (`~1.3–1.6` total validation loss floor) | `pitch_loss`, `mel_loss`, `postnet_mel_loss`, `duration_loss` |
| **2 (High)** | **`frame_level` Pitch & Energy Prediction (`config/Svarah/preprocess.yaml`) + Linear Interpolation Across `35.2%` Unvoiced/Silence Frames** | Inflates pitch and energy MSE by forcing a 3-tap 1D CNN to predict frame-by-frame glottal micro-jitter and 25-frame synthetic linear F0 ramps across `@sp` pauses (`+0.15–0.25` MSE vs `phoneme_level`) | `pitch_loss`, `energy_loss` |
| **3 (Medium-High)** | **Unclipped PyWorld Octave-Error Outliers (`+10.13` $\sigma$ Pitch, `+7.46` $\sigma$ Energy) Squared by `nn.MSELoss()` + Coarsened `pitch_bins`** | The top `1.50%` of validation frames with $|z| > 3\sigma$ contribute **`21.72%` (`0.271` MSE)** of the entire validation pitch squared error and compress normal speech into `<38%` of the 256 quantization bins | `pitch_loss`, `energy_loss`, `mel_loss` |
| **4 (Medium)** | **Demographic & Acoustic Shift Between the `66` Train Speakers and `22` Val Speakers** | Validation is **`63.70%` Male** (vs `47.63%` Male in train), shifting validation mean pitch to **`-0.2438` $\sigma$** (`p50 = -0.6339` $\sigma$) vs `+0.0364` $\sigma$ in train | `pitch_loss`, `mel_loss` |
| **5 (Medium)** | **Optimization Dynamics: `200` Steps/Epoch (`3,208` utts at `batch_size=16`) + `warm_up_step=4000` (`20` Epochs) + `weight_decay=0.0`** | Model reaches `25` full passes over `train.txt` at step `5,000` while LR is near its `9.88e-4` peak (`8.84e-4`), memorizing the `3,208` training utterances (`train_loss < 1.0`) thereafter | Train/Val divergence after step `5,000` |

---

## 2. Ten-Area Quantitative Audit

### Area 1: Train vs. Validation Audio-Duration & Sequence-Length Distributions

Using exact frame counts (`sum(duration)`) and `hop_length = 256` at `22,050 Hz` (`11.61 ms/frame`):

| Metric | Split | Count | Mean | Std | Min | P5 | P25 | Median (P50) | P75 | P95 | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Utterance Duration (sec)** | **Train** | 3,208 | 5.121 s | 2.993 s | 0.372 s | 1.188 s | 2.786 s | 4.534 s | 7.012 s | 10.898 s | 14.756 s |
| **Utterance Duration (sec)** | **Val** | 664 | 4.972 s | 3.211 s | 0.139 s | 1.130 s | 2.487 s | 4.093 s | 6.940 s | 11.122 s | 14.698 s |
| **Mel Frame Length ($L_{\text{mel}}$)** | **Train** | 3,208 | 441.1 | 257.8 | 32 | 102.4 | 240.0 | 390.5 | 604.0 | 938.6 | 1,271 |
| **Mel Frame Length ($L_{\text{mel}}$)** | **Val** | 664 | 428.3 | 276.6 | 12 | 97.3 | 214.3 | 352.5 | 597.8 | 958.0 | 1,266 |
| **Phoneme Sequence Length ($L_{\text{src}}$)** | **Train** | 3,208 | 49.91 | 27.75 | 2 | 13.0 | 29.0 | 45.0 | 67.0 | 104.0 | 173 |
| **Phoneme Sequence Length ($L_{\text{src}}$)** | **Val** | 664 | 49.02 | 31.02 | 2 | 11.0 | 25.0 | 41.5 | 67.0 | 110.0 | 178 |
| **Frames per Token ($d_i$)** | **Train** | 160,109 | 8.838 | 9.738 | 1 | 2.0 | 4.0 | 7.0 | 10.0 | 21.0 | 497 |
| **Frames per Token ($d_i$)** | **Val** | 32,547 | 8.737 | 8.766 | 1 | 3.0 | 5.0 | 7.0 | 10.0 | 19.0 | 221 |
| **Log-Duration $\log(d_i + 1)$** | **Train** | 160,109 | 2.073 | 0.592 | 0.693 | 1.099 | 1.609 | 2.079 | 2.398 | 3.091 | 6.211 |
| **Log-Duration $\log(d_i + 1)$** | **Val** | 32,547 | 2.091 | 0.558 | 0.693 | 1.386 | 1.792 | 2.079 | 2.398 | 2.996 | 5.403 |

**Verdict (Area 1):** Audio durations, mel frame counts, and per-token log-durations are well-matched between `train.txt` and `val.txt`. Notice that `max_seq_len: 1000` in [`config/Svarah/model.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/model.yaml#L29) dynamically expands at runtime in [`common/transformer/models.py`](file:///Users/swaruppal/FastSpeech2_Project/common/transformer/models.py), so utterances with $1000 < L_{\text{mel}} \le 1271$ (`~3.5%` of data) are handled without truncation errors. Duration distribution mismatch is **not** the cause of the plateau.

---

### Area 2: Speaker & Demographic Distributions (Primary Root Cause Discovered)

Inspecting [`preprocessed_data/Svarah/speakers.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/speakers.json), [`English/metadata/speaker_profiles.json`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/speaker_profiles.json), and [`English/metadata/full_manifest.json`](file:///Users/swaruppal/FastSpeech2_Project/English/metadata/full_manifest.json):

| Metric | Training Split (`train.txt`) | Validation Split (`val.txt`) | Overlap / Implication |
| :--- | ---: | ---: | :--- |
| **Utterances** | `3,208` | `664` | `0` shared utterances |
| **Distinct Speaker IDs** | **`66` speakers** | **`22` speakers** | **`0` shared speakers (`0.00%` seen!)** |
| **Seen Validation Speakers** | — | **`0 / 22` (`0` utts, `0.00%`)** | Every validation speaker is **100% unseen** in training |
| **Unseen Validation Speakers** | — | **`22 / 22` (`664` utts, `100.00%`)** | `spk_000, spk_006, spk_011, spk_019, spk_022, spk_024, spk_033, spk_035, spk_045, spk_053, spk_054, spk_064, spk_066, spk_069, spk_071, spk_082, spk_084, spk_089, spk_103, spk_106, spk_110, spk_112` |
| **Female Utterances** | `1,680` (**`52.37%`**) | `241` (**`36.30%`**) | **`-16.07%` fewer female utterances in `val.txt`** |
| **Male Utterances** | `1,528` (**`47.63%`**) | `423` (**`63.70%`**) | **`+16.07%` more male utterances in `val.txt`** |
| **Age `60+` Utterances** | `121` (`3.77%`) | `129` (**`19.43%`**) | `+15.66%` shift toward `60+` speakers in `val.txt` |
| **Top Native Languages (L1)** | Nepali (`10.4%`), Bodo (`9.0%`), Urdu (`7.6%`), Kashmiri (`7.5%`) | Konkani (`17.6%`), Marathi (`14.9%`), Tamil (`10.7%`), Dogri (`10.1%`) | Disjoint demographic profile partition |

#### Why 100% Unseen Validation Speakers Break Lookup-Table Conditioning (`nn.Embedding`)
In [`FastSpeech2.__init__()`](file:///Users/swaruppal/FastSpeech2_Project/common/model/fastspeech2.py#L127-L143) and [`FastSpeech2.forward()`](file:///Users/swaruppal/FastSpeech2_Project/common/model/fastspeech2.py#L262-L265):
```python
self.speaker_emb = nn.Embedding(n_spk, hidden)  # nn.Embedding(115, 256)
...
if self.speaker_emb is not None:
    output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, src_len_max, -1)
```
1. **Zero Gradient Updates for Validation Speakers:** Because the `22` validation speaker IDs never appear in `train.txt` (and `27` test speaker IDs only appear in `test.txt`), **only `66` of the `115` rows in `self.speaker_emb.weight` ever receive a non-zero gradient**. With `weight_decay: 0.0` in [`config/Svarah/train.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/train.yaml#L9), the `22` validation speaker rows remain **permanently frozen at their initial random Gaussian values $\mathcal{N}(0, I_{256})$** ($\|e_{\text{spk}}\|_2 \approx \sqrt{256} = 16.0$).
2. **Corrupting the Encoder Output at Validation Time:** During training (`step 0 -> 5000+`), `VarianceAdaptor` and `Decoder` learn to expect trained speaker embeddings that reside in a structured low-norm manifold encoding speaker register (fundamental frequency baseline), speaking rate, and vocal-tract timbre. During `Trainer.validate()`, adding an untrained $\mathcal{N}(0, I_{256})$ vector to `self.encoder(texts, src_masks)` injects random out-of-distribution bias into every token before duration, pitch, energy, and mel prediction.
3. **Mathematical Proof via Pitch Variance Decomposition:** Even if the speaker embedding were zero instead of random noise, an acoustic model given only text tokens and an unknown speaker ID **cannot know whether the speaker is a `115 Hz` male (`z ≈ -1.06`) or a `235 Hz` female (`z ≈ +0.96`)**. Decomposing the exact squared pitch error across all `284,360` validation frames proves this directly:

| Baseline / Oracle Predictor on Validation Set (`284,360` frames) | Validation Pitch MSE | Explanation |
| :--- | ---: | :--- |
| **1. Predict Global Training Mean (`z = 0.0` everywhere)** | **`1.2468`** | Irreducible MSE floor when speaker identity is unknown and model predicts average pitch |
| **2. Predict Training Speaker Mean (`0.0` for unseen speakers)** | **`1.2468`** | Identical to `1.2468` because **0 of 22 validation speakers appear in `train.txt`** |
| **3. FastSpeech 2 Model at Step 5,000+ (with random untrained `speaker_emb`)** | **`1.30–1.50`** | Random $\mathcal{N}(0, 1)$ speaker embedding shifts pitch predictions randomly toward male or female register on each of the 22 unseen speakers |
| **4. Predict Oracle Validation Speaker Mean (`spk_mean` known)** | **`0.5239`** | **`-0.7229` MSE reduction (`-58.0%`)** simply from knowing each validation speaker's baseline F0 register! |
| **5. Predict Oracle Utterance Mean (pure within-utterance contour variance)** | **`0.4512`** | Theoretical MSE before modeling word/syllable prosody |
| **6. Predict Oracle Phoneme-Level Mean (intra-phoneme micro-prosody floor)** | **`0.0910`** | Residual frame-to-frame F0 variance within individual phonemes |

> [!IMPORTANT]
> **`58.0%` (`0.7229` MSE units) of the validation pitch variance is pure between-speaker register difference (e.g., Male vs. Female baseline F0).** Because `val.txt` contains 100% unseen speakers whose `nn.Embedding(115, 256)` rows are never trained, **no model using an indexed lookup table can ever drive validation pitch MSE below `~1.1–1.2` on this split**, and random initial embeddings push it to **`1.3–1.5`**.

---

### Area 3: Phoneme Frequency Distributions

Comparing the `160,109` phoneme tokens in `train.txt` (`69` active ARPAbet + pause symbols) and `32,547` phoneme tokens in `val.txt` (`67` active symbols):

| Phoneme | Train Count | Train % | Val Count | Val % | Abs Diff (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `N` | 11,481 | 7.17% | 2,554 | 7.85% | `+0.68%` |
| `AH0` | 12,094 | 7.55% | 2,306 | 7.09% | `-0.47%` |
| `IH0` | 5,376 | 3.36% | 996 | 3.06% | `-0.30%` |
| `DH` | 3,362 | 2.10% | 601 | 1.85% | `-0.25%` |
| `AH1` | 2,901 | 1.81% | 664 | 2.04% | `+0.23%` |
| `AE1` | 3,856 | 2.41% | 710 | 2.18% | `-0.23%` |
| `sp` (`@sp`) | 8,329 | 5.20% | 1,765 | 5.42% | `+0.22%` |
| `AA1` | 3,208 | 2.00% | 593 | 1.82% | `-0.18%` |
| `ER0` | 2,734 | 1.71% | 611 | 1.88% | `+0.17%` |
| `D` | 6,069 | 3.79% | 1,180 | 3.63% | `-0.17%` |

- Only two ultra-rare stress variants (`OY2` and `UH0`, each occurring only `8` times in `train.txt` [`0.005%`]) have `0` occurrences in `val.txt`, and `val.txt` contains **zero OOV phonemes**.
- **Verdict (Area 3):** Phoneme coverage and token frequency distributions are tightly aligned (`max diff = 0.68%`) and do **not** contribute to the validation plateau.

---

### Area 4 & Area 5: Pitch and Energy Distributions, Normalization, and Outlier Tail Analysis

From [`preprocessed_data/Svarah/stats.json`](file:///Users/swaruppal/FastSpeech2_Project/preprocessed_data/Svarah/stats.json):
- **Frame-level pitch stats:** `min = -2.9928`, `max = +10.1252`, `mean = 177.9324 Hz`, `std = 59.4508 Hz`
- **Phoneme-level pitch stats:** `mean = 180.6591 Hz`, `std = 58.1337 Hz`
- **Frame-level energy stats:** `min = -0.8998`, `max = +7.4643`, `mean = 24.5356`, `std = 27.2689`
- **Phoneme-level energy stats:** `mean = 26.8669`, `std = 24.7166`

| Feature Array | Split | Frames / Tokens | Mean ($z$) | Std ($z$) | Min ($z$) | P5 | P25 | Median (P50) | P75 | P95 | P99 | Max ($z$) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`pitch/` (frame)** | **Train** | 1,415,088 | `+0.036` | `1.076` | `-2.993` | `-1.340` | `-0.867` | `-0.034` | `+0.695` | `+1.848` | `+3.297` | **`+10.125`** |
| **`pitch/` (frame)** | **Val** | 284,360 | **`-0.244`** | `1.090` | `-2.001` | `-1.400` | `-1.040` | **`-0.634`** | `+0.433` | `+1.667` | `+3.465` | **`+9.436`** |
| **`pitch_phoneme/`** | **Train** | 160,109 | `+0.028` | `1.048` | `-3.079` | `-1.338` | `-0.879` | `-0.015` | `+0.701` | `+1.790` | `+3.103` | **`+10.156`** |
| **`pitch_phoneme/`** | **Val** | 32,547 | **`-0.271`** | `1.062` | `-1.930` | `-1.396` | `-1.056` | **`-0.667`** | `+0.428` | `+1.601` | `+3.261` | **`+7.827`** |
| **`energy/` (frame)** | **Train** | 1,415,088 | `+0.016` | `1.021` | `-0.900` | `-0.888` | `-0.775` | `-0.362` | `+0.503` | `+2.102` | `+3.462` | **`+7.464`** |
| **`energy/` (frame)** | **Val** | 284,360 | `-0.020` | `0.885` | `-0.899` | `-0.874` | `-0.714` | `-0.307` | `+0.438` | `+1.724` | `+2.910` | **`+6.722`** |
| **`energy_phoneme/`** | **Train** | 160,109 | `+0.013` | `1.015` | `-1.071` | `-1.023` | `-0.781` | `-0.306` | `+0.543` | `+2.056` | `+3.306` | **`+6.519`** |
| **`energy_phoneme/`** | **Val** | 32,547 | `-0.074` | `0.859` | `-1.071` | `-0.995` | `-0.766` | `-0.306` | `+0.396` | `+1.586` | `+2.705` | **`+5.288`** |

#### Two Critical Findings in Pitch & Energy Normalization:
1. **Unclipped PyWorld Pitch-Doubling Spikes (`+10.13` $\sigma$ $\approx 780\text{ Hz}$) Squared by MSE Loss:**
   - In [`English/alignment/preprocess_features.py`](file:///Users/swaruppal/FastSpeech2_Project/English/alignment/preprocess_features.py#L305-L348), `remove_outliers()` filters out values outside $[Q_{25} - 1.5\,\text{IQR},\; Q_{75} + 1.5\,\text{IQR}]$ *when fitting `StandardScaler`*, yielding `mean = 177.93 Hz, std = 59.45 Hz`.
   - However, lines `343–344` and `409–410` normalize the **unclipped** raw arrays `(p_f - pitch_frame_mean) / pitch_frame_std` without clamping extreme tracking spikes (`500–780 Hz` octave-doubling errors from PyWorld DIO+Stonemask on reverberant/noisy chunks).
   - Because `FastSpeech2Loss` uses **`nn.MSELoss()`** (`(pred - target)^2`), the **`1.503%` of validation frames with $|z| > 3.0$ account for `21.72%` (`0.2708` MSE units) of the entire validation pitch `E[z^2]`**! Similarly, the `0.886%` of validation energy frames with $|z| > 3.0$ account for **`15.37%`** of validation energy `E[z^2]`.
2. **Quantization Bin Compression in `VarianceAdaptor`:**
   - [`VarianceAdaptor._make_bins()`](file:///Users/swaruppal/FastSpeech2_Project/common/model/variance_adaptor.py#L129-L140) constructs `255` linear boundaries spanning `[pitch_min, pitch_max] = [-2.9928, +10.1252]` (total span `13.118` $\sigma$, bin width `0.0514` $\sigma$).
   - Because `pitch_max` is anchored by a single `+10.13` $\sigma$ (`780 Hz`) outlier frame, **`62%` of the 256 pitch embedding bins (`[+2.0, +10.13]` $\sigma$) cover `<3.5%` of speech frames**, while `95%` of normal speech frames are squeezed into fewer than `100` bins.

---

### Area 6: Unvoiced-Frame Handling & `frame_level` vs. `phoneme_level` Granularity

In [`English/alignment/preprocess_features.py`](file:///Users/swaruppal/FastSpeech2_Project/English/alignment/preprocess_features.py#L169-L182):
```python
nonzero_ids = np.where(pitch != 0)[0]
interp_fn = interp1d(
    nonzero_ids,
    pitch[nonzero_ids],
    fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
    bounds_error=False,
)
pitch = interp_fn(np.arange(0, len(pitch)))
```
Across the dataset:
- **Internal `@sp` silence frames:** `212,043 / 1,415,088` (**`14.98%`**) in `train.txt` (`8,329` pauses, mean duration **`25.46` frames $\approx 296\text{ ms}$**) and `42,288 / 284,360` (**`14.87%`**) in `val.txt` (`1,765` pauses, mean duration **`23.96` frames $\approx 278\text{ ms}$**).
- **Voiceless consonant frames (`P, T, K, S, SH, F, TH, HH, CH`):** **`19.86%`** in `train.txt` and **`20.35%`** in `val.txt`.
- **Total unvoiced + silence frames:** **`34.85%` in `train.txt` and `35.22%` in `val.txt`**.

#### Why `frame_level` Pitch Prediction Plateaus Higher Than `phoneme_level`:
1. In [`config/Svarah/preprocess.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/preprocess.yaml#L24-L29), `pitch.feature` and `energy.feature` are set to `"frame_level"` (whereas reference FastSpeech 2 uses `"phoneme_level"`).
2. At `"frame_level"`, [`VarianceAdaptor`](file:///Users/swaruppal/FastSpeech2_Project/common/model/variance_adaptor.py#L30-L31) first expands the phoneme hidden states $x \in \mathbb{R}^{B \times L_{\text{src}} \times 256}$ via `LengthRegulator` — repeating an `@sp` token's hidden vector **25 identical times in a row** — and then passes those identical vectors into `self.pitch_predictor` (a 2-layer 1D CNN with `kernel_size = 3`, receptive field = `5` frames).
3. Inside a 25-frame `@sp` pause, the central `21` frames see **identical input vectors** (`[h_sp, h_sp, h_sp, h_sp, h_sp]`), making it **mathematically impossible** for a receptive-field-5 CNN without positional encoding after `LengthRegulator` to predict a sloped linear interpolation ramp between the preceding word's final vowel F0 and the next word's initial vowel F0!
4. Furthermore, at `"frame_level"`, every 25-frame `@sp` pause receives **25$\times$ the weight** of a 1-frame consonant in the frame-averaged `nn.MSELoss()`, whereas at `"phoneme_level"` each phoneme token receives equal weight (`1 / L_src`) and averages out intra-phoneme glottal jitter (`0.0910` MSE).

---

### Area 7: Training and Validation Preprocessing Consistency

Auditing [`English/alignment/preprocess_features.py`](file:///Users/swaruppal/FastSpeech2_Project/English/alignment/preprocess_features.py#L300-L433):
- **Zero Data Leakage:** `pitch_frame_mean`, `pitch_frame_std`, `energy_frame_mean`, `energy_frame_std`, `pitch_mean`, `pitch_std`, `energy_mean`, and `energy_std` are computed **strictly from `train.txt`** (lines `300–353`) and applied frozen to `val.txt` (lines `409–417`).
- **Identical Feature Extraction:** Both splits use the exact same `SvarahFeatureExtractor` instance (`sr=22050`, `n_fft=1024`, `hop_length=256`, `win_length=1024`, `n_mels=80`, `fmin=0`, `fmax=8000`, `DIO+Stonemask`, exact TextGrid duration reconciliation).
- **Bucket Boundary Coverage:** `0.0000%` of validation pitch or energy frames fall outside the training `[min, max]` stored in `stats.json` (and even if any did, `torch.bucketize` safely maps `< min` to bin `0` and `> max` to bin `255`).
- **Verdict (Area 7):** Preprocessing between `train` and `val` is 100% consistent and leak-free.

---

### Area 8: Loss Calculation, Masking, and Teacher-Forcing Mechanics

Auditing [`common/model/loss.py`](file:///Users/swaruppal/FastSpeech2_Project/common/model/loss.py) and [`common/training/engine.py`](file:///Users/swaruppal/FastSpeech2_Project/common/training/engine.py#L246-L273):
1. **Masking Convention:** `src_masks` and `mel_masks` use `True = padded, False = valid`. `FastSpeech2Loss` inverts the masks (`~src_masks`, `~mel_masks`) via `.masked_select()` before calling `nn.L1Loss()` (for `mel_loss` and `postnet_mel_loss`) and `nn.MSELoss()` (for `pitch_loss`, `energy_loss`, and `duration_loss`). Thus **zero padded tokens or frames ever leak into any loss term**.
2. **Equal Unweighted Sum:** `total_loss = mel_loss + postnet_mel_loss + pitch_loss + energy_loss + duration_loss` (all weights = `1.0`). Because `pitch_loss` and `energy_loss` use **MSE** on unit-variance z-scores with heavy positive tails (`max = +10.13` $\sigma$), while `mel_loss` uses **L1 (MAE)** and `duration_loss` uses **MSE on compressed $\log(d+1)$** (where $\text{Var}(\log(d+1)) = 0.311$), `pitch_loss` naturally dominates the numerical scale on unseen speakers.
3. **Why `mel_loss` Plateaus at `0.84` Despite `pitch_loss` Being `1.3–1.5`:**
   - In [`Trainer.validate_step()`](file:///Users/swaruppal/FastSpeech2_Project/common/training/engine.py#L259-L260), validation calls `self.model(*(batch_dev[2:]))`, which passes **ground-truth `p_targets`, `e_targets`, and `d_targets`** into `VarianceAdaptor`.
   - Inside [`VarianceAdaptor._embed()`](file:///Users/swaruppal/FastSpeech2_Project/common/model/variance_adaptor.py#L189-L196), when `target is not None`, the model feeds `embedding(torch.bucketize(target, bins))` (teacher-forced ground-truth pitch/energy/duration) to the `Decoder`!
   - Thus, the `0.84` validation `mel_loss` is **not** caused by `pitch_predictor` errors cascading into the Decoder; rather, both `pitch_loss` (`1.3–1.5`) and `mel_loss` (`0.84`) are independently bounded by the **untrained random `speaker_emb` vector** added to the Encoder output (`utt_mel_means` has a between-utterance standard deviation of `0.89 dB` in train and `0.76 dB` in val, plus `+0.32 dB` mean shift in val).

---

### Area 9: Dropout, Weight Decay, and Learning-Rate Schedule

Inspecting [`config/Svarah/model.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/model.yaml), [`config/Svarah/train.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/train.yaml), and [`common/model/optimizer.py`](file:///Users/swaruppal/FastSpeech2_Project/common/model/optimizer.py):
- **DatasetSteps per Epoch:** `3,208` utterances / `batch_size: 16` = **`200.5` steps/epoch**.
- **Warmup Schedule (`warm_up_step: 4000`):** Originally designed for LJSpeech (`~24,000` steps / `100` epochs) or LibriTTS (`~100,000` steps). On Svarah (`3,208` utterances), `4,000` steps equals **20 full epochs of linearly increasing learning rate**, peaking at step `4,000` at:
  $$\text{lr}_{\max} = 256^{-0.5} \times 4000^{-0.5} = 0.0625 \times 0.015811 = \mathbf{9.882 \times 10^{-4}}$$
- **At Step `5,000` (Best Validation Loss = `3.9105`):** The model is at **Epoch `25`**, with $\text{lr} = \mathbf{8.839 \times 10^{-4}}$ (still `89.4%` of peak LR), and `anneal_steps: [300000, 400000, 500000]` is never reached within a typical Colab run (`10,000–50,000` steps).
- **Zero Weight Decay (`weight_decay: 0.0`):** With `35.2M` parameters for only `3,208` utterances (`4.55` hours) and `weight_decay: 0.0`, once the model passes 25 epochs at $\text{lr} \approx 8.8 \times 10^{-4}$, the Encoder (`dropout: 0.2`), Decoder (`dropout: 0.2`), and `VariancePredictor` (`dropout: 0.5`) steadily memorize the training utterances (`train_loss < 1.0`) while validation loss on the 22 unseen speakers slowly drifts upward from `3.91` to `4.0–4.1`.

---

### Area 10: Representativeness of Validation Examples

Summarizing the distribution shifts between `train.txt` and `val.txt`:
1. **Speaker Overlap:** **`0.00%`** (`0 / 22` validation speakers appear in `train.txt`).
2. **Gender Ratio Shift:** `train.txt` is `52.37%` Female / `47.63%` Male, whereas `val.txt` is **`36.30%` Female / `63.70%` Male** (`+16.07%` male skew).
3. **Pitch Distribution Shift:** Because `val.txt` is `63.70%` Male, validation median normalized frame pitch is **`-0.6339` $\sigma$** (mean `-0.2438` $\sigma$), compared to training median **`-0.0336` $\sigma$** (mean `+0.0364` $\sigma$).
4. **Utterance-Level Mel Mean Shift:** Validation mean mel energy is `-5.322` (`std = 0.764`) vs training `-5.643` (`std = 0.891`), a `+0.321` shift across unseen recording environments/speakers.

---

## 3. Prioritized Corrective Experiments & Expected Trade-Offs

Below are the concrete corrective experiments ordered by expected validation loss reduction and scientific rigor, along with their exact trade-offs:

### Experiment 1 (Highest Impact — Structural Fix): Seen-Speaker Stratified Split vs. Zero-Vector / Mean-Speaker Fallback for Unseen Speakers
Because `self.speaker_emb = nn.Embedding(115, 256)` is a closed-set lookup table, **there are only two mathematically valid ways to evaluate a multi-speaker lookup-embedding TTS model**:
- **Option 1A (Stratified Seen-Speaker Validation Split — Standard Multi-Speaker FastSpeech 2 Protocol):**
  - Partition utterances **within** each speaker profile (`80%` train / `10%` val / `10%` test per speaker, stratified by duration and transcript-deduplicated) so that all active speakers have trained embeddings in `self.speaker_emb`.
  - **Expected Impact:** Validation pitch MSE will drop from **`1.30–1.50` -> `0.50–0.65`**, and validation mel/PostNet L1 will drop from **`0.84` -> `0.58–0.68`**, reducing total validation loss from **`3.91` -> `~2.4–2.7`**.
  - **Trade-Off:** Requires re-splitting `train.txt` / `val.txt` (or moving `val` files between already-extracted feature directories and re-normalizing `stats.json`). Evaluates multi-speaker in-domain synthesis rather than zero-shot unseen-speaker synthesis.
- **Option 1B (Keep Current Split Unchanged, Zero/Mean-Initialize Unseen Speaker Embeddings or Mask Unseen Speakers to Mean Trained Embedding):**
  - If the current `train.txt` / `val.txt` split must remain frozen, unseen speaker rows in `self.speaker_emb` are currently random $\mathcal{N}(0, I_{256})$ vectors (`norm ≈ 16`). Replacing unseen speaker embeddings at evaluation time with the **mean of the 66 trained speaker embeddings** (or initializing `speaker_emb` with small variance $\mathcal{N}(0, 0.01)$ / enabling `weight_decay`) eliminates the random vector corruption.
  - **Expected Impact:** Reduces validation pitch MSE from **`1.30–1.50` -> `~1.15–1.25`** and mel L1 by `~0.05–0.08` without touching dataset manifests.
  - **Trade-Off:** Cannot fall below the `1.15–1.25` unseen-speaker register ambiguity floor because a lookup table still cannot know whether an unseen speaker ID is male or female.

### Experiment 2 (High Impact — Config-Only or Fast Re-Save): Switch Pitch & Energy from `frame_level` to `phoneme_level`
- **Mechanism:** In [`config/Svarah/preprocess.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/preprocess.yaml#L24-L29), change `pitch.feature` and `energy.feature` from `"frame_level"` to `"phoneme_level"`.
- **Readiness on Disk:** [`English/alignment/preprocess_features.py`](file:///Users/swaruppal/FastSpeech2_Project/English/alignment/preprocess_features.py#L351-L361) **already extracted and saved** normalized phoneme-level arrays to `preprocessed_data/Svarah/pitch_phoneme/` and `preprocessed_data/Svarah/energy_phoneme/`, and stored their mean/std in `stats.json["phoneme_level"]`! (Only `pitch/` and `energy/` paths + `[min, max]` in `stats.json` would need to point to the phoneme-level arrays).
- **Expected Impact:**
  - Eliminates the impossible task of predicting 25-frame linear interpolation ramps across `@sp` pauses using a receptive-field-5 CNN after `LengthRegulator`.
  - Reduces validation pitch MSE by **`~0.15–0.25`** and energy MSE by **`~0.08–0.12`**, while substantially improving synthesized prosody stability at inference time (the standard finding of the original FastSpeech 2 paper, Section 3.2.2).
- **Trade-Off:** Requires starting a fresh training run because `VarianceAdaptor` applies `pitch_predictor` and `energy_predictor` *before* `LengthRegulator` (`[B, L_src]`) rather than *after* (`[B, L_mel]`), and `pitch_bins` / `energy_bins` buffers change to the phoneme-level `[min, max]`.

### Experiment 3 (Medium-High Impact): Clip Normalized Pitch & Energy Outliers to `[-3.5, +3.5]` $\sigma$ (or Use Smooth L1 / Huber Loss)
- **Mechanism:** Either clip normalized `pitch` and `energy` targets at `±3.5` $\sigma$ (removing the `+10.13` $\sigma$ PyWorld octave-doubling spikes that contribute `21.72%` of validation squared error from just `1.5%` of frames and waste `62%` of the 256 quantization bins) **or** replace `nn.MSELoss()` on pitch/energy with `nn.SmoothL1Loss(beta=1.0)`.
- **Expected Impact:** Reduces pitch MSE/Smooth-L1 outlier penalty by **`~0.20–0.27`** and restores `2.5x` finer pitch quantization resolution across normal human speech (`[-2.0, +2.0]` $\sigma$).
- **Trade-Off:** Modifying the loss function or clipping `.npy` targets changes the numerical scale of `pitch_loss` / `energy_loss`, so raw loss numbers before/after clipping are not directly comparable on the same scale.

### Experiment 4 (Medium Impact — Hyperparameter Tuning): Reduce `warm_up_step` (`4000 -> 1500` or `2000`) and Enable `weight_decay: 1e-6` to `1e-4`
- **Mechanism:** Because `3,208` utterances at `batch_size = 16` is only `200` steps/epoch, `warm_up_step: 4000` warms up for `20` full epochs and peaks at $\text{lr} = 9.88 \times 10^{-4}$ right before overfitting begins at step `5,000` (`Epoch 25`). Setting `warm_up_step: 2000` (`10` epochs) with `weight_decay: 1e-5` (excluding `pitch_bins`/`energy_bins` and LayerNorm biases, or applying small Adam weight decay) prevents post-step-5000 memorization.
- **Expected Impact:** Stabilizes validation loss after step `5,000` and prevents validation drift from `3.91 -> 4.10`.
- **Trade-Off:** Note that `ScheduledOptim` in `torch.optim.Adam(..., weight_decay=...)` applies L2 penalty to all parameters with `requires_grad=True`, which will also shrink the 22 unseen speaker embeddings in `self.speaker_emb` smoothly toward `0.0` instead of leaving them at $\mathcal{N}(0, 1)$!

---

## 4. Recommended Next Training Configuration & Decision Matrix

Before modifying any files or starting a new training run, choose one of the two paths below based on project constraints:

### Path A — Zero-Dataset-Change Stabilization (Resume or Retrain on Existing `train.txt` / `val.txt`)
Use this path if `train.txt` (`66` speakers) and `val.txt` (`22` unseen speakers) **must remain strictly untouched**:
1. **Use Checkpoint `step_5000` (`val_loss = 3.9105`) as the Best Model** for the current run: because all `22` validation speakers are unseen (`E[z^2] = 1.2468` baseline pitch variance), `~3.85–3.91` is already at the theoretical generalization ceiling of the current `frame_level` + random-unseen-speaker-embedding configuration. Continuing past `5,000–10,000` steps only memorizes the `66` training speakers (`train_loss < 1.0`).
2. **If running a short corrective retraining run on the same split:**
   - Set `weight_decay: 1.0e-4` (or zero-initialize `model.speaker_emb`) so unseen speaker IDs evaluate with a `0.0` vector instead of random $\mathcal{N}(0, 1)$ noise.
   - Set `warm_up_step: 2000` in [`config/Svarah/train.yaml`](file:///Users/swaruppal/FastSpeech2_Project/config/Svarah/train.yaml).

### Path B — Full Acoustic & Prosodic Improvement (Recommended if Preprocessing/Config Updates Are Permitted)
If you want validation loss to genuinely reflect multi-speaker synthesis quality and drop from **`3.91` -> `~2.1–2.4`**:
1. **Stratified Seen-Speaker Split:** Ensure all speakers in `val.txt` have training utterances in `train.txt` (`80/10/10` per-speaker utterance split with zero transcript leakage), **or** map `val.txt` speaker IDs to a gender-matched seen speaker / mean embedding.
2. **Switch `config/Svarah/preprocess.yaml` to `phoneme_level`** for both `pitch.feature` and `energy.feature` (using the already-computed `pitch_phoneme/` and `energy_phoneme/` features).
3. **Clamp normalized pitch/energy targets to `[-3.5, +3.5]` $\sigma$** during normalization so PyWorld DIO octave-doubling spikes (`+10.13` $\sigma$) do not dominate `nn.MSELoss()` or compress `VarianceAdaptor` quantization bins.
4. **Update `config/Svarah/train.yaml`:** `warm_up_step: 2000`, `weight_decay: 1.0e-5`, early stopping patience at `5,000–8,000` steps (`25–40` epochs).

