# FastSpeech 2 Variance Adaptor Technical Analysis (`VARIANCE_ADAPTOR_ANALYSIS.md`)

## 1. File Locations & Scope

- **Primary File Inspected**: `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py` (296 lines)
- **Directly Connected Files Inspected**:
  - `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/utils/tools.py` (`get_mask_from_lengths` lines 91–99, `expand` lines 102–106, `synth_samples` lines 164–211, `pad` lines 299–317)
  - `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/fastspeech2.py` (`FastSpeech2.__init__` line 21, `FastSpeech2.forward` lines 73–92)
  - `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/loss.py` (`FastSpeech2Loss.forward` lines 19–92)
  - `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/preprocessor/preprocessor.py` (`build_from_path` lines 92–135, `process_utterance` lines 181–245, `normalize` lines 303–314)
  - `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/preprocessed_data/{LJSpeech,LibriTTS,AISHELL3}/stats.json`
  - `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/config/{LJSpeech,LibriTTS,AISHELL3,LJSpeech_paper}/{preprocess,model}.yaml`

---

## 2. Summary of Classes in `original_fastspeech2/model/modules.py`

| Class | Lines | Purpose |
|---|---|---|
| `VarianceAdaptor(nn.Module)` | `17–158` | Top-level variance module that predicts duration, pitch, and energy, converts continuous pitch/energy into quantized embeddings, adds them to the hidden representations, and regulates sequence length from phoneme level to Mel-frame level. |
| `LengthRegulator(nn.Module)` | `161–194` | Expands each phoneme-level hidden vector $x_{b, i} \in \mathbb{R}^{256}$ by repeating it $d_{b, i}$ times along the time axis and pads the resulting variable-length Mel-frame sequences across the batch. |
| `VariancePredictor(nn.Module)` | `197–250` | Shared neural network architecture instantiated three times inside `VarianceAdaptor` (for `duration_predictor`, `pitch_predictor`, and `energy_predictor`): 2-layer 1D Conv + ReLU + LayerNorm + Dropout followed by a scalar linear projection. |
| `Conv(nn.Module)` | `253–296` | Wrapper around `nn.Conv1d` that transposes `[B, L, C]` $\rightarrow$ `[B, C, L]` before convolution and transposes back to `[B, L, C]` afterward so `nn.LayerNorm` can operate on the last dimension `C`. |

---

## 3. Component-by-Component Technical Analysis

### 3.1 Component 1: `VarianceAdaptor` (`model/modules.py`, lines 17–158)

- **Purpose**: Bridges the phoneme-level `Encoder` and the frame-level `Decoder` by injecting prosodic variance information (duration, F0 pitch, and frame energy) into the hidden sequence.
- **Initialization (`__init__`, lines 20–78)**:
  1. Instantiates three independent `VariancePredictor(model_config)` networks:
     - `self.duration_predictor` (line 22)
     - `self.pitch_predictor` (line 24)
     - `self.energy_predictor` (line 25)
  2. Instantiates `self.length_regulator = LengthRegulator()` (line 23).
  3. Reads `self.pitch_feature_level` (`preprocess_config["preprocessing"]["pitch"]["feature"]`, lines 27–29) and `self.energy_feature_level` (`preprocess_config["preprocessing"]["energy"]["feature"]`, lines 30–32), asserting each is in `["phoneme_level", "frame_level"]` (lines 33–34).
  4. Reads quantization settings from `model_config["variance_embedding"]` (lines 36–40): `pitch_quantization` (`"linear"` or `"log"`), `energy_quantization` (`"linear"` or `"log"`), and `n_bins` (`256`).
  5. Opens `<preprocessed_path>/stats.json` (lines 41–46) to load `pitch_min, pitch_max` and `energy_min, energy_max`, constructs `self.pitch_bins` and `self.energy_bins` (`nn.Parameter` of shape `[255]`, `requires_grad=False`, lines 48–71), and creates `self.pitch_embedding = nn.Embedding(256, 256)` and `self.energy_embedding = nn.Embedding(256, 256)` (lines 73–78).
- **Connection to `FastSpeech2.forward` (`model/fastspeech2.py`, lines 73–92)**:
  - Receives `output` (`[B, L_src, 256]`, Encoder output + optional speaker embedding), `src_masks` (`[B, L_src]`), `mel_masks` (`[B, L_mel]` or `None`), `max_mel_len` (`int` or `None`), `p_targets`, `e_targets`, `d_targets`, `p_control`, `e_control`, `d_control`.
  - Returns a 7-tuple `(output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks)`, where `output` (`[B, L_mel, 256]`) and `mel_masks` (`[B, L_mel]`) are passed directly into `self.decoder(output, mel_masks)` at line 94 of `model/fastspeech2.py`, while all 7 outputs are forwarded to `FastSpeech2.forward`'s return tuple.

---

### 3.2 Component 2: `VariancePredictor` & `Conv` (`model/modules.py`, lines 197–296)

- **Purpose**: Predicts a 1D per-token (or per-frame) scalar sequence from a sequence of `256`-dimensional hidden states, zeroing out padded positions.
- **Configuration Parameters (`model.yaml`)**:
  - `model_config["transformer"]["encoder_hidden"]`: `256` (`self.input_size`)
  - `model_config["variance_predictor"]["filter_size"]`: `256` (`self.filter_size`, `self.conv_output_size`)
  - `model_config["variance_predictor"]["kernel_size"]`: `3` (`self.kernel`)
  - `model_config["variance_predictor"]["dropout"]`: `0.5` (`self.dropout`)
- **Internal Layer Stack (`lines 209–240`)**:
  1. `conv1d_1`: `Conv(256, 256, kernel_size=3, padding=(3-1)//2 = 1)` — transposes `[B, L, 256]` $\rightarrow$ `[B, 256, L]`, applies `nn.Conv1d`, transposes back to `[B, L, 256]`.
  2. `relu_1`: `nn.ReLU()`
  3. `layer_norm_1`: `nn.LayerNorm(256)` (normalizes across channel dimension `-1`)
  4. `dropout_1`: `nn.Dropout(0.5)`
  5. `conv1d_2`: `Conv(256, 256, kernel_size=3, padding=1)` *(Note: `padding=1` is hardcoded at line 230 rather than `(self.kernel - 1) // 2`!)*
  6. `relu_2`: `nn.ReLU()`
  7. `layer_norm_2`: `nn.LayerNorm(256)`
  8. `dropout_2`: `nn.Dropout(0.5)`
  9. `self.linear_layer`: `nn.Linear(256, 1)`
- **Forward Pass & Tensor Shapes (`lines 242–250`)**:
  - **Input**:
    - `encoder_output`: `[B, L, 256]` (`torch.float32`), where $L = L_{\text{src}}$ for duration and phoneme-level pitch/energy, or $L = L_{\text{mel}}$ for frame-level pitch/energy.
    - `mask`: `[B, L]` (`torch.bool`, `True` at padded positions).
  - **Operations**:
    - `out = self.conv_layer(encoder_output)` $\rightarrow$ `[B, L, 256]`
    - `out = self.linear_layer(out)` $\rightarrow$ `[B, L, 1]`
    - `out = out.squeeze(-1)` $\rightarrow$ `[B, L]`
    - `if mask is not None: out = out.masked_fill(mask, 0.0)` $\rightarrow$ `[B, L]`
  - **Output**: `out` of shape `[B, L]` (`torch.float32`).
- **Training vs. Inference**:
  - During training (`model.train()`), the two `nn.Dropout(0.5)` layers drop 50% of activations, and the predicted values are supervised by `FastSpeech2Loss` (MSE against ground-truth targets).
  - During inference (`model.eval()`), dropout is disabled, and the predicted values directly drive `LengthRegulator` (`duration_predictor`) and quantized embedding lookups (`pitch_predictor`, `energy_predictor`).

---

### 3.3 Component 3: `LengthRegulator` (`model/modules.py`, lines 161–194)

- **Purpose**: Aligns the length of the hidden representation sequence $x$ (`[B, L_src, 256]`) with the target Mel-spectrogram frame length $L_{\text{mel}}$ by repeating each token's hidden state $x_{b, i}$ according to its integer duration $d_{b, i}$.
- **Methods & Execution Flow**:
  1. `forward(self, x, duration, max_len)` (lines 192–194): Delegates directly to `self.LR(x, duration, max_len)`.
  2. `expand(self, batch, predicted)` (lines 182–190):
     - Operates on a single batch element `batch` (`[L_src, 256]`) and its 1D duration vector `predicted` (`[L_src]`).
     - Loops over each token $i \in \{0, \dots, L_{\text{src}}-1\}$:
       ```python
       expand_size = predicted[i].item()
       out.append(vec.expand(max(int(expand_size), 0), -1))
       ```
     - Here `vec` has shape `[256]`, so `vec.expand(k, -1)` creates a 2D view of shape `[k, 256]` repeating `vec` $k = \max(\text{int}(\text{expand\_size}), 0)$ times without copying memory until `out = torch.cat(out, 0)` (line 188) concatenates all token expansions into `expanded` of shape `[T_b, 256]`, where $T_b = \sum_{i=0}^{L_{\text{src}}-1} \max(\text{int}(d_{b, i}), 0)$.
  3. `LR(self, x, duration, max_len)` (lines 167–180):
     - Iterates over `(batch, expand_target)` in `zip(x, duration)`, collecting `expanded` (`[T_b, 256]`) into `output` and `T_b = expanded.shape[0]` into `mel_len`.
     - Pads the list of 2D tensors `output` to `[B, L_mel, 256]` via `pad(output, max_len)` (lines 175–178, imported from `utils/tools.py` lines 299–317), which zero-pads each tensor along dimension 0 (`F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)`) and stacks them via `torch.stack(out_list)`.
     - Returns `output, torch.LongTensor(mel_len).to(device)` (line 180).
- **Input & Output Tensor Shapes**:
  - **Inputs**:
    - `x`: `[B, L_src, 256]` (`torch.float32`)
    - `duration`: `[B, L_src]` (`torch.int64` during training; `torch.float32` during inference)
    - `max_len`: `int` ($L_{\text{mel}}$) during training; `None` during inference
  - **Outputs**:
    - `output`: `[B, L_mel, 256]` (`torch.float32`), where $L_{\text{mel}} = \text{max\_len}$ if provided, else $\max_b T_b$.
    - `mel_len`: `[B]` (`torch.int64`, allocated on module-global `device`).
- **Training vs. Inference**:
  - **Training (`duration_target is not None`, lines 128–130)**: Uses ground-truth MFA durations `duration_target` (`[B, L_src]`) and dataset `max_mel_len` so `output` exactly matches the padded target Mel-spectrogram length `L_mel`.
  - **Inference (`duration_target is None`, lines 131–138)**: Uses predicted `duration_rounded` (`[B, L_src]`) with `max_len=None`, so `output` is padded dynamically to the longest synthesized utterance in the inference batch, and `mel_mask = get_mask_from_lengths(mel_len)` is constructed from the resulting `mel_len`.

---

### 3.4 Component 4: Pitch Embedding and Quantization (`model/modules.py`, lines 36–59, 73–75, 80–89, 117–121, 139–143)

- **Purpose**: Predicts continuous F0 values (normalized or raw depending on `preprocess.yaml`), discretizes them into `n_bins = 256` buckets, maps bucket indices to learnable `256`-dim embedding vectors, and adds them to the hidden representation `x`.
- **Configuration Parameters**:
  - `preprocess_config["preprocessing"]["pitch"]["feature"]`: `"phoneme_level"` (in `LJSpeech`, `LibriTTS`, `AISHELL3`) or `"frame_level"` (in `LJSpeech_paper`).
  - `preprocess_config["preprocessing"]["pitch"]["normalization"]`: `True` (Z-score normalized F0) or `False` (raw Hz F0).
  - `model_config["variance_embedding"]["pitch_quantization"]`: `"linear"` or `"log"`.
  - `model_config["variance_embedding"]["n_bins"]`: `256`.
- **Execution in `get_pitch_embedding(self, x, target, mask, control)` (`lines 80–89`)**:
  ```python
  def get_pitch_embedding(self, x, target, mask, control):
      prediction = self.pitch_predictor(x, mask)
      if target is not None:
          embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
      else:
          prediction = prediction * control
          embedding = self.pitch_embedding(
              torch.bucketize(prediction, self.pitch_bins)
          )
      return prediction, embedding
  ```
- **Input & Output Tensor Shapes**:
  - **Inputs**: `x` `[B, L, 256]`, `target` `[B, L]` or `None`, `mask` `[B, L]`, `control` (`float`, `p_control`). Here $L = L_{\text{src}}$ if `"phoneme_level"`, or $L = L_{\text{mel}}$ if `"frame_level"`.
  - **Intermediate Bucket Index Tensor**: `torch.bucketize(..., self.pitch_bins)` $\rightarrow$ `[B, L]` (`torch.int64` with values in `0..255`).
  - **Outputs**: `prediction` `[B, L]` (`torch.float32`), `embedding` `[B, L, 256]` (`torch.float32`).
- **Training vs. Inference**:
  - **Training (`target is not None`)**: `self.pitch_predictor(x, mask)` computes `prediction` (returned for `pitch_loss` MSE computation), while **ground-truth `target`** is bucketized to produce `embedding` (teacher forcing). Note: `p_control` is ignored when `target is not None`.
  - **Inference (`target is None`)**: `prediction` is multiplied in-place by `p_control` (`prediction = prediction * control`), and the scaled `prediction` is both bucketized into `embedding` and returned as `pitch_prediction` (`output[2]`).

---

### 3.5 Component 5: Energy Embedding and Quantization (`model/modules.py`, lines 37–40, 46, 60–71, 76–78, 91–100, 122–126, 144–148)

- **Purpose**: Predicts frame-energy values (L2 norm of STFT magnitude, optionally phoneme-averaged and Z-score normalized), discretizes them into `256` buckets using `self.energy_bins`, looks up `self.energy_embedding` (`256 x 256`), and adds the embedding to `x`.
- **Configuration Parameters**:
  - `preprocess_config["preprocessing"]["energy"]["feature"]`: `"phoneme_level"` or `"frame_level"`.
  - `preprocess_config["preprocessing"]["energy"]["normalization"]`: `True` or `False`.
  - `model_config["variance_embedding"]["energy_quantization"]`: `"linear"` or `"log"`.
  - `model_config["variance_embedding"]["n_bins"]`: `256`.
- **Execution in `get_energy_embedding(self, x, target, mask, control)` (`lines 91–100`)**:
  ```python
  def get_energy_embedding(self, x, target, mask, control):
      prediction = self.energy_predictor(x, mask)
      if target is not None:
          embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
      else:
          prediction = prediction * control
          embedding = self.energy_embedding(
              torch.bucketize(prediction, self.energy_bins)
          )
      return prediction, embedding
  ```
- **Input & Output Tensor Shapes**:
  - **Inputs**: `x` `[B, L, 256]`, `target` `[B, L]` or `None`, `mask` `[B, L]`, `control` (`float`).
  - **Outputs**: `prediction` `[B, L]` (`torch.float32`), `embedding` `[B, L, 256]` (`torch.float32`).
- **Training vs. Inference**:
  - **Training (`target is not None`)**: Computes `prediction` for `energy_loss` MSE and bucketizes ground-truth `energy_target` for `embedding`.
  - **Inference (`target is None`)**: Scales `prediction = prediction * control` and bucketizes it. *(See Confirmed Bug A below: `VarianceAdaptor.forward` passes `p_control` instead of `e_control` in both `phoneme_level` and `frame_level` branches!)*

---

### 3.6 Component 6: Duration Prediction and Rounding (`model/modules.py`, lines 22, 116, 128–138)

- **Purpose**: Predicts the log-scale duration $\hat{d}_{\text{log}} = \log(\hat{d} + 1)$ of each phoneme token and converts it during inference into non-negative frame counts `duration_rounded` for `LengthRegulator`.
- **Mathematical Formulation**:
  - **Training Target (`model/loss.py`, line 42)**:
    $$d_{\text{log\_target}} = \log\left(d_{\text{target}} + 1\right)$$
    where $d_{\text{target}} \in \mathbb{Z}_{\ge 0}$ is the integer Mel-frame duration extracted from the MFA `TextGrid`.
  - **Inference Inverse Transform (`model/modules.py`, lines 132–135)**:
    $$\hat{d}_{\text{rounded}} = \max\left(0,\; \text{round}\left(\exp(\hat{d}_{\text{log}}) - 1\right) \cdot d_{\text{control}}\right)$$
- **Input & Output Tensor Shapes**:
  - `log_duration_prediction`: `[B, L_src]` (`torch.float32`)
  - `duration_rounded`:
    - Training (`line 130`): `duration_rounded = duration_target` $\rightarrow$ `[B, L_src]` (`torch.int64`).
    - Inference (`lines 132–135`): `[B, L_src]` (`torch.float32`).

---

## 4. Detailed Investigation of the Five Target Issues (A–E)

### Investigation A: Verification of the Energy-Control Bug (`p_control` vs. `e_control`) — **CONFIRMED BUG**

Tracing `VarianceAdaptor.forward()` in `original_fastspeech2/model/modules.py` (lines 102–158):
- Line 111–113 defines the control parameters:
  ```python
  p_control=1.0,
  e_control=1.0,
  d_control=1.0,
  ```
- **Occurrence 1 — Phoneme-Level Energy (`lines 122–126`)**:
  ```python
  if self.energy_feature_level == "phoneme_level":
      energy_prediction, energy_embedding = self.get_energy_embedding(
          x, energy_target, src_mask, p_control
      )
      x = x + energy_embedding
  ```
- **Occurrence 2 — Frame-Level Energy (`lines 144–148`)**:
  ```python
  if self.energy_feature_level == "frame_level":
      energy_prediction, energy_embedding = self.get_energy_embedding(
          x, energy_target, mel_mask, p_control
      )
      x = x + energy_embedding
  ```
- **Verdict**:
  1. **Confirmed in BOTH branches (`line 124` and `line 146`)**: `p_control` is mistakenly passed instead of `e_control` in **both** `phoneme_level` and `frame_level` energy blocks.
  2. The parameter `e_control` (line 112) is **completely unused** anywhere inside `VarianceAdaptor.forward()`.
  3. **Impact**:
     - **Training**: Unaffected, because `energy_target` is not `None`, so `get_energy_embedding` (line 93) executes `if target is not None:` and ignores `control`.
     - **Inference**: Passing `--energy_control 0.8` to `synthesize.py` (as advertised in `README.md` line 66!) has **zero effect** on the synthesized audio or predicted energy. Conversely, passing `--pitch_control 1.2` unintentionally scales **both** pitch and energy predictions by `1.2`.

---

### Investigation B: Verification of Device Mismatch Risk — **CONFIRMED BUG ON MPS / NON-DEFAULT DEVICES**

Every tensor allocation in `original_fastspeech2/model/modules.py` and its imported helpers in `original_fastspeech2/utils/tools.py` was traced:

| File & Line | Code Statement | Device Used | Input Tensor Device | Risk & Failure Mode |
|---|---|---|---|---|
| `model/modules.py` line 14 | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Module-global `device` (`"cpu"` on Apple Silicon M2) | N/A | Hardcodes `"cuda"` or `"cpu"` at import time; ignores `"mps"` or explicit device passed to `model.to(device)`. |
| `model/modules.py` line 180 (`LengthRegulator.LR`) | `return output, torch.LongTensor(mel_len).to(device)` | Module-global `device` (`"cpu"` on M2 Mac) | `x.device` (e.g., `"mps:0"`) | `output` is on `x.device` (`"mps:0"`), but `mel_len` is placed on `"cpu"`. |
| `utils/tools.py` line 15 | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Module-global `device` (`"cpu"` on M2 Mac) | N/A | Same module-global `"cuda"`/`"cpu"` hardcoding. |
| `utils/tools.py` lines 96–97 (`get_mask_from_lengths`) | `ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)`<br>`mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)` | `ids` is on module-global `device` (`"cpu"` on M2 Mac) | `lengths.device` | **Crash 1 (Training & Inference, `fastspeech2.py` line 59)**: When `src_lens` is on `"mps:0"`, `ids` (`"cpu"`) `>=` `src_lens` (`"mps:0"`) immediately raises `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!`<br>**Crash 2 (Inference, `modules.py` line 137 -> `fastspeech2.py` line 94)**: During inference, `mel_len` (from line 180) is on `"cpu"`, so `mel_mask = get_mask_from_lengths(mel_len)` succeeds on `"cpu"`, returning `mel_mask` on `"cpu"`. Then `Decoder.forward` (`transformer/Layers.py` line 25) runs `enc_output.masked_fill(mask.unsqueeze(-1), 0)` where `enc_output` is on `"mps:0"` and `mask` is on `"cpu"`, crashing with a device mismatch! |
| `model/modules.py` lines 49–71 (`VarianceAdaptor.__init__`) | `self.pitch_bins = nn.Parameter(..., requires_grad=False)`<br>`self.energy_bins = nn.Parameter(..., requires_grad=False)` | Registered as `nn.Parameter` | Moves with `model.to(device)` | **Safe**: Because `pitch_bins` and `energy_bins` are wrapped in `nn.Parameter`, `model.to(device)` moves them to the model's device alongside `x`. |
| `utils/tools.py` lines 308–316 (`pad`) | `F.pad(batch, ..., "constant", 0.0)` and `torch.stack(out_list)` | Inherits `batch.device` (`x.device`) | `x.device` | **Safe**: `F.pad` and `torch.stack` preserve the device of `x`. |

- **Required Fix in `common/`**:
  - In `LengthRegulator.LR()` (`model/modules.py` line 180): change `.to(device)` to `.to(x.device)` (or `torch.tensor(mel_len, dtype=torch.long, device=x.device)`).
  - In `get_mask_from_lengths()` (`utils/tools.py` line 96): change `torch.arange(0, max_len, ...).to(device)` to `torch.arange(0, max_len, device=lengths.device)`.

---

### Investigation C: Pitch & Energy Statistics Loading and Quantization Bin Construction

1. **How `stats.json` Is Generated (`preprocessor/preprocessor.py`, lines 85–135 & 303–314)**:
   - During `Preprocessor.build_from_path()`, `pitch_scaler = StandardScaler()` and `energy_scaler = StandardScaler()` accumulate `partial_fit` over IQR-filtered (`remove_outlier`) pitch and energy values across the corpus.
   - If `normalization: True` (`preprocess.yaml`), `pitch_mean = pitch_scaler.mean_[0]` and `pitch_std = pitch_scaler.scale_[0]`; if `normalization: False`, `pitch_mean = 0, pitch_std = 1`.
   - `self.normalize(dir, mean, std)` (lines 303–314) overwrites every `<speaker>-pitch-<basename>.npy` and `<speaker>-energy-<basename>.npy` file with `(values - mean) / std` and tracks the global post-normalization `min_value` and `max_value`.
   - `stats.json` (lines 119–134) saves:
     ```json
     {
       "pitch": [pitch_min, pitch_max, pitch_mean, pitch_std],
       "energy": [energy_min, energy_max, energy_mean, energy_std]
     }
     ```
2. **How `VarianceAdaptor.__init__` Loads `stats.json` and Builds Bins (`model/modules.py`, lines 41–71)**:
   - Reads only the first two elements (`[:2]`) of each list:
     ```python
     pitch_min, pitch_max = stats["pitch"][:2]
     energy_min, energy_max = stats["energy"][:2]
     ```
   - Actual values in the repository's precomputed `stats.json` files:
     - **LJSpeech**: `pitch_min = -2.9171`, `pitch_max = 11.3913` | `energy_min = -1.4310`, `energy_max = 8.1843`
     - **LibriTTS**: `pitch_min = -2.6463`, `pitch_max = 11.9220` | `energy_min = -1.2487`, `energy_max = 9.7555`
     - **AISHELL3**: `pitch_min = -2.8393`, `pitch_max = 11.0224` | `energy_min = -1.6206`, `energy_max = 6.7718`
   - Builds `n_bins - 1 = 255` bin thresholds:
     - `"linear"`: `torch.linspace(min_val, max_val, 255)`
     - `"log"`: `torch.exp(torch.linspace(np.log(min_val), np.log(max_val), 255))`
   - **Potential Issue Confirmed (`"log"` quantization with non-positive `min_val`)**:
     - When `normalization: True`, `pitch_min` and `energy_min` are negative (around `-2.9` and `-1.4`). If `"log"` quantization is selected, `np.log(pitch_min)` evaluates `np.log` of a negative float, returning `NaN` and corrupting `self.pitch_bins` with `NaN`s without raising an exception!
     - Furthermore, in `config/LJSpeech_paper/model.yaml` (line 19), `pitch_quantization: "log"` is used with `normalization: False` (`config/LJSpeech_paper/preprocess.yaml` line 27). However, for `frame_level` pitch (`LJSpeech_paper/preprocess.yaml` line 26), `Preprocessor.process_utterance` (`preprocessor.py` lines 198–218) **only** performs linear interpolation over unvoiced frames (`pitch == 0`) when `self.pitch_phoneme_averaging` is `True` (`phoneme_level`)! For `frame_level`, unvoiced frames remain `0.0`, so `pitch_min` in `stats.json` is `0.0`, and `np.log(0.0)` evaluates to `-inf`, making `torch.linspace(-inf, ...)` fill `self.pitch_bins` with `NaN`s!

---

### Investigation D: Phoneme-Level vs. Frame-Level Pitch/Energy Configuration Differences

| Dimension | `phoneme_level` (`LJSpeech`, `LibriTTS`, `AISHELL3`) | `frame_level` (`LJSpeech_paper`) |
|---|---|---|
| **Preprocessing (`preprocessor.py` lines 198–229)** | Unvoiced F0 frames (`0`) are linearly interpolated across voiced anchors (`interp1d`), then F0 and energy are averaged over each phoneme's integer frame duration `d[i]`, producing target `.npy` arrays of length `L_src`. | No F0 interpolation and no per-phoneme averaging in `preprocessor.py`; target `.npy` arrays retain full frame length `L_mel` (`sum(duration)`). |
| **Position in `VarianceAdaptor.forward` (`modules.py`)** | Executed at **lines 117–126** (**before** `self.length_regulator`). | Executed at **lines 139–148** (**after** `self.length_regulator`). |
| **Predictor Input & Mask (`modules.py`)** | Input `x` is `[B, L_src, 256]`; mask is `src_mask` `[B, L_src]`. | Input `x` is `[B, L_mel, 256]`; mask is `mel_mask` `[B, L_mel]`. |
| **Prediction & Embedding Shapes** | `pitch_prediction`, `energy_prediction`: `[B, L_src]`<br>`pitch_embedding`, `energy_embedding`: `[B, L_src, 256]` | `pitch_prediction`, `energy_prediction`: `[B, L_mel]`<br>`pitch_embedding`, `energy_embedding`: `[B, L_mel, 256]` |
| **Sequential Coupling Between Pitch & Energy** | At line 121, `x = x + pitch_embedding` mutates `x` **before** line 123 calls `self.get_energy_embedding(x, ...)`. Thus, `energy_predictor` takes `encoder_output + pitch_embedding` as input. | At line 143, `x = x + pitch_embedding` mutates `x` **before** line 145 calls `self.get_energy_embedding(x, ...)`. Thus, `energy_predictor` takes `length_regulated_output + pitch_embedding` as input. |
| **Loss Masking (`model/loss.py` lines 51–64)** | Masked using `src_masks` (`[B, L_src]`). | Masked using `mel_masks` (`[B, L_mel_dec]`). *(Potential bug if `L_mel > max_seq_len` (1000) in training: `Decoder` truncates `mel_masks` to `1000`, whereas `pitch_predictions`/`energy_predictions` from `VarianceAdaptor` have untruncated length `L_mel`, causing `masked_select` in `loss.py` line 55/62 to raise a shape mismatch `RuntimeError`!)* |
| **Inference Visualization (`utils/tools.py` lines 173–182)** | `synth_samples` expands phoneme-level predictions to frame level via `expand(pitch, duration)` (`[mel_len]`) before plotting. | `synth_samples` slices `predictions[2][i, :mel_len]` directly without `expand()`. |

---

### Investigation E: Duration Prediction, Rounding, Clamping, and Length Regulation (Including Zero-Duration Tokens)

1. **Training Behavior (`duration_target is not None`, `model/modules.py` lines 116 & 128–130)**:
   - `log_duration_prediction = self.duration_predictor(x, src_mask)` (`[B, L_src]`).
   - `x, mel_len = self.length_regulator(x, duration_target, max_len)` uses ground-truth integer durations `duration_target` (`torch.int64`).
   - `duration_rounded = duration_target` (`torch.int64`).
   - In `FastSpeech2Loss` (`model/loss.py` lines 42, 65–66, 79), `log_duration_targets = torch.log(duration_targets.float() + 1)` is compared against `log_duration_prediction` via `nn.MSELoss()`. Note that when `duration_target == 0`, $\log(0 + 1) = 0.0$.
2. **Inference Rounding & Clamping (`duration_target is None`, `model/modules.py` lines 132–135)**:
   ```python
   duration_rounded = torch.clamp(
       (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
       min=0,
   )
   ```
   - **Subtle Issue 1 — `d_control` Multiplied *After* `torch.round()`**:
     - Because `* d_control` is outside `torch.round(...)`, whenever `d_control != 1.0`, `duration_rounded` contains fractional floats (e.g., `1.0 * 0.8 = 0.8`, `2.0 * 0.8 = 1.6`).
     - Subsequently, `LengthRegulator.expand()` (line 187) applies `int(expand_size)`, which **floors** positive floats instead of rounding them (`int(0.8) -> 0`, `int(1.6) -> 1`).
     - Consequently, setting `--duration_control 0.8` (as shown in `README.md` line 66) causes every phoneme with a 1-frame predicted duration to be truncated to **`0` frames** (dropped completely)! Applying `torch.round((torch.exp(log_duration_prediction) - 1) * d_control)` (rounding *after* multiplying by `d_control`) avoids this truncation distortion.
   - **Subtle Issue 2 — Dtype Inconsistency of `duration_rounded`**:
     - In training, `duration_rounded` is `torch.int64`; in inference, `duration_rounded` is `torch.float32`.
3. **Zero-Duration Tokens in `LengthRegulator.expand()` (`lines 182–190`)**:
   ```python
   def expand(self, batch, predicted):
       out = list()
       for i, vec in enumerate(batch):
           expand_size = predicted[i].item()
           out.append(vec.expand(max(int(expand_size), 0), -1))
       out = torch.cat(out, 0)
       return out
   ```
   - **Single Token with Duration `0` (including all `<PAD>` tokens)**:
     - For padded tokens (`src_mask == True`), `duration_predictor` masks `log_duration_prediction` to `0.0` (line 248), so $\exp(0.0) - 1 = 0.0$, giving `expand_size = 0`.
     - `vec.expand(0, -1)` creates a valid empty tensor of shape `[0, 256]`.
     - `torch.cat(out, 0)` concatenates `[0, 256]` tensors alongside positive-duration `[d_i, 256]` tensors without error, effectively dropping zero-duration tokens from the frame sequence.
   - **Edge Case — Utterance Where *Every* Token Has Duration `0` (`sum(d_i) == 0`)**:
     - If all tokens in an utterance are predicted to have `0` duration (for instance, a very short input or small `d_control`), `expanded` has shape `[0, 256]` and `mel_len` is `0`.
     - In a single-utterance batch (`B = 1`), `output` from `LengthRegulator` has shape `[1, 0, 256]` and `mel_mask` has shape `[1, 0]`, which will cause downstream convolution layers (`PositionwiseFeedForward`, `PostNet`, and `HiFi-GAN` `Conv1d` with kernel size 7/9) to fail on a zero-length time axis. Clamping the total utterance length or ensuring at least 1 frame when `mel_len == 0` prevents this crash.

---

## 5. Summary Table: Confirmed Bugs vs. Potential Issues in `model/modules.py`

| Category | ID | Location | Description | Severity |
|---|---|---|---|---|
| **Confirmed Bug** | **B1** | `model/modules.py` lines **124** & **146** | `VarianceAdaptor.forward()` passes `p_control` instead of `e_control` to `self.get_energy_embedding()` in **both** `phoneme_level` (line 124) and `frame_level` (line 146) branches. `e_control` (line 112) is completely unused. | **High** (Breaks `--energy_control` and couples energy to `--pitch_control` at inference) |
| **Confirmed Bug** | **B2** | `model/modules.py` line **14** & line **180**; `utils/tools.py` line **15** & line **96** | `LengthRegulator.LR()` and `get_mask_from_lengths()` move newly created tensors (`mel_len`, `ids`) to a module-global `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` instead of `x.device` / `lengths.device`. | **Critical on MPS** (Raises `RuntimeError` device mismatch between `mps:0` and `cpu`) |
| **Confirmed Bug (Latent Config)** | **B3** | `model/modules.py` lines **51** & **63** | `np.log(pitch_min)` and `np.log(energy_min)` in `"log"` quantization produce `NaN` when `normalization: True` (negative `min`) or when `frame_level` unvoiced pitch has `pitch_min == 0.0` (as in `LJSpeech_paper`). | **Medium** (Affects `"log"` quantization mode; standard configs use `"linear"`) |
| **Confirmed Bug (Latent Config)** | **B4** | `model/modules.py` line **230** | `VariancePredictor`'s second conv layer (`conv1d_2`) hardcodes `padding=1` instead of `padding=(self.kernel - 1) // 2` (used in `conv1d_1` at line 218). | **Low/Medium** (Works for `kernel_size=3`, crashes with shape mismatch if `kernel_size != 3`) |
| **Potential Issue** | **P1** | `model/modules.py` lines **132–135** & **187** | `d_control` is multiplied *after* `torch.round()`, producing float `duration_rounded` that `LengthRegulator.expand()` truncates via `int()`, deleting 1-frame phonemes when `d_control < 1.0`. | **Medium** (Distorts speaking-rate control and deletes short phones when `d_control < 1.0`) |
| **Potential Issue** | **P2** | `model/modules.py` lines **139–148** vs. `transformer/Models.py` line **154** & `model/loss.py` lines **55, 62** | In `frame_level` mode, if a training utterance has `L_mel > max_seq_len` (`1000`), `VarianceAdaptor` outputs `[B, L_mel]` pitch/energy predictions while `Decoder` truncates `mel_masks` to `[B, 1000]`, breaking `masked_select` in `FastSpeech2Loss`. | **Low** (Only triggers in `frame_level` mode when `L_mel > 1000`) |
| **Design Note** | **D1** | `model/modules.py` lines **121–125** & **143–147** | `x = x + pitch_embedding` is executed in-place before `self.get_energy_embedding(x, ...)`, meaning `energy_predictor` takes `x + pitch_embedding` rather than raw encoder output `x`. | **Informational** (Intentional architectural choice in `ming024/FastSpeech2`; must be preserved if loading pretrained checkpoints, or documented if kept in `common/`) |

---

## 6. Questions Requiring Further Investigation in Subsequent Steps

1. **Preserving `x = x + pitch_embedding` Order Before `energy_predictor`**:
   - Since we are training our English (Svarah) and Telugu (IndicTTS) models from scratch, we can keep the exact `x = x + pitch_embedding` $\rightarrow$ `get_energy_embedding(x, ...)` flow while fixing `p_control` $\rightarrow$ `e_control` on lines 124 and 146.
2. **Rounding Order for `d_control` in `common/`**:
   - Should `common/model/modules.py` change line 133 from `torch.round(torch.exp(log_duration_prediction) - 1) * d_control` to `torch.round((torch.exp(log_duration_prediction) - 1) * d_control)` so that `duration_rounded` is always an integer count and 1-frame phonemes are rounded rather than floored when `d_control = 0.8`? (When `d_control == 1.0`, both expressions are 100% identical).
3. **Handling `stats.json` During Unit Tests (`tests/`)**:
   - Because `VarianceAdaptor.__init__` opens `<preprocessed_path>/stats.json` on line 42, our upcoming unit tests in `tests/` will either pass a temporary/fixture `stats.json` path in `preprocess_config` or allow `stats` to be passed optionally in `model_config` / `preprocess_config`.

