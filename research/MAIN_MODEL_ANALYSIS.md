# FastSpeech 2 Main Model Technical Analysis (`MAIN_MODEL_ANALYSIS.md`)

## 1. File Location & Scope

- **Analyzed File**: `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/fastspeech2.py` (110 lines)
- **Package Export**: Re-exported as `FastSpeech2` in `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/__init__.py` (line 1).
- **Role in Repository**: Defines the top-level `FastSpeech2(nn.Module)` acoustic model that transforms phoneme/token ID sequences (and optional speaker IDs) into 80-channel Mel-spectrograms while jointly predicting phoneme durations, pitch (F0), and energy.

---

## 2. Main Classes and Methods

`original_fastspeech2/model/fastspeech2.py` defines a single class `FastSpeech2(nn.Module)` with two methods:

| Class / Method | Lines | Signature | Responsibility |
|---|---|---|---|
| `FastSpeech2` | `13–110` | `class FastSpeech2(nn.Module)` | Top-level PyTorch module encapsulating the Encoder, optional Speaker Embedding, Variance Adaptor, Decoder, Mel Linear Projection, and PostNet. |
| `FastSpeech2.__init__` | `16–41` | `def __init__(self, preprocess_config, model_config)` | Initializes all submodules from `preprocess.yaml` and `model.yaml` dicts and conditionally loads `<preprocessed_path>/speakers.json` if `model_config["multi_speaker"]` is `True`. |
| `FastSpeech2.forward` | `43–110` | `def forward(self, speakers, texts, src_lens, max_src_len, mels=None, mel_lens=None, max_mel_len=None, p_targets=None, e_targets=None, d_targets=None, p_control=1.0, e_control=1.0, d_control=1.0)` | Executes the full forward pass for **both** training/evaluation (when ground-truth targets are passed) and inference/synthesis (when targets are `None`). |

> **Note on Inference Method**: There is no separate `inference()` or `synthesize()` method in `FastSpeech2`. Instead, `synthesize.py` (lines 95–100) calls `model(*(batch[2:]), p_control=..., e_control=..., d_control=...)`, which invokes `FastSpeech2.forward()` with `mels=None, mel_lens=None, max_mel_len=None, p_targets=None, e_targets=None, d_targets=None`.

---

## 3. Architecture Explanation (In Execution Order)

### 3.1 Model Initialization (`FastSpeech2.__init__`, lines 16–41)
1. **Configuration Storage (line 18)**: Stores `self.model_config = model_config`.
2. **Encoder (line 20)**: `self.encoder = Encoder(model_config)` (imported from `transformer/Models.py`).
3. **Variance Adaptor (line 21)**: `self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)` (imported from `model/modules.py`).
   - *Filesystem side-effect at init*: `VarianceAdaptor.__init__` (lines 41–46 of `model/modules.py`) opens `os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")` to read `pitch_min, pitch_max` and `energy_min, energy_max` and construct quantization bin boundaries (`self.pitch_bins`, `self.energy_bins`).
4. **Decoder (line 22)**: `self.decoder = Decoder(model_config)` (imported from `transformer/Models.py`).
5. **Mel Linear Projection (lines 23–26)**:
   ```python
   self.mel_linear = nn.Linear(
       model_config["transformer"]["decoder_hidden"],
       preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
   )
   ```
   Maps `decoder_hidden` (`256`) to `n_mel_channels` (`80`).
6. **PostNet (line 27)**: `self.postnet = PostNet()` (imported from `transformer/Layers.py`).
   - Instantiated with default arguments (`n_mel_channels=80, postnet_embedding_dim=512, postnet_kernel_size=5, postnet_n_convolutions=5`).
7. **Multi-Speaker Embedding (lines 29–41)**:
   - Initializes `self.speaker_emb = None`.
   - If `model_config["multi_speaker"]` is `True`: opens `os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")`, counts `n_speaker = len(json.load(f))`, and creates `self.speaker_emb = nn.Embedding(n_speaker, model_config["transformer"]["encoder_hidden"])` (`[n_speaker, 256]`).

### 3.2 Text Embedding & Positional Encoding (Inside `Encoder`, `transformer/Models.py` lines 39–62, 82–92)
- Text embedding is encapsulated inside `self.encoder`:
  - `self.encoder.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)` where `n_src_vocab = len(symbols) + 1 = 361`, `d_word_vec = 256`, and `Constants.PAD = 0` (`transformer/Constants.py` line 1).
  - `self.encoder.position_enc`: Fixed sinusoidal encoding table of shape `[1, max_seq_len + 1, 256]` (`[1, 1001, 256]`, `requires_grad=False`).
  - During `self.encoder(texts, src_masks)`:
    - If `not self.training and src_seq.shape[1] > self.max_seq_len`: dynamically computes a longer sinusoidal table on the fly and moves it to `.to(src_seq.device)`.
    - Otherwise: slices `self.position_enc[:, :max_len, :]`, expands to `[B, L_src, 256]`, and adds it to `self.src_word_emb(src_seq)`.

### 3.3 Encoder (`fastspeech2.py` line 66 -> `transformer/Models.py` lines 33–100)
- Executed via `output = self.encoder(texts, src_masks)`.
- Constructs 3D key-padding attention mask `slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)` (`[B, L_src, L_src]`).
- Passes embeddings sequentially through `encoder_layer` (`4`) `FFTBlock` modules (`transformer/Layers.py` lines 11–30):
  1. `MultiHeadAttention` (`n_head=2`, `d_model=256`, `d_k=d_v=128`, `dropout=0.2`) with post-attention residual connection and `LayerNorm`.
  2. Padded query positions are zeroed via `enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)`.
  3. `PositionwiseFeedForward` (`transformer/SubLayers.py` lines 60–93): two 1D convolutions (`Conv1d(256, 1024, kernel_size=9, padding=4)` + ReLU + `Conv1d(1024, 256, kernel_size=1, padding=0)`) with dropout `0.2`, residual connection, and `LayerNorm`.
  4. Padded positions are again zeroed via `enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)`.
- Returns `output` of shape `[B, L_src, 256]`.

### 3.4 Optional Speaker Embedding Addition (`fastspeech2.py` lines 68–71)
```python
if self.speaker_emb is not None:
    output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
        -1, max_src_len, -1
    )
```
- Looks up speaker vector `self.speaker_emb(speakers)` (`[B, 256]`), unsqueezes time axis to `[B, 1, 256]`, broadcasts across all `max_src_len` tokens (`[B, L_src, 256]`), and adds to `output`.

### 3.5 Variance Adaptor (`fastspeech2.py` lines 73–92 -> `model/modules.py` lines 17–158)
Executed via:
```python
(
    output,
    p_predictions,
    e_predictions,
    log_d_predictions,
    d_rounded,
    mel_lens,
    mel_masks,
) = self.variance_adaptor(
    output, src_masks, mel_masks, max_mel_len,
    p_targets, e_targets, d_targets,
    p_control, e_control, d_control,
)
```
Inside `VarianceAdaptor.forward()`:
1. **Duration Prediction (line 116)**:
   - `log_duration_prediction = self.duration_predictor(x, src_mask)` -> `[B, L_src]`.
2. **Phoneme-Level Pitch & Energy Addition (lines 117–126, active when `feature == "phoneme_level"`)**:
   - Calls `self.get_pitch_embedding(x, pitch_target, src_mask, p_control)` -> predicts `pitch_prediction` (`[B, L_src]`), quantizes either `pitch_target` (training) or `pitch_prediction * p_control` (inference) into `256` bins via `torch.bucketize`, looks up `self.pitch_embedding` (`[B, L_src, 256]`), and adds `x = x + pitch_embedding`.
   - Calls `self.get_energy_embedding(x, energy_target, src_mask, p_control)` -> predicts `energy_prediction` (`[B, L_src]`), quantizes either `energy_target` (training) or `energy_prediction * p_control` (inference; note bug on line 124 where `p_control` is passed instead of `e_control`), looks up `self.energy_embedding` (`[B, L_src, 256]`), and adds `x = x + energy_embedding`.
3. **Length Regulation (lines 128–138)**:
   - **Training (`duration_target is not None`)**: Expands `x` along the time dimension using ground-truth integer frame counts `duration_target` via `self.length_regulator(x, duration_target, max_len)` -> `x` becomes `[B, L_mel, 256]`, `duration_rounded = duration_target`.
   - **Inference (`duration_target is None`)**: Converts log-duration predictions to integer frame counts:
     $$\hat{d}_{\text{rounded}} = \max\left(0,\; \text{round}\left((\exp(\hat{d}_{\text{log}}) - 1)\right) \cdot d_{\text{control}}\right)$$
     Expands `x, mel_len = self.length_regulator(x, duration_rounded, max_len)` and constructs `mel_mask = get_mask_from_lengths(mel_len)` (`[B, L_mel]`).
4. **Frame-Level Pitch & Energy Addition (lines 139–148, active only when `feature == "frame_level"`)**:
   - If configured for `frame_level` (as in `config/LJSpeech_paper/`), predicts and adds `pitch_embedding` (`[B, L_mel, 256]`) and `energy_embedding` (`[B, L_mel, 256]`) *after* length regulation using `mel_mask`.

### 3.6 Decoder (`fastspeech2.py` line 94 -> `transformer/Models.py` lines 103–171)
- Executed via `output, mel_masks = self.decoder(output, mel_masks)`.
- Adds frame-level sinusoidal positional encoding to `output`:
  - If `not self.training and enc_seq.shape[1] > self.max_seq_len`: generates a sinusoidal encoding table of length `L_mel` dynamically so long utterances are not truncated at inference.
  - Otherwise (including all training steps): clamps `max_len = min(max_len, self.max_seq_len)` (`1000` frames) and slices `enc_seq[:, :max_len, :]` and `mask[:, :max_len]`.
- Passes frame sequence through `decoder_layer` (`6` in standard configs, `4` in `LJSpeech_paper`) `FFTBlock` modules (same internal structure as Encoder blocks: 2-head self-attention + `[9, 1]` Conv1D FFN of filter size `1024`, dropout `0.2`).
- Returns `(output, mel_masks)` of shapes `([B, L_mel_dec, 256], [B, L_mel_dec])`.

### 3.7 Mel-Spectrogram Linear Projection (`fastspeech2.py` line 95)
- Executed via `output = self.mel_linear(output)`.
- Applies `nn.Linear(256, 80)` to each frame, producing the coarse (pre-PostNet) Mel-spectrogram prediction `output` of shape `[B, L_mel_dec, 80]`.

### 3.8 PostNet & Residual Addition (`fastspeech2.py` line 97 -> `transformer/Layers.py` lines 67–137)
- Executed via `postnet_output = self.postnet(output) + output`.
- Inside `PostNet.forward(x)`:
  - Transposes `x` from `[B, L_mel_dec, 80]` to `[B, 80, L_mel_dec]`.
  - Applies 4 blocks of `Conv1d(in, 512, kernel_size=5, padding=2) -> BatchNorm1d(512) -> torch.tanh -> F.dropout(0.5, training=self.training)`.
  - Applies 1 final block of `Conv1d(512, 80, kernel_size=5, padding=2) -> BatchNorm1d(80) -> F.dropout(0.5, training=self.training)` (no activation on final layer).
  - Transposes back to `[B, L_mel_dec, 80]`.
- Adds the residual refinement `self.postnet(output)` to the pre-PostNet Mel `output` to form `postnet_output` (`[B, L_mel_dec, 80]`).

---

## 4. Complete Forward-Pass Data Flow & Text Diagram

```text
Inputs:
  texts [B, L_src] (Long)        src_lens [B] (Long), max_src_len (int)
  speakers [B] (Long)                      │
         │                                 ▼
         │                       get_mask_from_lengths()
         │                                 │
         │                                 ▼
         │                       src_masks [B, L_src] (Bool, True=PAD)
         ▼                                 │
┌──────────────────────────────────────────┴──────────────────────────┐
│ 1. Encoder (transformer/Models.py :: Encoder)                       │
│    - src_word_emb: nn.Embedding(361, 256, padding_idx=0)            │
│    - + Sinusoidal position_enc [1, L_src, 256]                      │
│    - 4x FFTBlock (2-Head Self-Attn + Conv1D[9,1] FFN 256->1024->256)│
└──────────────────────────────────────────┬──────────────────────────┘
                                           │
                                           ▼
                                output [B, L_src, 256]
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         │ 2. Optional Speaker Conditioning (if multi_speaker == True)       │
         │    output = output + speaker_emb(speakers).unsqueeze(1)           │
         │                      .expand(-1, L_src, -1)   [B, L_src, 256]     │
         └─────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. Variance Adaptor (model/modules.py :: VarianceAdaptor)                  │
│                                                                            │
│  (a) Duration Predictor (VariancePredictor):                               │
│      output [B, L_src, 256] ──► log_d_predictions [B, L_src]               │
│                                                                            │
│  (b) If phoneme_level Pitch/Energy (default in LJSpeech/LibriTTS/AISHELL3):│
│      - Pitch Predictor ──► p_predictions [B, L_src]                        │
│        bucketize(p_targets OR p_predictions * p_control, pitch_bins)       │
│        ──► pitch_embedding [B, L_src, 256] ──► added to output             │
│      - Energy Predictor ──► e_predictions [B, L_src]                       │
│        bucketize(e_targets OR e_predictions * p_control*, energy_bins)     │
│        ──► energy_embedding [B, L_src, 256] ──► added to output            │
│                                                                            │
│  (c) Length Regulator (LengthRegulator):                                   │
│      - Training: d_rounded = d_targets [B, L_src]                          │
│      - Inference: d_rounded = clamp(round((exp(log_d) - 1) * d_control),0) │
│      Repeats each token i by d_rounded[b, i] frames & pads batch           │
│      ──► output [B, L_mel, 256], mel_lens [B], mel_masks [B, L_mel]        │
│                                                                            │
│  (d) If frame_level Pitch/Energy (LJSpeech_paper config):                  │
│      Predicts & adds pitch_embedding and energy_embedding at [B, L_mel,256]│
└──────────────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
                      output [B, L_mel, 256], mel_masks [B, L_mel]
                                           │
┌──────────────────────────────────────────┴─────────────────────────────────┐
│ 4. Decoder (transformer/Models.py :: Decoder)                              │
│    - Truncates L_mel to max_seq_len (1000) if training or L_mel <= 1000    │
│    - + Sinusoidal position_enc [1, L_mel_dec, 256]                         │
│    - 6x FFTBlock (2-Head Self-Attn + Conv1D[9,1] FFN 256->1024->256)       │
└──────────────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
                             output [B, L_mel_dec, 256]
                                           │
┌──────────────────────────────────────────┴─────────────────────────────────┐
│ 5. Mel Linear Projection (nn.Linear(256, 80))                              │
└──────────────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
                   output (Pre-PostNet Mel) [B, L_mel_dec, 80]
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     ▼
                        │              ┌─────────────────────────────────────┐
                        │              │ 6. PostNet (transformer/Layers.py)  │
                        │              │    5x Conv1d(k=5)+BN (+tanh on 1..4)│
                        │              │    80 -> 512 -> 512 -> 512 -> 512->80│
                        │              └──────────────────────┬──────────────┘
                        │                                     │
                        └──────────────────► (+) ◄────────────┘
                                              │
                                              ▼
                       postnet_output (Final Mel) [B, L_mel_dec, 80]
```

---

## 5. Input and Output Tensor Details

### 5.1 Input Arguments to `FastSpeech2.forward()`

Let $B$ = batch size, $L_{\text{src}}$ = `max_src_len` (maximum phoneme sequence length in batch), and $L_{\text{mel}}$ = `max_mel_len` (maximum Mel frame length in batch).

| Argument | Type & Dtype | Shape (Training / Eval) | Shape / Value (Inference) | Description |
|---|---|---|---|---|
| `speakers` | `torch.LongTensor` | `[B]` | `[B]` | Integer speaker IDs (`0` for single-speaker datasets). |
| `texts` | `torch.LongTensor` | `[B, L_src]` | `[B, L_src]` | Padded phoneme/symbol ID sequences (`0` = `_` PAD). |
| `src_lens` | `torch.LongTensor` | `[B]` | `[B]` | True unpadded token length of each sample in the batch. |
| `max_src_len` | `int` (or `np.int64`) | Scalar `int` ($L_{\text{src}}$) | Scalar `int` ($L_{\text{src}}$) | Maximum value of `src_lens` in the current batch. |
| `mels` | `torch.FloatTensor` or `None` | `[B, L_mel, 80]` | `None` | Ground-truth Mel-spectrograms (**accepted for tuple unpacking convenience in `model(*(batch[2:]))`, but completely unused inside `FastSpeech2.forward()`**). |
| `mel_lens` | `torch.LongTensor` or `None` | `[B]` | `None` | Ground-truth Mel frame lengths per sample. |
| `max_mel_len` | `int` or `None` | Scalar `int` ($L_{\text{mel}}$) | `None` | Maximum Mel frame length in the current batch. |
| `p_targets` | `torch.FloatTensor` or `None` | `[B, L_src]` (if `phoneme_level`) or `[B, L_mel]` (if `frame_level`) | `None` | Ground-truth pitch (F0) targets (Z-score normalized by default). |
| `e_targets` | `torch.FloatTensor` or `None` | `[B, L_src]` (if `phoneme_level`) or `[B, L_mel]` (if `frame_level`) | `None` | Ground-truth STFT magnitude L2-norm energy targets. |
| `d_targets` | `torch.LongTensor` or `None` | `[B, L_src]` | `None` | Ground-truth MFA phoneme durations in Mel frames. |
| `p_control` | `float` | `1.0` (default) | `float` (default `1.0`) | Multiplicative pitch scaling factor during inference. |
| `e_control` | `float` | `1.0` (default) | `float` (default `1.0`) | Multiplicative energy scaling factor during inference. |
| `d_control` | `float` | `1.0` (default) | `float` (default `1.0`) | Multiplicative speaking-rate / duration scaling factor during inference. |

### 5.2 Output 10-Tuple Returned by `FastSpeech2.forward()` (`lines 99–110`)

Let $L_{\text{mel\_dec}} = \min(L_{\text{mel}}, \text{max\_seq\_len})$ during training (or when $L_{\text{mel}} \le 1000$), and $L_{\text{mel\_dec}} = L_{\text{mel}}$ during inference when $L_{\text{mel}} > 1000$.

| Tuple Index | Return Variable | Dtype | Tensor Shape | Consumer in Repository |
|---|---|---|---|---|
| `0` | `output` | `torch.float32` | `[B, L_mel_dec, 80]` | `FastSpeech2Loss` (`mel_predictions`, L1 loss) |
| `1` | `postnet_output` | `torch.float32` | `[B, L_mel_dec, 80]` | `FastSpeech2Loss` (`postnet_mel_predictions`, L1 loss) & `synth_samples` / `vocoder_infer` (final Mel input to vocoder) |
| `2` | `p_predictions` | `torch.float32` | `[B, L_src]` (`phoneme_level`) or `[B, L_mel]` (`frame_level`) | `FastSpeech2Loss` (`pitch_predictions`, MSE loss) & `synth_samples` (F0 contour plot) |
| `3` | `e_predictions` | `torch.float32` | `[B, L_src]` (`phoneme_level`) or `[B, L_mel]` (`frame_level`) | `FastSpeech2Loss` (`energy_predictions`, MSE loss) & `synth_samples` (Energy contour plot) |
| `4` | `log_d_predictions` | `torch.float32` | `[B, L_src]` | `FastSpeech2Loss` (`log_duration_predictions`, MSE loss vs $\log(d+1)$) |
| `5` | `d_rounded` | `torch.int64` (train) / `torch.float32` (infer) | `[B, L_src]` | `synth_samples` (`predictions[5]`, used to expand phoneme-level pitch/energy to frame level) |
| `6` | `src_masks` | `torch.bool` | `[B, L_src]` | `FastSpeech2Loss` (inverted via `~src_masks` for `masked_select`) |
| `7` | `mel_masks` | `torch.bool` | `[B, L_mel_dec]` | `FastSpeech2Loss` (inverted via `~mel_masks` for `masked_select`) |
| `8` | `src_lens` | `torch.int64` | `[B]` | `synth_one_sample` / `synth_samples` (`predictions[8]`, slices unpadded phoneme length) |
| `9` | `mel_lens` | `torch.int64` | `[B]` | `synth_one_sample` / `synth_samples` (`predictions[9]`, slices unpadded Mel frames & audio length `mel_len * hop_length`) |

---

## 6. Important Configuration Parameters Used by `FastSpeech2`

### From `preprocess_config` (`preprocess.yaml`)
- `preprocess_config["preprocessing"]["mel"]["n_mel_channels"]`: Output dimension of `self.mel_linear` (`80`).
- `preprocess_config["path"]["preprocessed_path"]`: Directory containing `speakers.json` (read by `FastSpeech2.__init__` when `multi_speaker: True`) and `stats.json` (read by `VarianceAdaptor.__init__`).
- `preprocess_config["preprocessing"]["pitch"]["feature"]`: `"phoneme_level"` or `"frame_level"` (determines whether pitch is predicted/added before or after `LengthRegulator`).
- `preprocess_config["preprocessing"]["energy"]["feature"]`: `"phoneme_level"` or `"frame_level"` (determines whether energy is predicted/added before or after `LengthRegulator`).

### From `model_config` (`model.yaml`)
- `model_config["multi_speaker"]`: Boolean (`False` in `LJSpeech`, `True` in `LibriTTS`/`AISHELL3`); controls whether `self.speaker_emb` is created and added to encoder outputs.
- `model_config["max_seq_len"]`: Maximum precomputed sinusoidal positional encoding length (`1000`).
- `model_config["transformer"]`:
  - `encoder_layer` (`4`), `encoder_head` (`2`), `encoder_hidden` (`256`), `encoder_dropout` (`0.2`)
  - `decoder_layer` (`6` or `4`), `decoder_head` (`2`), `decoder_hidden` (`256`), `decoder_dropout` (`0.2`)
  - `conv_filter_size` (`1024`), `conv_kernel_size` (`[9, 1]`)
- `model_config["variance_predictor"]`: `filter_size` (`256`), `kernel_size` (`3`), `dropout` (`0.5`).
- `model_config["variance_embedding"]`: `pitch_quantization` (`"linear"` or `"log"`), `energy_quantization` (`"linear"` or `"log"`), `n_bins` (`256`).

---

## 7. How Duration, Pitch, and Energy Enter the Model & Training vs. Inference Differences

| Aspect | Training & Validation (`train.py`, `evaluate.py`) | Inference / Synthesis (`synthesize.py`) |
|---|---|---|
| **PyTorch Module Mode** | `model.train()` in `train.py`; `model.eval()` in `evaluate.py` (note: in `evaluate.py`, targets are still passed to compute validation loss!). | `model.eval()` with `torch.no_grad()`, and targets `mels, mel_lens, max_mel_len, p_targets, e_targets, d_targets` are all `None`. |
| **Mel Frame Mask (`mel_masks`)** | Computed at start of `FastSpeech2.forward()` (lines 60–64) from ground-truth `mel_lens` and `max_mel_len`. | Initially `None` at lines 60–64; computed dynamically inside `VarianceAdaptor.forward()` (line 137) from predicted `mel_len` after `LengthRegulator`. |
| **Duration & Length Regulation** | `DurationPredictor` outputs `log_d_predictions` (used for `duration_loss`), but **teacher forcing** is used for length regulation: `LengthRegulator` expands hidden states using ground-truth `d_targets` (`d_rounded = d_targets`). | No `d_targets`: `d_rounded = clamp(round((exp(log_d_predictions) - 1) * d_control), min=0)` (`FloatTensor`) is used by `LengthRegulator` to expand hidden states. |
| **Pitch Conditioning** | `PitchPredictor` outputs `p_predictions` (for `pitch_loss`), but **teacher forcing** is used for embedding lookup: `torch.bucketize(p_targets, self.pitch_bins)` is passed to `self.pitch_embedding`. | No `p_targets`: predicted pitch is scaled by `p_control` (`prediction = prediction * p_control`) and bucketized (`torch.bucketize(prediction, self.pitch_bins)`) to look up `self.pitch_embedding`. |
| **Energy Conditioning** | `EnergyPredictor` outputs `e_predictions` (for `energy_loss`), and **teacher forcing** is used for embedding lookup: `torch.bucketize(e_targets, self.energy_bins)`. | No `e_targets`: predicted energy is scaled by control factor (bug: `p_control` at `phoneme_level`, `e_control` at `frame_level`) and bucketized (`torch.bucketize(prediction, self.energy_bins)`) to look up `self.energy_embedding`. |
| **Positional Encoding & Sequence Length > `max_seq_len` (1000)** | When `self.training` is `True`, `Decoder.forward()` (lines 154–162 of `transformer/Models.py`) **truncates** `enc_seq` and `mask` to `min(max_len, self.max_seq_len)` (`1000` frames). (`FastSpeech2Loss` lines 43–44 then slices `mel_targets[:, :mel_masks.shape[1], :]` to match). | When `self.training` is `False` and `L_src > 1000` or `L_mel > 1000`, `Encoder.forward()` (line 82) and `Decoder.forward()` (line 145) dynamically compute a larger sinusoid table via `get_sinusoid_encoding_table()` so long utterances are synthesized without truncation. |
| **Dropout & BatchNorm** | Active in `Encoder`, `Decoder`, `VariancePredictor`, and `PostNet` (`F.dropout(..., 0.5, self.training)`). | Disabled because `model.eval()` sets `self.training = False`. |

---

## 8. Imported Dependencies and Their Source Locations

Every symbol imported in `original_fastspeech2/model/fastspeech2.py` (lines 1–10) traced to its exact definition:

| Import Statement in `model/fastspeech2.py` | Imported Symbol | Exact Source File & Lines | Implementation Notes |
|---|---|---|---|
| `import os` | `os.path.join` | Python standard library | Joins `preprocessed_path` and `"speakers.json"`. |
| `import json` | `json.load` | Python standard library | Parses `speakers.json` dictionary. |
| `import torch` | `torch` | PyTorch library | Core tensor library. |
| `import torch.nn as nn` | `nn.Module`, `nn.Linear`, `nn.Embedding` | PyTorch library | Neural network base class and layers. |
| `import torch.nn.functional as F` | `F` | PyTorch library | **Unused import** in `model/fastspeech2.py` (`F` is never referenced in this file). |
| `from transformer import Encoder, Decoder, PostNet` | `Encoder` | `original_fastspeech2/transformer/__init__.py` (line 1) -> `original_fastspeech2/transformer/Models.py` (lines 33–100) | Depends on `FFTBlock` (`transformer/Layers.py`), `Constants.PAD` (`transformer/Constants.py`), and `symbols` (`text/symbols.py`). |
| `from transformer import Encoder, Decoder, PostNet` | `Decoder` | `original_fastspeech2/transformer/__init__.py` (line 1) -> `original_fastspeech2/transformer/Models.py` (lines 103–171) | Depends on `FFTBlock` (`transformer/Layers.py`) and `get_sinusoid_encoding_table` (`transformer/Models.py` lines 10–30). |
| `from transformer import Encoder, Decoder, PostNet` | `PostNet` | `original_fastspeech2/transformer/__init__.py` (line 2) -> `original_fastspeech2/transformer/Layers.py` (lines 67–137) | Depends on `ConvNorm` (`transformer/Layers.py` lines 33–64). |
| `from .modules import VarianceAdaptor` | `VarianceAdaptor` | `original_fastspeech2/model/modules.py` (lines 17–158) | Depends on `VariancePredictor` (lines 197–250), `LengthRegulator` (lines 161–194), `Conv` (lines 253–296), and `get_mask_from_lengths`, `pad` (`utils/tools.py`). |
| `from utils.tools import get_mask_from_lengths` | `get_mask_from_lengths` | `original_fastspeech2/utils/tools.py` (lines 91–99) | Creates boolean mask `ids >= lengths.unsqueeze(1)` using module-global `device`. |

---

## 9. Portability Concerns, Device Assumptions & Coupling Issues

While `model/fastspeech2.py` itself contains no literal `.cuda()` calls, tracing its execution reveals **six important portability and architectural concerns**:

1. **Global `device` Hardcoding in Imported Helpers (`utils/tools.py` & `model/modules.py`)**:
   - `get_mask_from_lengths(lengths, max_len)` in `utils/tools.py` (lines 15 & 96) defines `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` at module import time and executes `ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)`.
   - Similarly, `LengthRegulator.LR()` in `model/modules.py` (lines 14 & 180) executes `torch.LongTensor(mel_len).to(device)` using the module-global `device`.
   - **Impact on Apple Silicon MPS / Multi-Device**: If a `FastSpeech2` model and its input tensors are moved to `mps` (`model.to("mps")`), `get_mask_from_lengths()` and `LengthRegulator.LR()` will still allocate `ids` and `mel_len` on `"cpu"`, causing a **device mismatch `RuntimeError` (`mps:0` vs `cpu`)** at `ids >= lengths.unsqueeze(1)`!
   - **Fix for `common/`**: Derive the device dynamically from the input tensor (`lengths.device` in `get_mask_from_lengths` and `x.device` in `LengthRegulator.LR`).
2. **Filesystem Dependency During Model Instantiation (`__init__`)**:
   - `FastSpeech2.__init__` (lines 31–37) requires `<preprocessed_path>/speakers.json` to exist on disk if `multi_speaker: True`, and `VarianceAdaptor.__init__` (lines 41–44 of `model/modules.py`) unconditionally requires `<preprocessed_path>/stats.json` to exist on disk.
   - **Impact**: You cannot instantiate `FastSpeech2(preprocess_config, model_config)` for a unit test or checkpoint inspection unless `stats.json` (and `speakers.json` for multi-speaker) exists at the path specified in `preprocess_config`.
3. **Global Vocabulary Coupling in `Encoder.__init__` (`transformer/Models.py` line 40)**:
   - `Encoder.__init__` imports `from text.symbols import symbols` directly and sets `n_src_vocab = len(symbols) + 1`.
   - **Impact**: If `text/symbols.py` is modified (e.g., by appending Telugu phoneme symbols), `n_src_vocab` changes, which changes the shape of `encoder.src_word_emb.weight` (`[n_src_vocab, 256]`) and invalidates previously saved checkpoints if `symbols` is not kept consistent or parameterized via `model_config`.
4. **Hardcoded `n_mel_channels=80` in `PostNet` (`fastspeech2.py` line 27)**:
   - Line 23 passes `preprocess_config["preprocessing"]["mel"]["n_mel_channels"]` to `self.mel_linear`, but line 27 calls `self.postnet = PostNet()` without passing `n_mel_channels`, relying on `PostNet.__init__`'s default `n_mel_channels=80`.
5. **Energy Control Bug in `VarianceAdaptor.forward` (`model/modules.py` line 124)**:
   - When `FastSpeech2.forward` passes `e_control` into `self.variance_adaptor(...)` (line 90), `VarianceAdaptor.forward` ignores `e_control` when `self.energy_feature_level == "phoneme_level"` and passes `p_control` instead (`self.get_energy_embedding(x, energy_target, src_mask, p_control)`).
6. **Checkpoint `DataParallel` State Dict Assumption (`train.py` vs `utils/model.py`)**:
   - `train.py` wraps `model = nn.DataParallel(model)` (line 42) and saves `model.module.state_dict()` (line 154), so checkpoints store unprefixed keys (`encoder.src_word_emb.weight`, etc.), which matches `FastSpeech2.load_state_dict()` in `utils/model.py` (line 21). However, `utils/model.py` calls `torch.load(ckpt_path)` without `map_location=device`.

---

## 10. Questions or Uncertainties Requiring Further Inspection in Subsequent Steps

1. **Vocabulary Size Strategy for English vs. Telugu (`common/` design)**:
   - Since `Encoder` sets `n_src_vocab = len(symbols) + 1`, should `common/` use a unified `symbols` list (containing both ARPAbet and Telugu phonemes from the start so both models share the same `n_src_vocab`), or should `n_src_vocab` be configurable per language via `model_config`? (A unified `symbols` list or config-driven `n_src_vocab` both work cleanly; we will confirm this when inspecting the text frontend and `transformer/Models.py`).
2. **Speaker Embedding Handling for Svarah (English) vs. IndicTTS (Telugu)**:
   - When `self.speaker_emb` is added to `output` (`lines 68–71`), padded positions in `output` become non-zero (`speaker_emb(speakers)`). Does `Encoder`/`VariancePredictor`/`LengthRegulator` completely mask out those non-zero padded vectors, or should `output` be re-masked with `src_masks` after adding `speaker_emb`?
     - *Verified from `model/modules.py`*: `VariancePredictor.forward()` masks its output via `out.masked_fill(mask, 0.0)`, and `LengthRegulator.expand()` repeats each position `i` by `predicted[i].item()` frames (which is `0` at masked positions), so padded positions are dropped during length regulation. Still, re-masking or keeping exact parity with `original_fastspeech2` will be decided when we inspect `model/modules.py` in the next step.

