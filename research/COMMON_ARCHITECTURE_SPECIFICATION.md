# Unified FastSpeech 2 Architecture Review & Implementation Specification (`COMMON_ARCHITECTURE_SPECIFICATION.md`)

**Stage**: Stage 2, Step 1.3G  
**Project Root**: `/Users/swaruppal/FastSpeech2_Project`  
**Original Reference Repository**: `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2`  
**Target Shared Architecture Directory**: `/Users/swaruppal/FastSpeech2_Project/common`  
**Synthesized From**:
- [`research/MAIN_MODEL_ANALYSIS.md`](file:///Users/swaruppal/FastSpeech2_Project/research/MAIN_MODEL_ANALYSIS.md)
- [`research/VARIANCE_ADAPTOR_ANALYSIS.md`](file:///Users/swaruppal/FastSpeech2_Project/research/VARIANCE_ADAPTOR_ANALYSIS.md)
- [`research/TRANSFORMER_ANALYSIS.md`](file:///Users/swaruppal/FastSpeech2_Project/research/TRANSFORMER_ANALYSIS.md)
- [`research/ATTENTION_FFT_DEEP_ANALYSIS.md`](file:///Users/swaruppal/FastSpeech2_Project/research/ATTENTION_FFT_DEEP_ANALYSIS.md)
- [`research/POSTNET_ANALYSIS.md`](file:///Users/swaruppal/FastSpeech2_Project/research/POSTNET_ANALYSIS.md)
- [`research/LOSS_FUNCTION_ANALYSIS.md`](file:///Users/swaruppal/FastSpeech2_Project/research/LOSS_FUNCTION_ANALYSIS.md)

---

## PART 1 — COMPLETE DATA FLOW

### 1.1 Global Notation & Conventions

- **Dimensions**:
  - $B$: Batch size (number of utterances in a sub-batch).
  - $L_{\text{src}}$: Maximum phoneme/token sequence length in the batch (`max_src_len = int(src_lens.max())`).
  - $L_{\text{mel}}$: Maximum Mel-spectrogram frame length in the batch (`max_mel_len` from ground-truth targets during training, or $\max_b \sum_i \hat{d}_{b,i}$ from `LengthRegulator` during inference).
  - $L_{\text{mel\_dec}}$: Frame sequence length output by `Decoder`:
    - During training (or when $L_{\text{mel}} \le L_{\max}$): $L_{\text{mel\_dec}} = \min(L_{\text{mel}}, L_{\max})$, where $L_{\max} = \text{max\_seq\_len}$ (default `1000`).
    - During inference when $L_{\text{mel}} > L_{\max}$: $L_{\text{mel\_dec}} = L_{\text{mel}}$ (dynamic sinusoidal encoding table expansion).
  - $H_{\text{enc}}, H_{\text{dec}}$: Transformer hidden dimensions (`encoder_hidden = 256`, `decoder_hidden = 256`; must satisfy $H_{\text{enc}} = H_{\text{dec}} = H$).
  - $C$: Number of Mel-spectrogram channels (`n_mel_channels`, default `80`).
  - $V$: Source vocabulary size (`n_src_vocab`, language-specific: e.g., English ARPAbet vs. Telugu phoneme inventory, where index `0` is reserved for `<PAD>`).
- **Padding Mask Convention (Model-Internal)**:
  - Throughout `FastSpeech2`, `Encoder`, `VarianceAdaptor`, `VariancePredictor`, `FFTBlock`, `MultiHeadAttention`, and `Decoder`, boolean masks (`src_masks`, `mel_masks`, `slf_attn_mask`) use **`True` = padded position (masked out)** and **`False` = valid token/frame**.
  - Inside `FastSpeech2Loss`, masks are bitwise-inverted via `~src_masks` and `~mel_masks` so that **`True` = valid token/frame** for `torch.Tensor.masked_select`.
- **Device Expectation**:
  - **Single-Source-of-Truth Device Rule**: No module or utility in `common/` may read a module-global `device` variable or call `.cuda()`. Every dynamically created tensor (`ids` in `get_mask_from_lengths`, `mel_len` in `LengthRegulator`, dynamic sinusoidal tables in `Encoder`/`Decoder`) **must** inherit its `device` directly from the input tensor (`lengths.device`, `x.device`, `src_seq.device`, `enc_seq.device`), ensuring seamless execution across `cpu`, `mps:0` (Apple Silicon M2), and `cuda`.

---

### 1.2 End-to-End Forward Pass Diagram

```text
Inputs:
  speakers [B] (int64)         texts [B, L_src] (int64)         src_lens [B] (int64)
           │                              │                              │
           │                              │                              ▼
           │                              │                 get_mask_from_lengths(src_lens)
           │                              │                              │
           │                              │                              ▼
           │                              │                src_masks [B, L_src] (bool, True=PAD)
           │                              ▼                              │
           │               ┌──────────────┴──────────────────────────────┴──────────────┐
           │               │ Stage 1: Encoder                                           │
           │               │  - Token Embedding: nn.Embedding(V, H, padding_idx=0)      │
           │               │  - + Sinusoidal Positional Encoding [1, L_src, H]          │
           │               │  - slf_attn_mask = src_masks.unsqueeze(1).expand(-1,L,-1)  │
           │               │  - N_enc x FFTBlock (MHA + mask_zero + ConvFFN + mask_zero)│
           │               └──────────────────────────────┬─────────────────────────────┘
           │                                              │
           │                                              ▼
           │                                 enc_output [B, L_src, H] (float32)
           │                                              │
           ▼                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Optional Speaker Embedding (if multi_speaker == True)                        │
│  - speaker_vec = speaker_emb(speakers).unsqueeze(1).expand(-1, L_src, -1) [B,L_src,H] │
│  - enc_output = enc_output + speaker_vec                                              │
└─────────────────────────────────────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Variance Adaptor (Pre-Regulator & Phoneme-Level Branch)                      │
│  3a. Duration Predictor:                                                              │
│      log_d_predictions = duration_predictor(x, src_masks)  ──► [B, L_src] (float32)   │
│  3b. Phoneme-Level Pitch (if pitch.feature == "phoneme_level"):                       │
│      p_predictions = pitch_predictor(x, src_masks)         ──► [B, L_src] (float32)   │
│      p_idx = bucketize(p_targets if train else p_predictions * p_control, pitch_bins) │
│      x = x + pitch_embedding(p_idx)                        ──► [B, L_src, H]          │
│  3c. Phoneme-Level Energy (if energy.feature == "phoneme_level"):                     │
│      e_predictions = energy_predictor(x, src_masks)        ──► [B, L_src] (float32)   │
│      e_idx = bucketize(e_targets if train else e_predictions * e_control, energy_bins)│
│      x = x + energy_embedding(e_idx)                       ──► [B, L_src, H]          │
└─────────────────────────────────────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Length Regulator (inside Variance Adaptor)                                   │
│  - Training (d_targets != None):                                                      │
│      d_rounded = d_targets                                 ──► [B, L_src] (int64)     │
│      x, mel_lens = length_regulator(x, d_rounded, max_mel_len)                        │
│  - Inference (d_targets == None):                                                     │
│      d_rounded = clamp(round((exp(log_d_predictions) - 1) * d_control), min=0).long() │
│      x, mel_lens = length_regulator(x, d_rounded, max_len=None)                       │
│      mel_masks = get_mask_from_lengths(mel_lens)           ──► [B, L_mel] (bool)      │
│  - Frame-Level Pitch/Energy (if feature == "frame_level"):                            │
│      Predict & add pitch_embedding / energy_embedding using mel_masks at [B, L_mel, H]│
└─────────────────────────────────────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                                 x [B, L_mel, H] (float32), mel_masks [B, L_mel] (bool)
                                                          │
                                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Decoder                                                                      │
│  - If training or L_mel <= max_seq_len: truncate L_mel_dec = min(L_mel, max_seq_len)  │
│  - Else (inference & L_mel > max_seq_len): L_mel_dec = L_mel (dynamic sinusoid table) │
│  - + Sinusoidal Positional Encoding [1, L_mel_dec, H]                                 │
│  - slf_attn_mask = mel_masks.unsqueeze(1).expand(-1, L_mel_dec, -1)                   │
│  - N_dec x FFTBlock (MHA + mask_zero + ConvFFN + mask_zero)                           │
└─────────────────────────────────────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                       dec_output [B, L_mel_dec, H] (float32), mel_masks [B, L_mel_dec]
                                                          │
                                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 6: Mel Linear Projection                                                        │
│  - mel_output = mel_linear(dec_output)                     ──► [B, L_mel_dec, C]      │
└─────────────────────────────────────────────────────────┬─────────────────────────────┘
                                                          │
                                      ┌───────────────────┴───────────────────┐
                                      │                                       ▼
                                      │             ┌───────────────────────────────────┐
                                      │             │ Stage 7: PostNet                  │
                                      │             │  5x (Conv1d k=5 + BN1d            │
                                      │             │      + tanh[0..3] + dropout(0.5)) │
                                      │             │  Residual ΔMel [B, L_mel_dec, C]  │
                                      │             └─────────────────┬─────────────────┘
                                      │                               │
                                      └─────────────► (+) ◄───────────┘
                                                       │
                                                       ▼
                                    postnet_output [B, L_mel_dec, C] (float32)
```

---

### 1.3 Stage-by-Stage Tensor Specification

#### Stage 0: Input Preparation & Source Mask Creation
- **Source**: [`original_fastspeech2/model/fastspeech2.py` (lines 59–64)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/fastspeech2.py#L59-L64), [`original_fastspeech2/utils/tools.py` (lines 91–99)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/utils/tools.py#L91-L99)
- **Inputs**:
  - `speakers`: `LongTensor[B]` (`torch.int64`) — Speaker index per utterance (`0` for single-speaker).
  - `texts`: `LongTensor[B, L_src]` (`torch.int64`) — Padded phoneme/token IDs (`0 = PAD`).
  - `src_lens`: `LongTensor[B]` (`torch.int64`) — Unpadded token count per utterance ($1 \le \text{src\_lens}_b \le L_{\text{src}}$).
  - `max_src_len`: `int` (optional, defaults to `int(src_lens.max().item())`).
  - `mel_lens`: `LongTensor[B]` (`torch.int64`) or `None` (provided during training/validation; `None` during inference).
  - `max_mel_len`: `int` or `None` (provided during training/validation; `None` during inference).
- **Outputs**:
  - `src_masks`: `BoolTensor[B, L_src]` (`torch.bool`, `True` where index $\ge \text{src\_lens}_b$, on `src_lens.device`).
  - `mel_masks`: `BoolTensor[B, L_mel]` (`torch.bool`, `True` where index $\ge \text{mel\_lens}_b$, on `mel_lens.device`) if `mel_lens is not None`, else `None`.
- **Downstream Consumers**:
  - `src_masks` $\rightarrow$ `Encoder`, `VarianceAdaptor`, `FastSpeech2` return tuple index `6` ($\rightarrow$ `FastSpeech2Loss`).
  - `mel_masks` $\rightarrow$ `VarianceAdaptor`, `Decoder`, `FastSpeech2` return tuple index `7` ($\rightarrow$ `FastSpeech2Loss`).

#### Stage 1: Encoder
- **Source**: [`original_fastspeech2/transformer/Models.py` (lines 33–100)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Models.py#L33-L100)
- **Inputs**:
  - `src_seq` (`texts`): `LongTensor[B, L_src]` (`torch.int64`).
  - `mask` (`src_masks`): `BoolTensor[B, L_src]` (`torch.bool`, `True` = PAD).
  - `return_attns`: `bool` (default `False`).
- **Internal Transformations**:
  1. `slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)` $\rightarrow$ `BoolTensor[B, L_src, L_src]`.
  2. Token lookup `self.src_word_emb(src_seq)` (`FloatTensor[B, L_src, H]`) + sinusoidal positional encoding `self.position_enc[:, :max_len, :]` (`FloatTensor[1, L_src, H]`).
  3. Passes through `encoder_layer` (`4`) `FFTBlock` modules; each block zeroes padded positions (`mask.unsqueeze(-1)`) after `MultiHeadAttention` and after `PositionwiseFeedForward`.
- **Outputs**:
  - `enc_output`: `FloatTensor[B, L_src, H]` (`torch.float32`, `0.0` at padded positions, on `src_seq.device`).
  - *(Optional if `return_attns=True`)*: `enc_slf_attn_list` (`list` of `encoder_layer` tensors, each `FloatTensor[n_head * B, L_src, L_src]`).
- **Downstream Consumers**:
  - `enc_output` $\rightarrow$ Stage 2 (Speaker Embedding) or directly Stage 3 (`VarianceAdaptor`).

#### Stage 2: Speaker Embedding (Optional)
- **Source**: [`original_fastspeech2/model/fastspeech2.py` (lines 29–41, 68–71)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/fastspeech2.py#L29-L41)
- **Inputs**:
  - `output` (`enc_output`): `FloatTensor[B, L_src, H]` (`torch.float32`).
  - `speakers`: `LongTensor[B]` (`torch.int64`).
- **Operation** (when `self.speaker_emb is not None`):
  - `speaker_vec = self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)` (`FloatTensor[B, L_src, H]`).
  - `output = output + speaker_vec`.
- **Outputs**:
  - `output`: `FloatTensor[B, L_src, H]` (`torch.float32`, on `texts.device`).
- **Downstream Consumers**:
  - `output` $\rightarrow$ Stage 3 (`VarianceAdaptor`).

#### Stage 3: Variance Adaptor (Duration, Pitch, and Energy Prediction & Embedding)
- **Source**: [`original_fastspeech2/model/modules.py` (lines 17–158)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py#L17-L158)
- **Inputs**:
  - `x` (`output`): `FloatTensor[B, L_src, H]` (`torch.float32`).
  - `src_mask`: `BoolTensor[B, L_src]` (`torch.bool`, `True` = PAD).
  - `mel_mask`: `BoolTensor[B, L_mel]` (`torch.bool`) or `None`.
  - `max_len` (`max_mel_len`): `int` or `None`.
  - `pitch_target`: `FloatTensor[B, L_src]` (phoneme-level) or `FloatTensor[B, L_mel]` (frame-level) or `None`.
  - `energy_target`: `FloatTensor[B, L_src]` (phoneme-level) or `FloatTensor[B, L_mel]` (frame-level) or `None`.
  - `duration_target`: `LongTensor[B, L_src]` (`torch.int64`) or `None`.
  - `p_control`, `e_control`, `d_control`: `float` (default `1.0`).
- **Internal Sequence**:
  1. `log_duration_prediction = self.duration_predictor(x, src_mask)` $\rightarrow$ `FloatTensor[B, L_src]`.
  2. If `pitch_feature_level == "phoneme_level"`:
     - `pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)` $\rightarrow$ `([B, L_src], [B, L_src, H])`.
     - `x = x + pitch_embedding`.
  3. If `energy_feature_level == "phoneme_level"`:
     - `energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, e_control)` *(fixed from `p_control`)* $\rightarrow$ `([B, L_src], [B, L_src, H])`.
     - `x = x + energy_embedding`.
  4. **Stage 4 (`LengthRegulator`)** runs here (see below), transforming `x` from `[B, L_src, H]` to `[B, L_mel, H]`.
  5. If `pitch_feature_level == "frame_level"`:
     - `pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)` $\rightarrow$ `([B, L_mel], [B, L_mel, H])`.
     - `x = x + pitch_embedding`.
  6. If `energy_feature_level == "frame_level"`:
     - `energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mel_mask, e_control)` *(fixed from `p_control`)* $\rightarrow$ `([B, L_mel], [B, L_mel, H])`.
     - `x = x + energy_embedding`.

#### Stage 4: Length Regulator
- **Source**: [`original_fastspeech2/model/modules.py` (lines 128–138, 161–194)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py#L128-L194)
- **Inputs**:
  - `x`: `FloatTensor[B, L_src, H]` (`torch.float32`).
  - `duration`: `LongTensor[B, L_src]` (`torch.int64`, either `duration_target` during training or rounded/clamped inference durations `d_rounded`).
  - `max_len`: `int` or `None`.
- **Operation**:
  - For each utterance $b$ and phoneme $i$, repeats $x_{b,i} \in \mathbb{R}^H$ for $\max(\text{int}(d_{b,i}), 0)$ frames, concatenates along time to length $T_b = \sum_i d_{b,i}$, and zero-pads all utterances in the batch to $L_{\text{mel}} = \text{max\_len}$ (if given) or $\max_b T_b$.
- **Outputs (of `VarianceAdaptor`, including `LengthRegulator`)**:
  - `x`: `FloatTensor[B, L_mel, H]` (`torch.float32`, on `x.device`).
  - `pitch_prediction`: `FloatTensor[B, L_src]` or `FloatTensor[B, L_mel]` (`torch.float32`).
  - `energy_prediction`: `FloatTensor[B, L_src]` or `FloatTensor[B, L_mel]` (`torch.float32`).
  - `log_duration_prediction`: `FloatTensor[B, L_src]` (`torch.float32`).
  - `duration_rounded`: `LongTensor[B, L_src]` (`torch.int64`).
  - `mel_len`: `LongTensor[B]` (`torch.int64`, on `x.device`).
  - `mel_mask`: `BoolTensor[B, L_mel]` (`torch.bool`, `True` = PAD, on `x.device`).
- **Downstream Consumers**:
  - `x`, `mel_mask` $\rightarrow$ Stage 5 (`Decoder`).
  - `pitch_prediction`, `energy_prediction`, `log_duration_prediction`, `duration_rounded`, `mel_len` $\rightarrow$ `FastSpeech2` return tuple indices `2, 3, 4, 5, 9` ($\rightarrow$ `FastSpeech2Loss` and inference synthesis).

#### Stage 5: Decoder
- **Source**: [`original_fastspeech2/transformer/Models.py` (lines 103–171)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Models.py#L103-L171)
- **Inputs**:
  - `enc_seq` (`x` from `VarianceAdaptor`): `FloatTensor[B, L_mel, H]` (`torch.float32`).
  - `mask` (`mel_masks`): `BoolTensor[B, L_mel]` (`torch.bool`, `True` = PAD).
  - `return_attns`: `bool` (default `False`).
- **Internal Sequence**:
  - If `not self.training and enc_seq.shape[1] > self.max_seq_len`: preserves full $L_{\text{mel\_dec}} = L_{\text{mel}}$ and builds a dynamic sinusoidal table of length $L_{\text{mel}}$ on `enc_seq.device`.
  - Else: truncates $L_{\text{mel\_dec}} = \min(L_{\text{mel}}, \text{max\_seq\_len})$, slicing both `enc_seq[:, :L_mel_dec, :]` and `mask[:, :L_mel_dec]`.
  - Adds sinusoidal positional encoding and passes through `decoder_layer` (`6`) `FFTBlock` modules with bidirectional self-attention over non-padded frames and `masked_fill(mask.unsqueeze(-1), 0)` after each sublayer.
- **Outputs**:
  - `dec_output`: `FloatTensor[B, L_mel_dec, H]` (`torch.float32`, `0.0` at padded frames, on `enc_seq.device`).
  - `mask` (`mel_masks`): `BoolTensor[B, L_mel_dec]` (`torch.bool`, `True` = PAD).
  - *(Optional if `return_attns=True`)*: `dec_slf_attn_list` (`list` of `decoder_layer` tensors, each `FloatTensor[n_head * B, L_mel_dec, L_mel_dec]`).
- **Downstream Consumers**:
  - `dec_output` $\rightarrow$ Stage 6 (`mel_linear`).
  - `mel_masks` $\rightarrow$ `FastSpeech2` return tuple index `7` ($\rightarrow$ `FastSpeech2Loss`).

#### Stage 6: Mel-Spectrogram Linear Projection
- **Source**: [`original_fastspeech2/model/fastspeech2.py` (lines 23–26, 95)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/fastspeech2.py#L23-L26)
- **Inputs**:
  - `dec_output`: `FloatTensor[B, L_mel_dec, H]` (`torch.float32`).
- **Operation**:
  - `output = self.mel_linear(dec_output)` (`nn.Linear(H, C)`).
- **Outputs**:
  - `output` (Initial Mel prediction): `FloatTensor[B, L_mel_dec, C]` (`torch.float32`, on `dec_output.device`).
- **Downstream Consumers**:
  - Passed into Stage 7 (`PostNet`), added as the residual base (`self.postnet(output) + output`), and returned at `FastSpeech2` return tuple index `0` ($\rightarrow$ `FastSpeech2Loss` for `mel_loss`).

#### Stage 7: PostNet & Residual Refinement
- **Source**: [`original_fastspeech2/transformer/Layers.py` (lines 67–137)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Layers.py#L67-L137), [`original_fastspeech2/model/fastspeech2.py` (line 97)](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/fastspeech2.py#L97)
- **Inputs**:
  - `output`: `FloatTensor[B, L_mel_dec, C]` (`torch.float32`).
- **Operation**:
  - `PostNet` transposes `[B, L_mel_dec, C]` $\rightarrow$ `[B, C, L_mel_dec]`, applies 4 layers of `Conv1d(k=5, p=2) -> BatchNorm1d(512) -> tanh -> dropout(0.5)` and 1 final layer of `Conv1d(512, C, k=5, p=2) -> BatchNorm1d(C) -> dropout(0.5)`, and transposes back to `[B, L_mel_dec, C]`.
  - Residual addition: `postnet_output = self.postnet(output) + output`.
- **Outputs**:
  - `postnet_output` (Refined Mel prediction): `FloatTensor[B, L_mel_dec, C]` (`torch.float32`, on `output.device`).
- **Downstream Consumers**:
  - Returned at `FastSpeech2` return tuple index `1`:
    - Supervised by `FastSpeech2Loss` (`postnet_mel_loss`).
    - Used at inference/evaluation as the final acoustic Mel-spectrogram fed to the HiFi-GAN vocoder and spectrogram plotter.

#### Stage 8: Final Outputs of `FastSpeech2.forward()`
Returns a 10-tuple matching the original repository contract:

| Index | Name | Dtype | Shape | Used By |
| :--- | :--- | :--- | :--- | :--- |
| `0` | `output` (`mel_predictions`) | `torch.float32` | `[B, L_mel_dec, C]` | `FastSpeech2Loss` (`mel_loss`) |
| `1` | `postnet_output` (`postnet_mel_predictions`) | `torch.float32` | `[B, L_mel_dec, C]` | `FastSpeech2Loss` (`postnet_mel_loss`), Vocoder, Spectrogram Plotter |
| `2` | `p_predictions` | `torch.float32` | `[B, L_src]` or `[B, L_mel]` | `FastSpeech2Loss` (`pitch_loss`), Pitch Plotter |
| `3` | `e_predictions` | `torch.float32` | `[B, L_src]` or `[B, L_mel]` | `FastSpeech2Loss` (`energy_loss`), Energy Plotter |
| `4` | `log_d_predictions` | `torch.float32` | `[B, L_src]` | `FastSpeech2Loss` (`duration_loss`) |
| `5` | `d_rounded` | `torch.int64` | `[B, L_src]` | Pitch/Energy frame expansion in visualization |
| `6` | `src_masks` | `torch.bool` | `[B, L_src]` | `FastSpeech2Loss` (`~src_masks`) |
| `7` | `mel_masks` | `torch.bool` | `[B, L_mel_dec]` | `FastSpeech2Loss` (`~mel_masks`) |
| `8` | `src_lens` | `torch.int64` | `[B]` | Unpadded length slicing |
| `9` | `mel_lens` | `torch.int64` | `[B]` | Unpadded Mel & waveform length slicing |

---

## PART 2 — INTERFACE CONTRACTS

Below are the exact constructor (`__init__`) and `forward` signatures for all 11 core classes, explicitly contrasting the **Original Repository (`original_fastspeech2/`)** against the **Proposed Shared Implementation (`common/`)**.

---

### 2.1 `Encoder`
- **Original (`original_fastspeech2/transformer/Models.py:33–100`)**:
  - `__init__(self, config)`: Hardcodes `n_src_vocab = len(symbols) + 1` via global import `from text.symbols import symbols`.
  - `forward(self, src_seq, mask, return_attns=False) -> Tensor`: Always returns only `enc_output`, silently discarding `enc_slf_attn_list` even when `return_attns=True`.
- **Proposed (`common/transformer/models.py`)**:
  ```python
  class Encoder(nn.Module):
      def __init__(self, config: dict, n_src_vocab: int | None = None) -> None: ...
      def forward(
          self,
          src_seq: torch.Tensor,          # LongTensor [B, L_src]
          mask: torch.Tensor | None = None, # BoolTensor [B, L_src] (True = PAD)
          return_attns: bool = False,
      ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: ...
  ```
- **Key Differences**:
  1. `n_src_vocab` is read from `n_src_vocab` argument or `config["n_src_vocab"]` (or `config["transformer"]["n_src_vocab"]`), decoupling `Encoder` from any global `text/symbols.py` import so English and Telugu models can have independent vocabulary sizes.
  2. Rejects empty sequences (`src_seq.shape[1] == 0`) with a clear `ValueError`.
  3. When `return_attns=False` (default), returns `enc_output` (`[B, L_src, H]`), preserving 100% backward compatibility with `FastSpeech2.forward`. When `return_attns=True`, returns `(enc_output, enc_slf_attn_list)`.
  4. Handles `mask=None` cleanly by constructing an all-`False` mask on `src_seq.device`.

---

### 2.2 `MultiHeadAttention`
- **Original (`original_fastspeech2/transformer/SubLayers.py:8–57`)**:
  - `__init__(self, n_head, d_model, d_k, d_v, dropout=0.1)`
  - `forward(self, q, k, v, mask=None) -> tuple[Tensor, Tensor]`: Crashes with `AttributeError` when `mask=None` because line 46 unconditionally calls `mask.repeat(n_head, 1, 1)`. Also produces `NaN` in `ScaledDotProductAttention` if an entire key row in `mask` is `True`.
- **Proposed (`common/transformer/sublayers.py`)**:
  ```python
  class MultiHeadAttention(nn.Module):
      def __init__(
          self,
          n_head: int,
          d_model: int,
          d_k: int,
          d_v: int,
          dropout: float = 0.1,
      ) -> None: ...

      def forward(
          self,
          q: torch.Tensor,                  # FloatTensor [B, L_q, d_model]
          k: torch.Tensor,                  # FloatTensor [B, L_k, d_model]
          v: torch.Tensor,                  # FloatTensor [B, L_v, d_model]
          mask: torch.Tensor | None = None, # BoolTensor  [B, L_q, L_k] (True = PAD)
      ) -> tuple[torch.Tensor, torch.Tensor]: ...
  ```
- **Key Differences**:
  1. Guards `if mask is not None: mask = mask.repeat(n_head, 1, 1)` so `mask=None` works without error.
  2. Inside `ScaledDotProductAttention` (`common/transformer/modules.py`), guards against all-masked rows (where `mask.all(dim=-1, keepdim=True)` is `True`) by zeroing `attn` post-softmax where a row is completely masked (`attn = torch.nan_to_num(attn, nan=0.0)` or pre-softmax row guard), preventing `0 / 0 = NaN` from polluting `attn` or backward gradients.

---

### 2.3 `PositionwiseFeedForward`
- **Original (`original_fastspeech2/transformer/SubLayers.py:60–93`)**:
  - `__init__(self, d_in, d_hid, kernel_size, dropout=0.1)`
  - `forward(self, x) -> Tensor`
- **Proposed (`common/transformer/sublayers.py`)**:
  ```python
  class PositionwiseFeedForward(nn.Module):
      def __init__(
          self,
          d_in: int,
          d_hid: int,
          kernel_size: list[int] | tuple[int, int] = (9, 1),
          dropout: float = 0.1,
      ) -> None: ...

      def forward(self, x: torch.Tensor) -> torch.Tensor: ... # [B, L, d_in] -> [B, L, d_in]
  ```
- **Key Differences**:
  1. Validates in `__init__` that both `kernel_size[0]` and `kernel_size[1]` are odd positive integers (`k % 2 == 1`) so `(k - 1) // 2` same-padding is guaranteed to preserve sequence length $L$.
  2. Preserves exact parameter names (`w_1`, `w_2`, `layer_norm`, `dropout`).

---

### 2.4 `FFTBlock`
- **Original (`original_fastspeech2/transformer/Layers.py:11–30`)**:
  - `__init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1)`
  - `forward(self, enc_input, mask=None, slf_attn_mask=None) -> tuple[Tensor, Tensor]`: Crashes with `AttributeError` if `mask=None` due to unconditional `mask.unsqueeze(-1)` at lines 25 and 28.
- **Proposed (`common/transformer/layers.py`)**:
  ```python
  class FFTBlock(nn.Module):
      def __init__(
          self,
          d_model: int,
          n_head: int,
          d_k: int,
          d_v: int,
          d_inner: int,
          kernel_size: list[int] | tuple[int, int],
          dropout: float = 0.1,
      ) -> None: ...

      def forward(
          self,
          enc_input: torch.Tensor,                   # FloatTensor [B, L, d_model]
          mask: torch.Tensor | None = None,          # BoolTensor  [B, L] (True = PAD)
          slf_attn_mask: torch.Tensor | None = None, # BoolTensor  [B, L, L] (True = PAD)
      ) -> tuple[torch.Tensor, torch.Tensor]: ...
  ```
- **Key Differences**:
  1. Wraps lines 25 and 28 in `if mask is not None:` before calling `enc_output.masked_fill(mask.unsqueeze(-1), 0)`.
  2. Preserves exact submodule names (`slf_attn`, `pos_ffn`).

---

### 2.5 `Decoder`
- **Original (`original_fastspeech2/transformer/Models.py:103–171`)**:
  - `__init__(self, config)`
  - `forward(self, enc_seq, mask, return_attns=False) -> tuple[Tensor, Tensor]`: Discards `dec_slf_attn_list` even when `return_attns=True`.
- **Proposed (`common/transformer/models.py`)**:
  ```python
  class Decoder(nn.Module):
      def __init__(self, config: dict) -> None: ...
      def forward(
          self,
          enc_seq: torch.Tensor,            # FloatTensor [B, L_mel, H]
          mask: torch.Tensor | None = None, # BoolTensor  [B, L_mel] (True = PAD)
          return_attns: bool = False,
      ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]: ...
  ```
- **Key Differences**:
  1. Slices `mask = mask[:, :max_len]` *before* constructing `slf_attn_mask` in the `else` branch.
  2. Returns `(dec_output, mask)` when `return_attns=False` (default) and `(dec_output, mask, dec_slf_attn_list)` when `return_attns=True`.
  3. Handles `mask=None` cleanly.

---

### 2.6 `VariancePredictor`
- **Original (`original_fastspeech2/model/modules.py:197–250`)**:
  - `__init__(self, model_config)`: Hardcodes `padding=1` in `conv1d_2` (line 230) instead of `(self.kernel - 1) // 2`.
  - `forward(self, encoder_output, mask) -> Tensor`
- **Proposed (`common/model/modules.py`)**:
  ```python
  class VariancePredictor(nn.Module):
      def __init__(self, model_config: dict) -> None: ...
      def forward(
          self,
          encoder_output: torch.Tensor,     # FloatTensor [B, L, H]
          mask: torch.Tensor | None = None, # BoolTensor  [B, L] (True = PAD)
      ) -> torch.Tensor: ...                # FloatTensor [B, L]
  ```
- **Key Differences**:
  1. Uses `padding=(self.kernel - 1) // 2` in **both** `conv1d_1` and `conv1d_2` so non-3 odd kernel sizes work without length mismatch.
  2. Preserves exact `OrderedDict` layer names (`conv1d_1`, `relu_1`, `layer_norm_1`, `dropout_1`, `conv1d_2`, `relu_2`, `layer_norm_2`, `dropout_2`, and `linear_layer`) for checkpoint compatibility.

---

### 2.7 `LengthRegulator`
- **Original (`original_fastspeech2/model/modules.py:161–194`)**:
  - `__init__(self)`
  - `LR(self, x, duration, max_len=None) -> tuple[Tensor, Tensor]`: Allocates `torch.LongTensor(mel_len).to(device)` using module-global `device`, crashing on MPS.
  - `expand(self, batch, predicted) -> Tensor`
  - `forward(self, x, duration, max_len) -> tuple[Tensor, Tensor]`
- **Proposed (`common/model/modules.py`)**:
  ```python
  class LengthRegulator(nn.Module):
      def __init__(self) -> None: ...
      def LR(
          self,
          x: torch.Tensor,              # FloatTensor [B, L_src, H]
          duration: torch.Tensor,       # LongTensor or FloatTensor [B, L_src]
          max_len: int | None = None,
      ) -> tuple[torch.Tensor, torch.Tensor]: ...
      def expand(
          self,
          batch: torch.Tensor,          # FloatTensor [L_src, H]
          predicted: torch.Tensor,      # 1D Tensor   [L_src]
      ) -> torch.Tensor: ...            # FloatTensor [T_b, H]
      def forward(
          self,
          x: torch.Tensor,
          duration: torch.Tensor,
          max_len: int | None = None,
      ) -> tuple[torch.Tensor, torch.Tensor]: ... # ([B, L_mel, H], LongTensor[B] on x.device)
  ```
- **Key Differences**:
  1. Allocates `mel_len` directly on `x.device`: `torch.tensor(mel_len, dtype=torch.long, device=x.device)`.
  2. Defaults `max_len: int | None = None` in `forward()`.
  3. Safety guard when all predicted durations in an utterance are `0` (`sum(duration[b]) == 0` and `max_len is None`): clamps minimum expanded utterance length to `1` zero-frame (or raises a descriptive error if configured) so downstream Conv1d/PostNet layers do not crash on a `0`-length time axis.

---

### 2.8 `VarianceAdaptor`
- **Original (`original_fastspeech2/model/modules.py:17–158`)**:
  - `__init__(self, preprocess_config, model_config)`: Always reads `<preprocessed_path>/stats.json` from disk; `np.log(pitch_min)` produces `NaN`/`-inf` if `pitch_min <= 0` in `"log"` mode.
  - `forward(self, x, src_mask, mel_mask=None, max_len=None, pitch_target=None, energy_target=None, duration_target=None, p_control=1.0, e_control=1.0, d_control=1.0)`: Passes `p_control` instead of `e_control` at lines 124 and 146; multiplies `d_control` *after* `torch.round()` at line 133.
- **Proposed (`common/model/modules.py`)**:
  ```python
  class VarianceAdaptor(nn.Module):
      def __init__(
          self,
          preprocess_config: dict,
          model_config: dict,
          stats: dict | None = None,
      ) -> None: ...

      def get_pitch_embedding(
          self,
          x: torch.Tensor,
          target: torch.Tensor | None,
          mask: torch.Tensor,
          control: float = 1.0,
      ) -> tuple[torch.Tensor, torch.Tensor]: ...

      def get_energy_embedding(
          self,
          x: torch.Tensor,
          target: torch.Tensor | None,
          mask: torch.Tensor,
          control: float = 1.0,
      ) -> tuple[torch.Tensor, torch.Tensor]: ...

      def forward(
          self,
          x: torch.Tensor,                            # FloatTensor [B, L_src, H]
          src_mask: torch.Tensor,                     # BoolTensor  [B, L_src]
          mel_mask: torch.Tensor | None = None,       # BoolTensor  [B, L_mel]
          max_len: int | None = None,
          pitch_target: torch.Tensor | None = None,
          energy_target: torch.Tensor | None = None,
          duration_target: torch.Tensor | None = None,
          p_control: float = 1.0,
          e_control: float = 1.0,
          d_control: float = 1.0,
      ) -> tuple[
          torch.Tensor, # x:                       FloatTensor [B, L_mel, H]
          torch.Tensor, # pitch_prediction:        FloatTensor [B, L_src] or [B, L_mel]
          torch.Tensor, # energy_prediction:       FloatTensor [B, L_src] or [B, L_mel]
          torch.Tensor, # log_duration_prediction: FloatTensor [B, L_src]
          torch.Tensor, # duration_rounded:        LongTensor  [B, L_src]
          torch.Tensor, # mel_len:                 LongTensor  [B]
          torch.Tensor, # mel_mask:                BoolTensor  [B, L_mel]
      ]: ...
  ```
- **Key Differences**:
  1. Fixes energy control bug at both `phoneme_level` and `frame_level` branches by passing `e_control` (not `p_control`) to `self.get_energy_embedding(...)`.
  2. Multiplies `d_control` *before* `torch.round`: `torch.clamp(torch.round((torch.exp(log_duration_prediction) - 1) * d_control), min=0).long()`, avoiding floor-truncation of 1-frame phonemes when `d_control < 1.0` and ensuring `duration_rounded` is consistently `torch.int64` in both training and inference.
  3. Validates `pitch_min > 0` (and `energy_min > 0`) if `"log"` quantization is selected, raising a clear `ValueError` instead of silently populating `pitch_bins` with `NaN`.
  4. Allows `stats` dictionary (`{"pitch": [min, max, ...], "energy": [min, max, ...]}`) to be passed directly in `stats` or `model_config["stats"]` as an alternative to reading `<preprocessed_path>/stats.json` from disk (falling back to `<preprocessed_path>/stats.json` when `stats is None`), enabling pure in-memory unit testing without temporary files.

---

### 2.9 `PostNet`
- **Original (`original_fastspeech2/transformer/Layers.py:67–137`)**:
  - `__init__(self, n_mel_channels=80, postnet_embedding_dim=512, postnet_kernel_size=5, postnet_n_convolutions=5)`
  - `forward(self, x) -> Tensor`
- **Proposed (`common/transformer/layers.py`)**:
  ```python
  class PostNet(nn.Module):
      def __init__(
          self,
          n_mel_channels: int = 80,
          postnet_embedding_dim: int = 512,
          postnet_kernel_size: int = 5,
          postnet_n_convolutions: int = 5,
          postnet_dropout: float = 0.5,
      ) -> None: ...

      def forward(
          self,
          x: torch.Tensor,                  # FloatTensor [B, L_mel_dec, C]
          mask: torch.Tensor | None = None, # Optional BoolTensor [B, L_mel_dec] (True = PAD)
      ) -> torch.Tensor: ...                # FloatTensor [B, L_mel_dec, C]
  ```
- **Key Differences**:
  1. `FastSpeech2.__init__` explicitly passes `n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"]` (and optional `model_config.get("postnet", {})` parameters) to `PostNet(...)`.
  2. Accepts optional `mask: torch.Tensor | None = None` in `forward` (defaulting to `None` to preserve exact original behavior, or zeroing padded positions when provided).

---

### 2.10 `FastSpeech2`
- **Original (`original_fastspeech2/model/fastspeech2.py:13–110`)**:
  - `__init__(self, preprocess_config, model_config)`: Calls `PostNet()` with hardcoded 80 channels; reads `speakers.json` from disk when `multi_speaker: True`.
  - `forward(self, speakers, texts, src_lens, max_src_len, mels=None, mel_lens=None, max_mel_len=None, p_targets=None, e_targets=None, d_targets=None, p_control=1.0, e_control=1.0, d_control=1.0)`
- **Proposed (`common/model/fastspeech2.py`)**:
  ```python
  class FastSpeech2(nn.Module):
      def __init__(
          self,
          preprocess_config: dict,
          model_config: dict,
          n_src_vocab: int | None = None,
          n_speaker: int | None = None,
          stats: dict | None = None,
      ) -> None: ...

      def forward(
          self,
          speakers: torch.Tensor,               # LongTensor  [B]
          texts: torch.Tensor,                  # LongTensor  [B, L_src]
          src_lens: torch.Tensor,               # LongTensor  [B]
          max_src_len: int | None = None,
          mels: torch.Tensor | None = None,     # FloatTensor [B, L_mel, C] (unused in forward)
          mel_lens: torch.Tensor | None = None, # LongTensor  [B]
          max_mel_len: int | None = None,
          p_targets: torch.Tensor | None = None,
          e_targets: torch.Tensor | None = None,
          d_targets: torch.Tensor | None = None,
          p_control: float = 1.0,
          e_control: float = 1.0,
          d_control: float = 1.0,
      ) -> tuple[
          torch.Tensor, # output (mel_predictions):         [B, L_mel_dec, C]
          torch.Tensor, # postnet_output:                   [B, L_mel_dec, C]
          torch.Tensor, # p_predictions:                    [B, L_src] or [B, L_mel]
          torch.Tensor, # e_predictions:                    [B, L_src] or [B, L_mel]
          torch.Tensor, # log_d_predictions:                [B, L_src]
          torch.Tensor, # d_rounded:                        [B, L_src]
          torch.Tensor, # src_masks:                        [B, L_src]
          torch.Tensor, # mel_masks:                        [B, L_mel_dec]
          torch.Tensor, # src_lens:                         [B]
          torch.Tensor, # mel_lens:                         [B]
      ]: ...
  ```
- **Key Differences**:
  1. Passes `n_mel_channels` from `preprocess_config` into `PostNet`.
  2. Supports explicit `n_src_vocab`, `n_speaker`, and `stats` via constructor arguments or config dicts (with automatic fallback to reading `speakers.json` and `stats.json` from `preprocess_config["path"]["preprocessed_path"]`), allowing both disk-backed training and pure in-memory instantiation in unit tests.
  3. Defaults `max_src_len = int(src_lens.max().item())` if `max_src_len is None`.
  4. Validates non-empty input (`texts.shape[1] > 0` and `(src_lens > 0).all()`).

---

### 2.11 `FastSpeech2Loss`
- **Original (`original_fastspeech2/model/loss.py:5–92`)**:
  - `__init__(self, preprocess_config, model_config)`
  - `forward(self, inputs, predictions) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]`: Does not slice `pitch_predictions`, `pitch_targets`, `energy_predictions`, or `energy_targets` to `:mel_masks.shape[1]` in `"frame_level"` mode when `L_mel > max_seq_len`.
- **Proposed (`common/model/loss.py`)**:
  ```python
  class FastSpeech2Loss(nn.Module):
      def __init__(self, preprocess_config: dict, model_config: dict) -> None: ...
      def forward(
          self,
          inputs: tuple | list,
          predictions: tuple | list,
      ) -> tuple[
          torch.Tensor, # total_loss:       Scalar FloatTensor []
          torch.Tensor, # mel_loss:         Scalar FloatTensor []
          torch.Tensor, # postnet_mel_loss: Scalar FloatTensor []
          torch.Tensor, # pitch_loss:       Scalar FloatTensor []
          torch.Tensor, # energy_loss:      Scalar FloatTensor []
          torch.Tensor, # duration_loss:    Scalar FloatTensor []
      ]: ...
  ```
- **Key Differences**:
  1. Slices `pitch_predictions`, `pitch_targets` (when `pitch_feature_level == "frame_level"`) and `energy_predictions`, `energy_targets` (when `energy_feature_level == "frame_level"`) along dimension 1 to `:mel_masks.shape[1]` before calling `.masked_select(mel_masks)`, fixing the `RuntimeError` when `L_mel > max_seq_len`.
  2. Validates `pitch_feature_level` and `energy_feature_level` in `("phoneme_level", "frame_level")` during `__init__` and uses `if ... elif ...` for both pitch and energy.
  3. Checks for empty masked selections (`src_masks.any()` and `mel_masks.any()`), raising a descriptive `ValueError` if an entire batch is masked out rather than silently returning `NaN`.

---

## PART 3 — CONFIGURATION SPECIFICATION

All architectural settings are driven by two YAML configuration dictionaries (`preprocess_config` and `model_config`) so that English (Svarah) and Telugu (IndicTTS) share 100% of the Python code in `common/` while maintaining independent language/dataset configurations in `English/config/` and `Telugu/config/`.

| Setting Category | Config Dict & Key Path | Type | Recommended Default | Notes / Language Considerations |
| :--- | :--- | :--- | :--- | :--- |
| **Source Vocabulary Size** | `model_config["n_src_vocab"]` *(or derived from language symbol table + 1 for `PAD=0`)* | `int` | Language-dependent (e.g., `361` in original repo; separate counts for English ARPAbet vs. Telugu phoneme set) | **Never assume English and Telugu have the same vocabulary size.** Index `0` is always `<PAD>`. |
| **Max Sequence Length** | `model_config["max_seq_len"]` | `int` | `1000` | Preallocated sinusoidal table length (`1001`); dynamic expansion used during inference when $L > 1000$. |
| **Encoder Layers** | `model_config["transformer"]["encoder_layer"]` | `int` | `4` | Number of `FFTBlock` layers in `Encoder`. |
| **Encoder Attention Heads** | `model_config["transformer"]["encoder_head"]` | `int` | `2` | Must divide `encoder_hidden` evenly ($d_k = d_v = 128$). |
| **Encoder Hidden Dim** | `model_config["transformer"]["encoder_hidden"]` | `int` | `256` | Token embedding and `Encoder` hidden dimension $H$. |
| **Encoder Dropout** | `model_config["transformer"]["encoder_dropout"]` | `float` | `0.2` | Applied in `Encoder` MHA and Conv1D FFN. |
| **Decoder Layers** | `model_config["transformer"]["decoder_layer"]` | `int` | `6` | Number of `FFTBlock` layers in `Decoder` (`6` in standard configs, `4` in `LJSpeech_paper`). |
| **Decoder Attention Heads** | `model_config["transformer"]["decoder_head"]` | `int` | `2` | Must divide `decoder_hidden` evenly ($d_k = d_v = 128$). |
| **Decoder Hidden Dim** | `model_config["transformer"]["decoder_hidden"]` | `int` | `256` | Must equal `encoder_hidden` ($H = 256$). |
| **Decoder Dropout** | `model_config["transformer"]["decoder_dropout"]` | `float` | `0.2` | Applied in `Decoder` MHA and Conv1D FFN. |
| **FFN Inner Filter Size** | `model_config["transformer"]["conv_filter_size"]` | `int` | `1024` | Inner channel size $d_{\text{hid}}$ of `PositionwiseFeedForward`. |
| **FFN Conv Kernel Sizes** | `model_config["transformer"]["conv_kernel_size"]` | `list[int]` | `[9, 1]` | `[k1, k2]` for `w_1` and `w_2` in `PositionwiseFeedForward` (both must be odd). |
| **Variance Predictor Filter Size** | `model_config["variance_predictor"]["filter_size"]` | `int` | `256` | Hidden channel size of `VariancePredictor` Conv1D layers. |
| **Variance Predictor Kernel Size** | `model_config["variance_predictor"]["kernel_size"]` | `int` | `3` | Kernel size for both `conv1d_1` and `conv1d_2` (must be odd). |
| **Variance Predictor Dropout** | `model_config["variance_predictor"]["dropout"]` | `float` | `0.5` | Dropout probability in `VariancePredictor`. |
| **Pitch Quantization Mode** | `model_config["variance_embedding"]["pitch_quantization"]` | `str` | `"linear"` | `"linear"` (required when `pitch.normalization: True`) or `"log"` (only valid if `pitch_min > 0`). |
| **Energy Quantization Mode** | `model_config["variance_embedding"]["energy_quantization"]` | `str` | `"linear"` | `"linear"` (required when `energy.normalization: True`) or `"log"` (only valid if `energy_min > 0`). |
| **Quantization Bins** | `model_config["variance_embedding"]["n_bins"]` | `int` | `256` | Number of embedding buckets (`n_bins - 1 = 255` bin thresholds). |
| **PostNet Embedding Dim** | `model_config.get("postnet", {}).get("postnet_embedding_dim", 512)` | `int` | `512` | Optional override; defaults to `512`. |
| **PostNet Kernel Size** | `model_config.get("postnet", {}).get("postnet_kernel_size", 5)` | `int` | `5` | Optional override; defaults to `5` (must be odd). |
| **PostNet Conv Layers** | `model_config.get("postnet", {}).get("postnet_n_convolutions", 5)` | `int` | `5` | Optional override; defaults to `5`. |
| **PostNet Dropout** | `model_config.get("postnet", {}).get("postnet_dropout", 0.5)` | `float` | `0.5` | Optional override; defaults to `0.5`. |
| **Multi-Speaker Mode** | `model_config["multi_speaker"]` | `bool` | `False` (single-speaker) / `True` (multi-speaker, e.g., Svarah / IndicTTS if multiple speakers exist) | Enables `self.speaker_emb = nn.Embedding(n_speaker, encoder_hidden)`. |
| **Speaker Count** | `model_config.get("n_speaker")` *(or length of `speakers.json`)* | `int` | Derived from `speakers.json` | Used when `multi_speaker: True`. |
| **Mel Channels** | `preprocess_config["preprocessing"]["mel"]["n_mel_channels"]` | `int` | `80` | Passed to **both** `self.mel_linear` and `self.postnet` (must match vocoder `num_mels`, which is `80` for HiFi-GAN). |
| **Pitch Feature Level** | `preprocess_config["preprocessing"]["pitch"]["feature"]` | `str` | `"phoneme_level"` | `"phoneme_level"` (recommended default) or `"frame_level"`. |
| **Energy Feature Level** | `preprocess_config["preprocessing"]["energy"]["feature"]` | `str` | `"phoneme_level"` | `"phoneme_level"` (recommended default) or `"frame_level"`. |

---

## PART 4 — CONSOLIDATED BUG AND IMPROVEMENT REGISTER

The table below consolidates all 15 confirmed bugs, latent configuration defects, and architectural coupling issues identified across Steps 1.3A–1.3F. No findings have been invented.

| ID | Issue Name | Source Location(s) | Status | Impact | Proposed Correction in `common/` | Test Needed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B-01** | **Global Device Mismatch on MPS / Non-Default Devices** | [`model/modules.py:14, 180`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py#L14) (`LengthRegulator.LR`), [`utils/tools.py:15, 96`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/utils/tools.py#L15) (`get_mask_from_lengths`) | **Confirmed Bug** (Empirically verified crash on `mps:0`) | Module-global `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` evaluates to `"cpu"` on Apple Silicon M2. When model & inputs are on `"mps:0"`, `ids` and `mel_len` are created on `"cpu"`, raising `RuntimeError: Expected all tensors to be on the same device, mps:0 and cpu`. | Remove module-global `device`. Allocate `ids` on `lengths.device` in `get_mask_from_lengths` and `mel_len` on `x.device` in `LengthRegulator.LR`. | Unit & integration test running `get_mask_from_lengths`, `LengthRegulator`, and full `FastSpeech2` forward/backward on both `cpu` and `mps:0`. |
| **B-02** | **Energy Control Incorrectly Uses Pitch Control (`p_control`)** | [`model/modules.py:124, 146`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py#L124) (`VarianceAdaptor.forward`) | **Confirmed Bug** | Both `phoneme_level` (line 124) and `frame_level` (line 146) energy blocks pass `p_control` instead of `e_control` to `self.get_energy_embedding(...)`. `e_control` is completely unused; `--energy_control` has no effect at inference, while `--pitch_control` scales both pitch and energy. | Pass `e_control` to `self.get_energy_embedding(...)` at both lines 124 and 146. | Unit test verifying that varying `e_control` (with `p_control=1.0`) changes `e_predictions` and `postnet_output` while leaving `p_predictions` unchanged, and vice versa. |
| **B-03** | **Log Quantization with Non-Positive Minimum Values Produces `NaN` Bins** | [`model/modules.py:51, 63`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py#L51) (`VarianceAdaptor.__init__`) | **Confirmed Bug (Latent Config)** | `np.log(pitch_min)` and `np.log(energy_min)` evaluate `log` of negative numbers when `normalization: True` (`pitch_min ≈ -2.9`) or `log(0.0) = -inf` when `frame_level` unvoiced pitch has `pitch_min == 0.0`, silently filling `pitch_bins` / `energy_bins` with `NaN`. | Check `if pitch_min <= 0: raise ValueError(...)` when `pitch_quantization == "log"` (and likewise for `energy_min` when `energy_quantization == "log"`). Default to `"linear"`. | Unit test asserting `ValueError` is raised when `"log"` quantization is requested with `min <= 0`, and verifying finite bins for `"linear"`. |
| **B-04** | **Duration Scaling Order & Integer Floor-Truncation** | [`model/modules.py:132–135, 187`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py#L132-L135) (`VarianceAdaptor.forward`, `LengthRegulator.expand`) | **Confirmed Defect / Potential Issue** | `torch.round(torch.exp(log_d) - 1) * d_control` multiplies `d_control` *after* rounding, returning float durations that `LengthRegulator.expand` truncates via `int()`. When `d_control = 0.8`, every 1-frame phoneme becomes `int(0.8) = 0` (deleted). Also causes `d_rounded` dtype to be `float32` in inference vs. `int64` in training. | Multiply `d_control` *inside* `torch.round`: `torch.clamp(torch.round((torch.exp(log_d) - 1) * d_control), min=0).long()`. (Identical when `d_control == 1.0`). | Unit test with `log_d = log(1 + 1)` and `d_control = 0.8` verifying `round(1.0 * 0.8) = 1` frame (not `0`) and `d_rounded.dtype == torch.int64`. |
| **B-05** | **Hardcoded PostNet 80 Mel Channels in `FastSpeech2.__init__`** | [`model/fastspeech2.py:27`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/fastspeech2.py#L27) (`FastSpeech2.__init__`) | **Confirmed Bug** | `self.mel_linear` reads `n_mel_channels` from `preprocess_config`, but `self.postnet = PostNet()` omits `n_mel_channels`, hardcoding `80`. Any `n_mel_channels != 80` crashes in `PostNet.convolutions[0]`. | Pass `n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"]` when instantiating `PostNet`. | Unit test instantiating and running `FastSpeech2` with `n_mel_channels=64` and `n_mel_channels=80`. |
| **B-06** | **Attention All-Masked Row Produces `NaN` in Softmax** | [`transformer/Modules.py:20–22`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Modules.py#L20-L22) (`ScaledDotProductAttention.forward`) | **Confirmed Edge-Case Bug** (Empirically verified on `cpu` & `mps:0`) | If an utterance has all key positions masked (`mask[b]` all `True`), `masked_fill(mask, -np.inf)` makes the entire row `-inf`, and `Softmax(dim=2)` computes `0/0 = NaN` across `attn`. | Reject empty/zero-length sequences at model entry (`(src_lens > 0).all()`) and apply `attn = attn.masked_fill(all_masked_rows, 0.0)` / `torch.nan_to_num(attn, nan=0.0)` after softmax in `ScaledDotProductAttention`. | Unit test passing a batch containing an all-masked row to `MultiHeadAttention` and verifying `not torch.isnan(attn).any()`. |
| **B-07** | **Missing `mask=None` Handling in `MultiHeadAttention` and `FFTBlock`** | [`transformer/SubLayers.py:29, 46`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/SubLayers.py#L46), [`transformer/Layers.py:21, 25, 28`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Layers.py#L25) | **Confirmed Bug** | Both methods specify `mask=None` as the default argument, but unconditionally call `mask.repeat(...)` and `mask.unsqueeze(-1)`, raising `AttributeError` when called without a mask. | Add `if mask is not None:` guards in `MultiHeadAttention.forward` (line 46) and `FFTBlock.forward` (lines 25, 28). | Unit test calling `MultiHeadAttention(q, k, v)` and `FFTBlock(x)` with `mask=None`. |
| **B-08** | **Broken `return_attns=True` Return Behavior in `Encoder` and `Decoder`** | [`transformer/Models.py:73, 97–100, 139, 168–171`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Models.py#L97-L100) | **Confirmed Bug** | Both `Encoder.forward` and `Decoder.forward` collect attention tensors into a list when `return_attns=True`, but never return the list at lines 100 and 171. | Return `(enc_output, enc_slf_attn_list)` from `Encoder` and `(dec_output, mask, dec_slf_attn_list)` from `Decoder` when `return_attns=True` (keeping single/2-tuple return when `return_attns=False`). | Unit test calling `Encoder(..., return_attns=True)` and `Decoder(..., return_attns=True)` and verifying returned attention list lengths and shapes. |
| **B-09** | **Frame-Level Loss Truncation Shape Mismatch When `L_mel > max_seq_len`** | [`model/loss.py:43–64`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/loss.py#L43-L64), [`transformer/Models.py:154–162`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Models.py#L154-L162) | **Confirmed Bug** (Empirically verified `RuntimeError`) | When `L_mel > max_seq_len` (`1000`) during training, `Decoder` truncates `mel_masks` to `1000` frames and `loss.py` slices `mel_targets` to `1000`, but `pitch_predictions`, `pitch_targets`, `energy_predictions`, and `energy_targets` (from `VarianceAdaptor`) still have length `L_mel > 1000`, crashing `masked_select(mel_masks)` in `"frame_level"` mode. | In `FastSpeech2Loss.forward`, when `feature == "frame_level"`, slice `pitch_predictions`, `pitch_targets`, `energy_predictions`, and `energy_targets` to `[:, :mel_masks.shape[1]]`. | Unit test running `FastSpeech2Loss` in `"frame_level"` mode with `L_mel = 1050` and `mel_masks` truncated to `1000`. |
| **B-10** | **Empty Masked Loss Selections Produce `NaN` Loss** | [`model/loss.py:51–79`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/loss.py#L51-L79) (`FastSpeech2Loss.forward`) | **Confirmed Edge-Case Bug** (Empirically verified) | If all positions in `src_masks` or `mel_masks` are `True` (padded), `~mask` is all `False`, `masked_select` returns a `0`-element tensor, and `nn.L1Loss()`/`nn.MSELoss()` returns `NaN`. | Validate `src_masks.any()` and `mel_masks.any()` after inversion in `FastSpeech2Loss.forward`, raising a clear `ValueError` on empty valid selections. | Unit test verifying that all-masked masks raise `ValueError` in `FastSpeech2Loss`. |
| **B-11** | **PostNet Padding Contamination (`BatchNorm1d` & 10-Frame Boundary Bleed)** | [`transformer/Layers.py:129–137`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Layers.py#L129-L137), [`model/fastspeech2.py:95–97`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/fastspeech2.py#L95-L97) | **Confirmed Architectural Behavior / Potential Issue** | `mel_linear` adds bias to padded frames, and `PostNet` runs 5 layers of `Conv1d(k=5) + BatchNorm1d` without zero-masking padded frames. Padded tail frames are excluded by `mel_masks` in `FastSpeech2Loss` and sliced off in `synth_samples`, but can slightly affect `BatchNorm1d` stats and the last 10 frames of shorter batch items when $B > 1$. | Support optional `mask: Tensor | None = None` in `PostNet.forward(x, mask=None)`; keep default unmasked behavior unless enabled via config so original training dynamics are preserved. | Unit test checking `PostNet` output shape and optional mask zeroing at padded positions. |
| **B-12** | **Global Vocabulary Coupling to `text/symbols.py`** | [`transformer/Models.py:7, 40`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Models.py#L7) (`Encoder.__init__`) | **Confirmed Architectural Coupling** | `Encoder` imports `from text.symbols import symbols` globally and sets `n_src_vocab = len(symbols) + 1`. English and Telugu have distinct phoneme inventories and cannot be cleanly decoupled if `Encoder` hardcodes a global import. | Pass `n_src_vocab` explicitly via `Encoder.__init__(config, n_src_vocab=...)` / `model_config["n_src_vocab"]`. | Unit test instantiating two `FastSpeech2` models in the same process with different vocabulary sizes (e.g., `n_src_vocab=80` for English and `n_src_vocab=65` for Telugu). |
| **B-13** | **Hardcoded `padding=1` in `VariancePredictor.conv1d_2`** | [`model/modules.py:230`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py#L230) (`VariancePredictor.__init__`) | **Confirmed Bug (Latent Config)** | `conv1d_1` uses `padding=(self.kernel - 1) // 2`, but `conv1d_2` hardcodes `padding=1`, breaking sequence length preservation if `variance_predictor.kernel_size != 3`. | Replace `padding=1` at line 230 with `padding=(self.kernel - 1) // 2`. | Unit test instantiating `VariancePredictor` with `kernel_size=5` and verifying output length equals input length $L$. |
| **B-14** | **Unused `w_init_gain` / `w_init` Parameters in `ConvNorm` and `Conv`** | [`transformer/Layers.py:43`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/transformer/Layers.py#L43) (`ConvNorm.__init__`), [`model/modules.py:267`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py#L267) (`Conv.__init__`) | **Confirmed Dead Parameter** | Both classes accept `w_init_gain` / `w_init` in `__init__` but never call `nn.init.xavier_uniform_`, leaving PyTorch's default Kaiming Uniform initialization active. | Keep parameter signature for compatibility and document that default `nn.Conv1d` Kaiming Uniform init is intentionally preserved (as `BatchNorm1d`/`LayerNorm` immediately follows each conv). | Code inspection / unit test verifying `ConvNorm` and `Conv` initialization succeeds. |
| **B-15** | **Mandatory Disk I/O (`stats.json`, `speakers.json`) During Model `__init__`** | [`model/modules.py:41–46`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/modules.py#L41-L46), [`model/fastspeech2.py:31–37`](file:///Users/swaruppal/FastSpeech2_Project/original_fastspeech2/model/fastspeech2.py#L31-L37) | **Confirmed Coupling Issue** | `FastSpeech2` and `VarianceAdaptor` cannot be instantiated without `stats.json` (and `speakers.json` if `multi_speaker: True`) existing on disk at `preprocessed_path`. | Allow `stats` dict and `n_speaker` int to be passed directly in constructor args or `model_config`, falling back to reading `stats.json` / `speakers.json` from `preprocessed_path` when omitted. | Unit test instantiating `FastSpeech2` purely in-memory with no files on disk. |

---

## PART 5 — ORIGINAL BEHAVIOR VS. NEW IMPLEMENTATION

### 5.1 Categorized Decision Matrix

| Category | Architectural / Behavioral Item | Decision Rationale |
| :--- | :--- | :--- |
| **1. Preserve Unchanged** | **Core Module Hierarchy & `state_dict` Parameter Names** (`encoder`, `variance_adaptor`, `decoder`, `mel_linear`, `postnet`, `speaker_emb`, and all internal sub-layer names) | Preserves the exact ~27M parameter FastSpeech 2 architecture from `ming024/FastSpeech2` and ensures 100% `state_dict` compatibility with reference tools and checkpoints. |
| **1. Preserve Unchanged** | **Post-LN Residual Structure & Interleaved `masked_fill` in `FFTBlock`** | Zeroing padded positions between `MultiHeadAttention` and `PositionwiseFeedForward` (`Conv1d` with `k=9`) and again after `PositionwiseFeedForward` is essential to prevent padding bleed into valid tokens. |
| **1. Preserve Unchanged** | **Sequential Pitch-then-Energy Conditioning (`x = x + pitch_embedding` before `energy_predictor`)** | In `VarianceAdaptor.forward`, `energy_predictor` receives `x + pitch_embedding`. Preserving this exact data flow maintains fidelity to the reference implementation. |
| **1. Preserve Unchanged** | **5-Term Unweighted Loss Formula & Log-Duration Target `torch.log(d.float() + 1)`** | Supervision of `mel_loss` (L1), `postnet_mel_loss` (L1), `pitch_loss` (MSE), `energy_loss` (MSE), and `duration_loss` (MSE on $\ln(d+1)$) with `masked_select` token/frame averaging is mathematically sound and well-tested. |
| **1. Preserve Unchanged** | **Default `nn.Conv1d` Kaiming Uniform Initialization in `ConvNorm` and `Conv`** | Every `ConvNorm` in `PostNet` is followed by `BatchNorm1d` and every `Conv` in `VariancePredictor` is followed by `LayerNorm`, absorbing linear weight scale differences. Preserving default init matches the original repo's training dynamics. |
| **2. Fix for Correctness** | **Dynamic Device Derivation (`lengths.device`, `x.device`) Instead of Global `device` (B-01)** | Mandatory for Apple Silicon `mps:0` compatibility and multi-device correctness. Zero impact on mathematical outputs. |
| **2. Fix for Correctness** | **Pass `e_control` Instead of `p_control` to `get_energy_embedding` (B-02)** | Fixes broken inference energy control and decouples pitch scaling from energy scaling when `p_control != 1.0` or `e_control != 1.0`. |
| **2. Fix for Correctness** | **Multiply `d_control` Before `torch.round()` and Cast `d_rounded` to `torch.long` (B-04)** | Prevents `int()` floor truncation from deleting 1-frame phonemes when `d_control < 1.0` and makes `d_rounded` dtype consistent (`int64`) across training and inference. |
| **2. Fix for Correctness** | **Validate Positive Min/Max for `"log"` Quantization (B-03)** | Prevents silent `NaN` corruption of `pitch_bins` and `energy_bins` when `min <= 0`. |
| **2. Fix for Correctness** | **Guard `if mask is not None:` in `MultiHeadAttention` & `FFTBlock` and Guard All-Masked Softmax Rows (B-06, B-07)** | Prevents `AttributeError` when `mask=None` and prevents `0/0 = NaN` in `ScaledDotProductAttention` when an entire row is masked. |
| **2. Fix for Correctness** | **Return Attention Lists When `return_attns=True` in `Encoder` and `Decoder` (B-08)** | Fixes dead code in `Encoder.forward` and `Decoder.forward` while keeping the `return_attns=False` return signature unchanged. |
| **2. Fix for Correctness** | **Slice `frame_level` Pitch/Energy Predictions and Targets to `:mel_masks.shape[1]` in `FastSpeech2Loss` (B-09)** | Fixes the `RuntimeError` shape mismatch when `feature == "frame_level"` and `L_mel > max_seq_len`. |
| **2. Fix for Correctness** | **Use `padding=(self.kernel - 1) // 2` in `VariancePredictor.conv1d_2` (B-13)** | Fixes length mismatch when `variance_predictor.kernel_size != 3`. |
| **2. Fix for Correctness** | **Reject Empty Sequences (`L_src == 0` or `src_lens <= 0`) and Empty Masked Loss Selections (B-10)** | Fails fast with a clear `ValueError` instead of propagating silent `NaN` tensors through the model or loss. |
| **3. Make Configurable** | **Language-Specific `n_src_vocab` in `Encoder` & `FastSpeech2` (B-12)** | Essential so English (Svarah) and Telugu (IndicTTS) models can each define their own phoneme inventory size without mutating a shared global `text/symbols.py`. |
| **3. Make Configurable** | **Pass `n_mel_channels` (and optional `postnet` config) to `PostNet` in `FastSpeech2.__init__` (B-05)** | Eliminates the hardcoded 80-channel discrepancy between `mel_linear` and `postnet`. |
| **3. Make Configurable** | **Allow In-Memory `stats` Dict and `n_speaker` in `FastSpeech2` / `VarianceAdaptor` `__init__` (B-15)** | Enables fast, self-contained unit tests without requiring `stats.json` or `speakers.json` files on disk, while keeping disk loading as the default fallback. |
| **4. Defer Pending Evidence** | **Masking Padded Frames Inside `PostNet` Between `ConvNorm` + `BatchNorm1d` Layers (B-11)** | While zeroing padded frames inside `PostNet` would eliminate the 10-frame boundary bleed in multi-utterance batches, it alters `BatchNorm1d` batch statistics compared to `original_fastspeech2`. We will support an optional `mask=None` parameter in `PostNet.forward` and defer enabling internal `PostNet` masking by default unless empirical validation shows a benefit. |
| **4. Defer Pending Evidence** | **Best-Validation-Loss Checkpoint Selection vs. Step-Interval Checkpointing** | Original `train.py` saves checkpoints only at fixed `save_step` intervals. Adding best-validation-loss checkpointing belongs to the training script design stage rather than the core `common/model/` specification. |

---

## PART 6 — TEST PLAN

> **Note on Verification Status**: During Steps 1.3D–1.3F, read-only inspection checks were executed against `original_fastspeech2` modules on `cpu` and `mps:0`. The test cases below define the **planned unit and integration test suite** (`tests/`) to be implemented and executed once the `common/` modules are written in the upcoming implementation stage.

| Test ID | Test Category | Target Module(s) | Test Description & Assertions |
| :--- | :--- | :--- | :--- |
| **T-01** | **CPU Execution** | All `common/` modules | Instantiate and run forward + backward passes of `Encoder`, `Decoder`, `VarianceAdaptor`, `PostNet`, `FastSpeech2`, and `FastSpeech2Loss` on `torch.device("cpu")`. Verify all output tensors reside on `cpu` and all parameter gradients are finite. |
| **T-02** | **Apple Silicon MPS Execution** | All `common/` modules | Move `FastSpeech2` and `FastSpeech2Loss` to `torch.device("mps")` (`mps:0`). Run both training forward/backward (`d_targets` provided) and inference forward (`d_targets=None`). Verify zero `mps:0` vs. `cpu` device mismatch errors and verify all returned tensors reside on `mps:0`. |
| **T-03** | **Variable-Length Batches & Padding Invariance** | `Encoder`, `Decoder`, `FFTBlock`, `VariancePredictor` | Construct a batch with $B = 3$ and `src_lens = [8, 5, 3]`, `mel_lens = [30, 20, 12]`. Verify that padded positions ($t \ge \text{len}_b$) in `Encoder`, `Decoder`, and `VariancePredictor` outputs are strictly `0.0`. |
| **T-04** | **Single-Token & Single-Frame Sequences** | `Encoder`, `Decoder`, `VarianceAdaptor`, `PostNet` (`eval`) | Test $L_{\text{src}} = 1$ (`src_lens = [1]`) and $L_{\text{mel}} = 1$ (`mel_lens = [1]`) in `eval()` mode (and $B \ge 2, L_{\text{mel}} = 1$ in `train()` mode for `PostNet` `BatchNorm1d`). Verify shapes `[B, 1, H]` and `[B, 1, C]` with no `NaN`s. |
| **T-05** | **Empty Input Rejection** | `Encoder`, `FastSpeech2`, `FastSpeech2Loss` | Pass `texts` with `shape[1] == 0` or `src_lens` containing `0` to `FastSpeech2.forward`, and pass all-`True` (all-padded) masks to `FastSpeech2Loss`. Assert that a descriptive `ValueError` is raised immediately instead of returning `NaN`. |
| **T-06** | **Padded Position Masking & `mask=None` Handling** | `MultiHeadAttention`, `FFTBlock`, `ScaledDotProductAttention` | Call `MultiHeadAttention(q, k, v, mask=None)` and `FFTBlock(x, mask=None, slf_attn_mask=None)` and assert no `AttributeError`. Also pass a key mask with an all-masked row to `MultiHeadAttention` and assert `not torch.isnan(attn).any()`. |
| **T-07** | **Duration Expansion & Rounding with `d_control`** | `LengthRegulator`, `VarianceAdaptor` | Test `LengthRegulator` with known durations `[[2, 0, 3], [1, 4, 0]]` and verify output shape `[2, 5, H]`, `mel_lens == [5, 5]`, and exact vector repetition. Test `VarianceAdaptor` in inference mode with a 1-frame predicted token (`log_d = log(2)`) and `d_control = 0.8`, asserting `d_rounded == 1` (`int64`), not `0`. |
| **T-08** | **Independent Pitch and Energy Controls (`p_control` vs. `e_control`)** | `VarianceAdaptor`, `FastSpeech2` | In inference mode (`p_targets=None, e_targets=None`), run twice with `(p_control=1.0, e_control=1.0)` vs. `(p_control=1.0, e_control=1.5)`. Assert `p_predictions` is unchanged while `e_predictions` is scaled by `1.5x` in both `"phoneme_level"` and `"frame_level"` modes. Repeat with `p_control=1.5, e_control=1.0`. |
| **T-09** | **PostNet Output Shapes & Configurable Mel Channels** | `PostNet`, `FastSpeech2` | Instantiate `FastSpeech2` and `PostNet` with `n_mel_channels = 80` and `n_mel_channels = 64`. Verify `output` and `postnet_output` both have shape `[B, L_mel_dec, C]` for both `C = 80` and `C = 64`. |
| **T-10** | **Loss Calculations (`phoneme_level`, `frame_level`, & `L_mel > max_seq_len`)** | `FastSpeech2Loss` | Verify exact L1 and MSE loss values against manual unpadded calculation; verify zero-duration tokens (`d=0`) produce finite `duration_loss`; and test `"frame_level"` mode with `L_mel = 1050` and `mel_masks` truncated to `1000` frames, asserting no shape mismatch `RuntimeError`. |
| **T-11** | **Independent English & Telugu Vocabulary Sizes** | `Encoder`, `FastSpeech2` | Instantiate two separate `FastSpeech2` instances in the same Python process—one with `n_src_vocab = 150` (simulating English) and one with `n_src_vocab = 95` (simulating Telugu)—and verify `encoder.src_word_emb.weight.shape` is `[150, 256]` and `[95, 256]` respectively, with valid forward passes up to token ID `V - 1`. |
| **T-12** | **End-to-End Forward & Backward Pass (Single-Speaker & Multi-Speaker)** | `FastSpeech2`, `FastSpeech2Loss` | Run full training forward + `FastSpeech2Loss` + `total_loss.backward()` and full inference forward (`targets=None`) for both `multi_speaker=False` and `multi_speaker=True` (`n_speaker=10`) on `cpu` and `mps:0`. Verify all 10 output tuple elements and all 6 loss tuple elements match the specification. |

---

## PART 7 — FINAL ARCHITECTURE DIRECTORY PLAN

Below is the recommended file layout under `/Users/swaruppal/FastSpeech2_Project/common/` (using the subdirectories created in Step 1.2) and the exact responsibilities of each file:

```text
common/
├── __init__.py
├── transformer/
│   ├── __init__.py
│   ├── constants.py
│   ├── modules.py
│   ├── sublayers.py
│   ├── layers.py
│   └── models.py
├── model/
│   ├── __init__.py
│   ├── modules.py
│   ├── fastspeech2.py
│   └── loss.py
├── modules/
│   └── __init__.py
├── audio/
│   ├── __init__.py
│   ├── stft.py
│   └── audio_processing.py
└── utils/
    ├── __init__.py
    └── tools.py
```

### File-by-File Responsibilities

1. **`common/__init__.py`** *(already exists)*:
   - Top-level package marker exposing shared package metadata.
2. **`common/transformer/constants.py`**:
   - Defines special token index constants: `PAD = 0`, `UNK = 1`, `BOS = 2`, `EOS = 3`.
3. **`common/transformer/modules.py`**:
   - Defines `ScaledDotProductAttention(nn.Module)` with $\frac{1}{\sqrt{d_k}}$ scaling, `-np.inf` key masking, and safe handling of all-masked query rows.
4. **`common/transformer/sublayers.py`**:
   - Defines `MultiHeadAttention(nn.Module)` (with `if mask is not None:` guard) and `PositionwiseFeedForward(nn.Module)` (1D Conv FFN with odd kernel size validation).
5. **`common/transformer/layers.py`**:
   - Defines `FFTBlock(nn.Module)` (with `if mask is not None:` guards on post-attention and post-FFN zero-masking), `ConvNorm(nn.Module)`, and `PostNet(nn.Module)` (configurable `n_mel_channels`, `postnet_embedding_dim`, `postnet_kernel_size`, `postnet_n_convolutions`, `postnet_dropout`, and optional `mask`).
6. **`common/transformer/models.py`**:
   - Defines `get_sinusoid_encoding_table()`, `Encoder(nn.Module)` (with configurable `n_src_vocab` and working `return_attns`), and `Decoder(nn.Module)` (with clean `max_seq_len` truncation and working `return_attns`).
7. **`common/transformer/__init__.py`**:
   - Re-exports `Encoder`, `Decoder`, `PostNet`, `FFTBlock`, `ConvNorm`, `MultiHeadAttention`, `PositionwiseFeedForward`, `ScaledDotProductAttention`, and `constants`.
8. **`common/model/modules.py`**:
   - Defines `VarianceAdaptor(nn.Module)` (with fixed `e_control`, pre-round `d_control` scaling, positive log-bin validation, and optional in-memory `stats`), `LengthRegulator(nn.Module)` (allocating `mel_len` on `x.device`), `VariancePredictor(nn.Module)` (using `(self.kernel - 1) // 2` padding in both conv layers), and `Conv(nn.Module)`.
9. **`common/model/fastspeech2.py`**:
   - Defines `FastSpeech2(nn.Module)` integrating `Encoder`, optional `speaker_emb`, `VarianceAdaptor`, `Decoder`, `mel_linear`, and `PostNet(n_mel_channels=...)`, with support for both config-file/disk initialization and in-memory unit-test initialization.
10. **`common/model/loss.py`**:
    - Defines `FastSpeech2Loss(nn.Module)` computing the 5 masked loss components (`mel_loss`, `postnet_mel_loss`, `pitch_loss`, `energy_loss`, `duration_loss`) with proper `"frame_level"` target/prediction slicing to `:mel_masks.shape[1]` and empty-mask validation.
11. **`common/model/__init__.py`**:
    - Re-exports `FastSpeech2`, `FastSpeech2Loss`, `VarianceAdaptor`, `VariancePredictor`, and `LengthRegulator`.
12. **`common/modules/__init__.py`**:
    - Re-exports shared building blocks (`VarianceAdaptor`, `VariancePredictor`, `LengthRegulator`, `Conv`) so imports from either `common.model.modules` or `common.modules` resolve cleanly.
13. **`common/utils/tools.py` & `common/utils/__init__.py`**:
    - Defines device-agnostic tensor utilities required by the model and training pipeline: `get_mask_from_lengths(lengths, max_len=None)` (creating `ids` on `lengths.device`), `pad(input_ele, mel_max_length=None)`, `expand(values, durations)`, and `to_device(data, device)` (without importing `matplotlib` at module top-level so core model/loss imports stay lightweight).
14. **`common/audio/` (`__init__.py`, `stft.py`, `audio_processing.py`)**:
    - Reserved for the upcoming audio/STFT preprocessing stage (converting the hardcoded `.cuda()` calls in `original_fastspeech2/audio/stft.py` to device-agnostic PyTorch/torchaudio operations).

---

## PART 8 — OPEN QUESTIONS

The following items cannot be determined solely from inspecting `original_fastspeech2`'s model files and require dataset inspection or design confirmation in subsequent stages:

1. **Exact Phoneme Vocabulary Sizes for English (Svarah) vs. Telugu (IndicTTS)**:
   - Because `common/` parameterizes `n_src_vocab` via configuration rather than hardcoding `361`, the exact `n_src_vocab` values for `English/config/model.yaml` and `Telugu/config/model.yaml` will be finalized when we inspect the Svarah and IndicTTS datasets and their MFA phoneme dictionaries / G2P frontends.
2. **Single-Speaker vs. Multi-Speaker Setting for Svarah and IndicTTS**:
   - Svarah contains multiple Indian-accented English speakers, whereas IndicTTS Telugu typically has 1–2 primary speakers (male/female). `FastSpeech2` in `common/` supports both `multi_speaker: True` and `multi_speaker: False` via `model.yaml`; the exact choice per dataset will be confirmed during dataset inspection.
3. **Distribution of Utterance Frame Lengths ($L_{\text{mel}}$) in Svarah and IndicTTS Relative to `max_seq_len: 1000`**:
   - At `22050 Hz` and `hop_length = 256`, `max_seq_len = 1000` frames corresponds to $\approx 11.61\text{ seconds}$. If either Svarah or IndicTTS Telugu contains training utterances longer than $11.61\text{ s}$, we can either increase `max_seq_len` (e.g., to `1500` or `2000`) in `model.yaml` or rely on the fixed truncation in `Decoder` and `FastSpeech2Loss`.
4. **All-Zero Predicted Duration Utterance Policy at Inference (`sum(d_rounded) == 0`)**:
   - If an untrained model or extreme `d_control` causes every token in an inference utterance to round to `0` frames, `LengthRegulator.expand` produces a `[0, H]` tensor. In `common/model/modules.py`, we recommend clamping the minimum total frame count of an utterance to `1` frame (repeating the first token once if `expanded.shape[0] == 0`) so downstream `Conv1d` layers never crash on a zero-length sequence.

