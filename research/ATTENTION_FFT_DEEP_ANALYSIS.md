# Deep Analysis of Multi-Head Attention & FFT Block (`ATTENTION_FFT_DEEP_ANALYSIS.md`)

## 1. Scope & Inspected Source Files

This document provides a low-level mathematical, shape-level, and edge-case verification of the FastSpeech 2 Transformer attention and Feed-Forward Transformer (`FFTBlock`) implementation in `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2/`.

| File Path | Lines | Inspected Components |
|---|---|---|
| `original_fastspeech2/transformer/SubLayers.py` | `1–93` | `MultiHeadAttention` (`8–57`), `PositionwiseFeedForward` (`60–93`) |
| `original_fastspeech2/transformer/Modules.py` | `1–25` | `ScaledDotProductAttention` (`6–25`) |
| `original_fastspeech2/transformer/Layers.py` | `1–137` | `FFTBlock` (`11–30`), `ConvNorm` (`33–64`), `PostNet` (`67–137`) |
| `original_fastspeech2/transformer/Models.py` | `1–171` | `Encoder` (`33–100`), `Decoder` (`103–171`), `get_sinusoid_encoding_table` (`10–30`) |
| `original_fastspeech2/model/fastspeech2.py` | `59–98` | Call sites of `self.encoder`, `self.decoder`, and `self.postnet` |

---

## 2. Symbolic & Concrete Walkthrough of `MultiHeadAttention`

### 2.1 Configuration Parameters Used
From `config/*/model.yaml` and `Encoder.__init__` / `Decoder.__init__` (`transformer/Models.py` lines 41–48, 110–117):
- Batch size: $B$ (`sz_b`)
- Sequence length: $L$ (`len_q = len_k = len_v = L` in self-attention)
- Model hidden dimension: $d_{\text{model}} = 256$
- Number of attention heads: $n_{\text{head}} = 2$
- Per-head key/value dimension: $d_k = d_v = d_{\text{model}} // n_{\text{head}} = 256 // 2 = 128$
- Attention temperature: $\tau = \sqrt{d_k} = \sqrt{128} = 8\sqrt{2} \approx 11.3137085$

### 2.2 Step-by-Step Q, K, V Tensor Shape Trace (`SubLayers.py` lines 29–57 & `Modules.py` lines 14–25)

| Stage | Source File & Line | Operation | Tensor(s) | Exact Shape (Symbolic) | Concrete Example ($B=2, L=4$) | Memory Layout / Stride Notes |
|---|---|---|---|---|---|---|
| **0. Input & Residual** | `SubLayers.py:30–37` | `residual = q` | `q, k, v`, `residual` | `[B, L, 256]` | `[2, 4, 256]` | Contiguous `[B, L, d_model]` |
| **1. Linear Projection** | `SubLayers.py:39–41` | `self.w_qs(q)`<br>`self.w_ks(k)`<br>`self.w_vs(v)` |Projected `q, k, v` | `[B, L, 256]`<br>(`n_head * d_k = 256`) | `[2, 4, 256]` | Three separate `nn.Linear(256, 256, bias=True)` layers |
| **2. Head Reshaping** | `SubLayers.py:39–41` | `.view(sz_b, len_q, n_head, d_k)` | Reshaped `q, k, v` | `[B, L, 2, 128]` | `[2, 4, 2, 128]` | Splits last dim `256` into `(n_head=2, d_k=128)` |
| **3. Head Permutation** | `SubLayers.py:42–44` | `.permute(2, 0, 1, 3)` | Permuted `q, k, v` | `[2, B, L, 128]` | `[2, 2, 4, 128]` | **Non-contiguous**; places `n_head` as dim 0 and `sz_b` as dim 1 |
| **4. Fold Heads into Batch** | `SubLayers.py:42–44` | `.contiguous().view(-1, len_q, d_k)` | Folded `q, k, v` | `[2 * B, L, 128]` | `[4, 4, 128]` | Contiguous copy; indices `0..B-1` = Head 0, `B..2B-1` = Head 1 |
| **5. Mask Repetition** | `SubLayers.py:46` | `mask = mask.repeat(n_head, 1, 1)` | `mask` | `[2 * B, L, L]` | `[4, 4, 4]` | Tiles `[B, L, L]` `n_head=2` times along dim 0, matching `[2*B, ...]` head-major order |
| **6. Key Transpose** | `Modules.py:16` | `k.transpose(1, 2)` | $K^\top$ | `[2 * B, 128, L]` | `[4, 128, 4]` | Swaps sequence length $L$ and head dim $128$ |
| **7. Raw Score BMM** | `Modules.py:16` | `attn = torch.bmm(q, k.transpose(1, 2))` | `attn` (unscaled) | `[2 * B, L, L]` | `[4, 4, 4]` | Batched matrix multiply: `[2B, L, 128] @ [2B, 128, L]` |
| **8. Temperature Scaling** | `Modules.py:17` | `attn = attn / self.temperature` | `attn` (scaled) | `[2 * B, L, L]` | `[4, 4, 4]` | Divides every score by $128^{0.5} \approx 11.3137$ |
| **9. Key Masking** | `Modules.py:19–20` | `attn = attn.masked_fill(mask, -np.inf)` | `attn` (masked) | `[2 * B, L, L]` | `[4, 4, 4]` | Sets scores where `mask == True` to $-\infty$ |
| **10. Softmax** | `Modules.py:22` | `attn = self.softmax(attn)` | `attn` (weights) | `[2 * B, L, L]` | `[4, 4, 4]` | `nn.Softmax(dim=2)` normalizes each query row over keys |
| **11. Value Aggregation** | `Modules.py:23` | `output = torch.bmm(attn, v)` | `output` (per-head) | `[2 * B, L, 128]` | `[4, 4, 128]` | Batched matrix multiply: `[2B, L, L] @ [2B, L, 128]` |
| **12. Unfold Head Dim** | `SubLayers.py:49` | `output.view(n_head, sz_b, len_q, d_v)` | Unfolded `output` | `[2, B, L, 128]` | `[2, 2, 4, 128]` | Separates `n_head=2` and `sz_b=B` |
| **13. Restore Batch Dim** | `SubLayers.py:51` | `.permute(1, 2, 0, 3)` | Permuted `output` | `[B, L, 2, 128]` | `[2, 4, 2, 128]` | Moves `sz_b` back to dim 0 and `n_head` to dim 2 |
| **14. Head Concatenation** | `SubLayers.py:51` | `.contiguous().view(sz_b, len_q, -1)` | Concatenated `output` | `[B, L, 256]` | `[2, 4, 256]` | Merges `(n_head=2, d_v=128)` into `256` |
| **15. Output Projection & Dropout** | `SubLayers.py:54` | `self.dropout(self.fc(output))` | Projected `output` | `[B, L, 256]` | `[2, 4, 256]` | `nn.Linear(256, 256, bias=True)` + `nn.Dropout(0.2)` |
| **16. Residual + Post-LN** | `SubLayers.py:55` | `self.layer_norm(output + residual)` | Final `output` | `[B, L, 256]` | `[2, 4, 256]` | `nn.LayerNorm(256)` over last dim |

---

## 3. Mathematical Explanation of the Attention Equation

In `ScaledDotProductAttention` (`transformer/Modules.py` lines 6–25) and `MultiHeadAttention` (`transformer/SubLayers.py` lines 8–57), for batch item $b \in \{0, \dots, B-1\}$ and head $h \in \{0, 1\}$ (indexed at folded batch index $m = h \cdot B + b$):

$$\text{Attention}(Q_m, K_m, V_m) = \text{Softmax}_{\text{col}}\left(\frac{Q_m K_m^\top}{\sqrt{d_k}} + M_m\right) V_m$$

where:
1. **Projections**:
   $$Q_m = X_b W_h^Q + \mathbf{1} (b_h^Q)^\top \in \mathbb{R}^{L \times 128}, \quad K_m = X_b W_h^K + \mathbf{1} (b_h^K)^\top \in \mathbb{R}^{L \times 128}, \quad V_m = X_b W_h^V + \mathbf{1} (b_h^V)^\top \in \mathbb{R}^{L \times 128}$$
2. **Scaling by $\frac{1}{\sqrt{d_k}}$ (`Modules.py` line 17)**:
   - If the components of $q_{m, i}$ and $k_{m, j}$ have zero mean and unit variance, their dot product $q_{m, i} \cdot k_{m, j} = \sum_{c=1}^{128} q_{m, i, c} k_{m, j, c}$ has mean $0$ and variance $d_k = 128$ (standard deviation $\sqrt{128} \approx 11.31$).
   - Dividing by `self.temperature = np.power(d_k, 0.5)` ($\sqrt{128}$) rescales the dot-product variance back to $1$, preventing large logits from saturating the softmax into one-hot distributions with near-zero gradients.
3. **Additive Key Masking $M_m$ (`Modules.py` lines 19–20)**:
   - Implemented via `attn = attn.masked_fill(mask, -np.inf)`, which corresponds to:
     $$M_{m, i, j} = \begin{cases} -\infty & \text{if } \text{mask}_{m, i, j} = \text{True} \;\;(\text{key position } j \text{ is padded}) \\ 0 & \text{if } \text{mask}_{m, i, j} = \text{False} \;\;(\text{key position } j \text{ is valid}) \end{cases}$$
4. **Row-wise Softmax over Keys (`Modules.py` line 22)**:
   - `self.softmax = nn.Softmax(dim=2)` computes for query position $i$ and key position $j$:
     $$A_{m, i, j} = \frac{\exp\left(S_{m, i, j} + M_{m, i, j}\right)}{\sum_{j'=0}^{L-1} \exp\left(S_{m, i, j'} + M_{m, i, j'}\right)}$$
   - For any padded key position $j$ ($M_{m, i, j} = -\infty$), $\exp(-\infty) = 0$, so $A_{m, i, j} = 0.0$ exactly (provided at least one key position $j'$ in row $i$ is unmasked).
5. **Multi-Head Concatenation & Post-LN Residual (`SubLayers.py` lines 49–56)**:
   $$\text{MHA}(X) = \text{LayerNorm}\Big(X + \text{Dropout}\big(\text{Concat}(\text{head}_0, \text{head}_1) W^O + b^O\big)\Big)$$

---

## 4. Mask Construction & How Padded Positions Are Masked

Tracing how padding masks flow from sequence lengths to `ScaledDotProductAttention` and `FFTBlock`:

### Step 1: 2D Padding Mask Generation (`utils/tools.py` lines 91–99)
Given `lengths` (`[B]`, e.g., `[4, 2]`) and `max_len = 4`:
```python
ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
```
For $B = 2, L = 4$ with `lengths = [4, 2]`:
$$\text{mask} = \begin{bmatrix} \text{False} & \text{False} & \text{False} & \text{False} \\ \text{False} & \text{False} & \text{True} & \text{True} \end{bmatrix} \quad \text{shape: } [2, 4]$$

### Step 2: 3D Self-Attention Mask Expansion (`transformer/Models.py` lines 79 & 157)
```python
slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
```
- `mask.unsqueeze(1)` has shape `[B, 1, L]`.
- `.expand(-1, L, -1)` broadcasts the same key-mask row across all $L$ query rows, yielding shape `[B, L, L]`.
- For batch item $b = 1$ (length 2, padded at indices 2 and 3):
  $$\text{slf\_attn\_mask}[1] = \begin{bmatrix}
  \text{F} & \text{F} & \mathbf{T} & \mathbf{T} \\
  \text{F} & \text{F} & \mathbf{T} & \mathbf{T} \\
  \text{F} & \text{F} & \mathbf{T} & \mathbf{T} \\
  \text{F} & \text{F} & \mathbf{T} & \mathbf{T}
  \end{bmatrix} \quad (\text{rows} = \text{queries } i \in 0..3,\; \text{cols} = \text{keys } j \in 0..3)$$

### Step 3: Why `slf_attn_mask` Only Masks Columns (Keys) and `FFTBlock` Masks Rows (Queries)
- Inside `ScaledDotProductAttention`:
  - Columns $j \in \{2, 3\}$ are set to $-\infty$, so neither valid queries ($i \in \{0, 1\}$) nor padded queries ($i \in \{2, 3\}$) can attend to padded keys $j \in \{2, 3\}$.
  - Why aren't rows $i \in \{2, 3\}$ also set to $-\infty$ inside `slf_attn_mask`? Because if an entire row $i$ were $-\infty$, `Softmax(dim=2)` would compute $\frac{\exp(-\infty)}{\sum \exp(-\infty)} = \frac{0}{0} = \text{NaN}$!
  - Instead, padded query rows $i \in \{2, 3\}$ are allowed to compute harmless finite attention over valid keys $j \in \{0, 1\}$, and then **`FFTBlock.forward` (line 25) immediately zeroes out rows $i \in \{2, 3\}$**:
    ```python
    enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
    ```

---

## 5. Edge-Case Verification (Executed & Verified on Both CPU and MPS)

We executed a read-only verification script using `/Users/swaruppal/FastSpeech2_Project/.venv/bin/python` on both `cpu` and `mps:0` to test all five edge cases:

### 5.1 Edge Case 1: All Positions Masked for a Sample (`length == 0`)
- **What happens**: If a batch item $b$ has `length == 0`, then `mask[b]` is `[True, True, ..., True]` and every entry in `slf_attn_mask[b]` (`[L, L]`) is `True`.
- **Empirical Result on CPU & MPS**:
  - Inside `ScaledDotProductAttention`, `attn.masked_fill(mask, -np.inf)` fills all columns of row $i$ with $-\infty$.
  - `self.softmax(attn)` evaluates $\frac{0}{0}$ and produces **`NaN` across the entire attention matrix `attn` (`has_nan_attn = True`)**!
  - In the **forward pass**, `FFTBlock` lines 25 and 28 overwrite `enc_output[b]` with `0.0` via `enc_output.masked_fill(mask.unsqueeze(-1), 0)`, so `out[b]` contains `0.0` (`has_nan_out = False`).
  - **CRITICAL BACKWARD PASS HAZARD**: Even though `masked_fill` zeroes the forward activation `enc_output[b]`, `attn` itself is `NaN` (`return_attns`), and if any upstream/downstream operation or custom loss inspects `attn`, it sees `NaN`.
  - **When could `length == 0` occur?**:
    1. If `text_to_sequence` filters out all unknown characters (e.g., unmapped Telugu script), yielding `src_len == 0`.
    2. If `LengthRegulator` predicts `duration_rounded == 0` for all tokens in an utterance (`mel_len == 0`).

### 5.2 Edge Case 2: Sequence Length $L = 1$
- **Empirical Result on CPU & MPS**:
  - `q, k, v` after permutation have shape `[2 * B, 1, 128]`.
  - `attn` has shape `[2 * B, 1, 1]` and after softmax equals `[[[1.0]]]`.
  - `PositionwiseFeedForward` (`Conv1d` with `kernel_size=9, padding=4`) pads 4 zeros on the left and 4 zeros on the right of the single time step (`1 + 2(4) - 9 + 1 = 1`), preserving shape `[B, 1, 256]`.
  - **Verdict**: Works without error on both CPU and MPS.

### 5.3 Edge Case 3: Different Sequence Lengths Within One Batch (`B = 2, L = 4, lens = [4, 2]`)
- **Empirical Result on CPU & MPS**:
  - Sample 0 (`len = 4`) has no masked positions; Sample 1 (`len = 2`) has positions `2, 3` masked (`True`).
  - After `self.slf_attn`: Sample 1's valid positions `0, 1` have zero attention weight on keys `2, 3`. Sample 1's padded positions `2, 3` are non-zero due to `self.fc` bias and `self.layer_norm`.
  - At `FFTBlock` line 25 (`enc_output.masked_fill(mask.unsqueeze(-1), 0)`): Sample 1's positions `2, 3` are zeroed out (`0.0`).
  - Inside `self.pos_ffn`: `self.w_1` (`Conv1d(k=9, pad=4)`) convolves across time. Because Sample 1's positions `2, 3` were zeroed at line 25, they act as exact zero-padding for valid positions `0, 1` (preventing padding length $L$ from altering the convolution output at valid positions `0, 1`!).
  - At `FFTBlock` line 28 (`enc_output.masked_fill(mask.unsqueeze(-1), 0)`): Sample 1's positions `2, 3` (which became non-zero again from `w_1` convolving positions `0, 1` into `2, 3` plus Conv1d/LayerNorm biases) are zeroed out again.
  - Verified output norm at padded positions: `out[1, 2:].abs().max().item() == 0.0` on both `cpu` and `mps`.

### 5.4 Edge Case 4: `mask=None` or `slf_attn_mask=None`
- **Code Trace**:
  - `FFTBlock.forward(self, enc_input, mask=None, slf_attn_mask=None)` (`Layers.py` line 21) and `MultiHeadAttention.forward(self, q, k, v, mask=None)` (`SubLayers.py` line 29) both declare `None` as the default parameter value.
  - However:
    - `SubLayers.py` line 46 unconditionally executes `mask = mask.repeat(n_head, 1, 1)` $\rightarrow$ raises `AttributeError: 'NoneType' object has no attribute 'repeat'` if `slf_attn_mask` is `None`.
    - `Layers.py` lines 25 and 28 unconditionally execute `enc_output.masked_fill(mask.unsqueeze(-1), 0)` $\rightarrow$ raises `AttributeError: 'NoneType' object has no attribute 'unsqueeze'` if `mask` is `None`.
- **Verdict**: Confirmed bug when called with default arguments.

### 5.5 Edge Case 5: CPU and Apple Silicon MPS Tensor Placement
- **Empirical Result**:
  - Tested `FFTBlock` on both `torch.device("cpu")` and `torch.device("mps")` (`PyTorch 2.10.0`).
  - All tensor operations inside `FFTBlock`, `MultiHeadAttention`, `ScaledDotProductAttention`, and `PositionwiseFeedForward` (`view`, `permute`, `contiguous`, `repeat`, `bmm`, `masked_fill` with `-np.inf` and `0`, `Softmax`, `Conv1d`, `LayerNorm`) execute natively on `mps:0` and `cpu`.
  - The **only** MPS failure point in the attention pipeline is upstream in `utils/tools.py` line 96 (`get_mask_from_lengths`), which allocates `ids` on module-global `device` (`"cpu"`). Once `mask` is on `mps:0`, `Encoder`, `Decoder`, and `FFTBlock` run on `mps:0` without any device errors.

---

## 6. Trace of `FFTBlock` Execution (Residuals, LayerNorm, and Padding Zeroing)

```text
Input: enc_input [B, L, 256] (padded positions = 0.0), mask [B, L], slf_attn_mask [B, L, L]
  │
  ├──► residual_1 = enc_input                                      [B, L, 256]
  │
  ├──► MultiHeadAttention:
  │      q, k, v projections -> ScaledDotProductAttention(slf_attn_mask)
  │      -> head concat -> fc -> dropout(0.2)                      [B, L, 256]
  │      -> LayerNorm(output + residual_1)                         [B, L, 256]
  │      (Note: padded positions are now NON-ZERO due to fc/LN biases!)
  │
  ├──► Line 25: enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
  │      (Padded positions reset to 0.0 before Conv1d k=9!)        [B, L, 256]
  │
  ├──► residual_2 = enc_output                                     [B, L, 256]
  │
  ├──► PositionwiseFeedForward:
  │      transpose(1,2) -> Conv1d(256->1024, k=9, pad=4) -> ReLU
  │      -> Conv1d(1024->256, k=1, pad=0) -> transpose(1,2)
  │      -> dropout(0.2)                                           [B, L, 256]
  │      -> LayerNorm(output + residual_2)                         [B, L, 256]
  │      (Note: padded positions are again NON-ZERO due to k=9 bleed + biases!)
  │
  └──► Line 28: enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
         (Padded positions reset to 0.0 before next FFTBlock!)     [B, L, 256]
```

---

## 7. Verification of `return_attns` Behavior & Repository Callers

1. **Is `return_attns` actually broken in `Encoder.forward` and `Decoder.forward`?**
   - **Yes, confirmed**:
     - `transformer/Models.py` lines 73 & 97–100 (`Encoder.forward`):
       ```python
       for enc_layer in self.layer_stack:
           enc_output, enc_slf_attn = enc_layer(
               enc_output, mask=mask, slf_attn_mask=slf_attn_mask
           )
           if return_attns:
               enc_slf_attn_list += [enc_slf_attn]

       return enc_output
       ```
     - `transformer/Models.py` lines 139 & 168–171 (`Decoder.forward`):
       ```python
       for dec_layer in self.layer_stack:
           dec_output, dec_slf_attn = dec_layer(
               dec_output, mask=mask, slf_attn_mask=slf_attn_mask
           )
           if return_attns:
               dec_slf_attn_list += [dec_slf_attn]

       return dec_output, mask
       ```
     - Neither method ever returns `enc_slf_attn_list` or `dec_slf_attn_list`.
2. **Does any existing caller in `original_fastspeech2` depend on `return_attns`?**
   - **No, confirmed via full-repository search (`grep -rn "return_attns\|self\.encoder(\|self\.decoder("`)**:
     - The **only** call to `self.encoder` in the entire repository is `model/fastspeech2.py` line 66: `output = self.encoder(texts, src_masks)`.
     - The **only** call to `self.decoder` in the entire repository is `model/fastspeech2.py` line 94: `output, mel_masks = self.decoder(output, mel_masks)`.
     - No file in the repository ever passes `return_attns=True`.

---

## 8. Investigation of Unused `ConvNorm.w_init_gain` & Its Effect on `PostNet`

1. **Code Trace (`transformer/Layers.py` lines 33–128)**:
   - `PostNet.__init__` constructs 5 `ConvNorm` layers:
     - Layer 0 (`lines 85–93`): passes `w_init_gain="tanh"`
     - Layers 1–3 (`lines 101–109`): passes `w_init_gain="tanh"`
     - Layer 4 (`lines 116–124`): passes `w_init_gain="linear"`
   - Inside `ConvNorm.__init__` (`lines 34–60`):
     ```python
     def __init__(
         self,
         in_channels,
         out_channels,
         kernel_size=1,
         stride=1,
         padding=None,
         dilation=1,
         bias=True,
         w_init_gain="linear",
     ):
         super(ConvNorm, self).__init__()
         if padding is None:
             assert kernel_size % 2 == 1
             padding = int(dilation * (kernel_size - 1) / 2)
         self.conv = torch.nn.Conv1d(
             in_channels, out_channels, kernel_size=kernel_size,
             stride=stride, padding=padding, dilation=dilation, bias=bias,
         )
     ```
   - `w_init_gain` is never referenced after the parameter list.
   - *(Note: The exact same unused parameter pattern also exists in `Conv.__init__` inside `model/modules.py` line 267: `w_init="linear"` is accepted and documented in the docstring as `"weight inits with xavier initialization"`, but never used!)*
2. **What Initialization Actually Occurs in `PostNet` & Does It Hurt Training?**:
   - Because `torch.nn.init.xavier_uniform_` is not called, `torch.nn.Conv1d` runs its built-in `reset_parameters()`, which initializes weights with **Kaiming Uniform** ($\mathcal{U}(-b, b)$ where $b = \frac{1}{\sqrt{C_{\text{in}} \cdot K}}$) and biases with $\mathcal{U}(-b, b)$.
   - For `PostNet`'s internal layers ($C_{\text{in}} = 512, C_{\text{out}} = 512, K = 5$):
     - Default Kaiming Uniform bound: $b_{\text{kaiming}} = \frac{1}{\sqrt{512 \times 5}} = \frac{1}{\sqrt{2560}} \approx 0.01976$.
     - Xavier Uniform bound with `gain("tanh") = 5/3`: $b_{\text{xavier}} = \frac{5}{3} \sqrt{\frac{6}{(512 + 512) \times 5}} \approx 0.05705$.
   - Crucially, **every single `ConvNorm` in `PostNet` is immediately followed by `nn.BatchNorm1d`** (`Layers.py` lines 94, 110, 125) *before* the `torch.tanh` activation (`line 133`). Because `BatchNorm1d` rescales any linear multiplication of `self.conv.weight` to zero mean and unit variance on the very first forward pass, the 2.88x scale difference between Kaiming and Xavier initialization is absorbed by `BatchNorm1d`.
   - Furthermore, all pretrained checkpoints in `ming024/FastSpeech2` were trained with this exact `ConvNorm` implementation.

---

## 9. Summary: Confirmed Findings vs. Edge Cases vs. Proposed Improvements for `common/`

### 9.1 Confirmed Findings
1. **Attention Head & Mask Alignment Is Mathematically Sound**: The head-major permutation `permute(2, 0, 1, 3)` (`[n_head * B, L, d_k]`) in `MultiHeadAttention.forward` (lines 42–44) matches `mask.repeat(n_head, 1, 1)` (line 46) and its inverse `view(n_head, sz_b, len_q, d_v).permute(1, 2, 0, 3)` (lines 49–51).
2. **Interleaved Padding Zeroing in `FFTBlock` Is Essential**: Zeroing padded positions at line 25 (between `MultiHeadAttention` and `PositionwiseFeedForward`) prevents `Conv1d(kernel_size=9)` from leaking padded states into valid tokens, and line 28 restores exact zeros at padded positions after `PositionwiseFeedForward`.
3. **`return_attns` Is Unused Dead Code**: Confirmed that `return_attns` is never passed by any caller in the repository and that `Encoder.forward` and `Decoder.forward` discard the collected lists.
4. **`ConvNorm.w_init_gain` Is Unused Dead Code**: Confirmed that `w_init_gain` is ignored in `ConvNorm.__init__`, leaving `nn.Conv1d`'s default Kaiming Uniform initialization active (mitigated by `nn.BatchNorm1d` in `PostNet`).
5. **Unconditional `mask` Dereferencing**: `MultiHeadAttention.forward` (line 46) and `FFTBlock.forward` (lines 25, 28) fail with `AttributeError` if called with their default `mask=None`.

### 9.2 Potential Edge Cases
1. **All-Masked Utterance (`len == 0`) Produces `NaN` in Softmax**: If every key in a row is masked (`True`), `Softmax(dim=2)` over all `-np.inf` produces `NaN` in `attn`.
2. **Training Sequence Length `> max_seq_len` (`1000`) in `Encoder`**: `Encoder.forward` (lines 88–91) does not clamp `max_len` during training (`self.training == True`), whereas `Decoder.forward` (line 154) does.

### 9.3 Behaviors to Preserve vs. Improve in `common/transformer/`

| Aspect | Action in `common/transformer/` | Rationale |
|---|---|---|
| **Module Hierarchy, Parameter Names & Tensor Shapes** | **Preserve 100%** (`Encoder`, `Decoder`, `FFTBlock`, `MultiHeadAttention`, `PositionwiseFeedForward`, `ScaledDotProductAttention`, `PostNet`, `ConvNorm`) | Guarantees checkpoint `state_dict` compatibility and identical model capacity (`~27M` parameters). |
| **Interleaved `masked_fill(mask.unsqueeze(-1), 0)` in `FFTBlock`** | **Preserve 100%** (with `if mask is not None:` guard) | Essential for preventing `Conv1d(kernel_size=9)` padding leakage across variable-length batches. |
| **`if mask is not None:` Guards in `MultiHeadAttention` & `FFTBlock`** | **Improve** | Guard `if mask is not None: mask = mask.repeat(n_head, 1, 1)` in `MultiHeadAttention.forward` and `if mask is not None: enc_output = enc_output.masked_fill(...)` in `FFTBlock.forward` so standalone unit tests with `mask=None` do not crash. |
| **`return_attns=True` Return Value in `Encoder` & `Decoder`** | **Improve** | Return `(enc_output, enc_slf_attn_list)` when `return_attns=True` (and `enc_output` when `False`, preserving default behavior) so attention maps can be inspected/plotted if desired. |
| **Configurable `n_src_vocab` in `Encoder.__init__`** | **Improve** | Allow `n_src_vocab = config.get("n_src_vocab", len(symbols) + 1)` so English and Telugu models can either share a unified `symbols` list or specify their vocabulary size explicitly. |
| **`PostNet` `n_mel_channels` Parameter** | **Improve** | Pass `n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"]` when instantiating `PostNet` in `FastSpeech2.__init__` (defaults to `80`). |

