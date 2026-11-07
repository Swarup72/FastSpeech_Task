"""Core Transformer primitives: Sinusoidal Positional Encoding and Scaled Dot-Product Attention.

Implements the specifications and approved corrections in:
- research/COMMON_ARCHITECTURE_SPECIFICATION.md
- research/IMPLEMENTATION_DECISIONS.md (DEC-01, DEC-08, DEC-14)
- research/ATTENTION_FFT_DEEP_ANALYSIS.md
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def get_sinusoid_encoding_table(
    n_position: int,
    d_hid: int,
    padding_idx: int | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a sinusoidal positional encoding table of shape `[n_position, d_hid]`.

    Preserves the exact mathematical formula of `original_fastspeech2/transformer/Models.py`:
        angle(pos, j) = pos / 10000^(2 * (j // 2) / d_hid)
        table[:, 0::2] = sin(angle[:, 0::2])
        table[:, 1::2] = cos(angle[:, 1::2])

    Args:
        n_position: Number of sequence positions to encode (must be >= 1).
        d_hid: Hidden representation dimension (must be >= 1).
        padding_idx: Optional position index to zero out.
        device: Optional target `torch.device` or device string (defaults to CPU).
        dtype: Output tensor floating-point dtype (defaults to `torch.float32`).

    Returns:
        `Tensor[n_position, d_hid]` on `device` with `dtype`.
    """
    if isinstance(n_position, bool) or not isinstance(n_position, int) or n_position < 1:
        raise ValueError(
            f"'n_position' must be a positive integer >= 1, got {n_position!r}."
        )
    if isinstance(d_hid, bool) or not isinstance(d_hid, int) or d_hid < 1:
        raise ValueError(
            f"'d_hid' must be a positive integer >= 1, got {d_hid!r}."
        )

    def cal_angle(position: int, hid_idx: int) -> float:
        return float(position / np.power(10000, 2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position: int) -> list[float]:
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)],
        dtype=np.float64,
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        if (
            isinstance(padding_idx, bool)
            or not isinstance(padding_idx, int)
            or not (-n_position <= padding_idx < n_position)
        ):
            raise ValueError(
                f"'padding_idx' ({padding_idx!r}) is out of bounds for n_position={n_position}."
            )
        sinusoid_table[padding_idx] = 0.0

    table = torch.tensor(sinusoid_table, dtype=dtype)
    if device is not None:
        table = table.to(device)
    return table


class PositionalEncoding(nn.Module):
    """Device-aware sinusoidal positional encoding module (`[1, max_seq_len + 1, d_model]`).

    Stores `self.position_enc` as a non-trainable `nn.Parameter` matching the original
    FastSpeech 2 `Encoder` / `Decoder` representation, and dynamically expands on
    `x.device` when `seq_len > max_seq_len` during evaluation.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 1000,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model < 1:
            raise ValueError(
                f"'d_model' must be a positive integer >= 1, got {d_model!r}."
            )
        if (
            isinstance(max_seq_len, bool)
            or not isinstance(max_seq_len, int)
            or max_seq_len < 1
        ):
            raise ValueError(
                f"'max_seq_len' must be a positive integer >= 1, got {max_seq_len!r}."
            )
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        n_position = max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(
                n_position, d_model, padding_idx=padding_idx
            ).unsqueeze(0),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding to `x` (`[B, L, d_model]`)."""
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            raise ValueError(
                f"PositionalEncoding expects a 3D tensor [B, L, d_model], got "
                f"{type(x).__name__} with shape {getattr(x, 'shape', None)}."
            )
        batch_size, seq_len, d_hid = x.shape
        if seq_len < 1:
            raise ValueError(
                f"PositionalEncoding requires sequence length L >= 1, got shape {tuple(x.shape)}."
            )
        if d_hid != self.d_model:
            raise ValueError(
                f"Input hidden dimension ({d_hid}) does not match PositionalEncoding d_model ({self.d_model})."
            )

        if seq_len > self.position_enc.size(1):
            pos_table = (
                get_sinusoid_encoding_table(
                    seq_len,
                    self.d_model,
                    device=x.device,
                    dtype=x.dtype,
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
        else:
            pos_table = (
                self.position_enc[:, :seq_len, :]
                .to(device=x.device, dtype=x.dtype)
                .expand(batch_size, -1, -1)
            )
        return x + pos_table


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention (`Attention(Q, K, V) = Softmax(QK^T / temperature) V`).

    Padding mask convention: `True = PAD` (masked key positions are filled with `-inf`).
    Implements DEC-08 / B-06 to safely handle all-masked query rows without producing
    `0 / 0 = NaN` in forward attention weights or backward gradients.
    """

    def __init__(self, temperature: float) -> None:
        super().__init__()
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float, np.number))
            or float(temperature) <= 0.0
        ):
            raise ValueError(
                f"'temperature' must be a positive number > 0, got {temperature!r}."
            )
        self.temperature = float(temperature)
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        Args:
            q: Query tensor `[N, L_q, d_k]`.
            k: Key tensor `[N, L_k, d_k]`.
            v: Value tensor `[N, L_v, d_v]` where `L_v == L_k`.
            mask: Optional boolean mask `[N, L_q, L_k]` where `True` indicates a padded position.

        Returns:
            2-tuple `(output, attn)`:
              - `output`: `[N, L_q, d_v]`
              - `attn`:   `[N, L_q, L_k]`
        """
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                f"ScaledDotProductAttention expects 3D tensors (q, k, v), got shapes "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
            )
        if q.size(0) == 0 or q.size(1) == 0 or k.size(1) == 0:
            raise ValueError(
                f"ScaledDotProductAttention requires non-empty batch and sequence dimensions, "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}."
            )
        if q.size(0) != k.size(0) or q.size(0) != v.size(0):
            raise ValueError(
                f"Batch size mismatch across q ({q.size(0)}), k ({k.size(0)}), v ({v.size(0)})."
            )
        if k.size(1) != v.size(1):
            raise ValueError(
                f"Key length ({k.size(1)}) must equal value length ({v.size(1)})."
            )
        if q.size(2) != k.size(2):
            raise ValueError(
                f"Query feature dim ({q.size(2)}) must equal key feature dim ({k.size(2)})."
            )

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            if mask.shape != attn.shape:
                raise ValueError(
                    f"Attention mask shape {tuple(mask.shape)} does not match attention score "
                    f"shape {tuple(attn.shape)}."
                )
            if mask.dtype != torch.bool:
                mask = mask.bool()

            # DEC-08 / B-06: Guard against all-True key rows where every key is masked.
            # Keeping all-masked rows finite before Softmax prevents `0 / 0 = NaN` inside
            # both Softmax forward and SoftmaxBackward0, and masking with 0.0 post-Softmax
            # zeroes out the attention weights and gradients for those rows cleanly.
            all_masked_rows = mask.all(dim=-1, keepdim=True)
            if all_masked_rows.any():
                effective_mask = mask & ~all_masked_rows
                attn = attn.masked_fill(effective_mask, -np.inf)
                attn = self.softmax(attn)
                attn = attn.masked_fill(all_masked_rows, 0.0)
            else:
                attn = attn.masked_fill(mask, -np.inf)
                attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        output = torch.bmm(attn, v)
        return output, attn


__all__ = [
    "get_sinusoid_encoding_table",
    "PositionalEncoding",
    "ScaledDotProductAttention",
]

