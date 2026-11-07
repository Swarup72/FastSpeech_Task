"""Transformer sublayers: Multi-Head Attention and Position-wise Convolutional Feed-Forward.

Implements the specifications and approved corrections in:
- research/COMMON_ARCHITECTURE_SPECIFICATION.md (Sections 2.2, 2.3)
- research/IMPLEMENTATION_DECISIONS.md (DEC-07, DEC-08, DEC-14)
- research/ATTENTION_FFT_DEEP_ANALYSIS.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config.base import (
    _validate_dropout,
    _validate_odd_positive_int,
    _validate_positive_int,
)
from common.config.model import ModelConfig, TransformerConfig
from common.transformer.modules import ScaledDotProductAttention


def _extract_transformer_config(
    config: TransformerConfig | ModelConfig | Mapping[str, Any],
) -> TransformerConfig:
    if isinstance(config, TransformerConfig):
        return config
    if isinstance(config, ModelConfig):
        return config.transformer
    if isinstance(config, Mapping):
        if "transformer" in config and isinstance(
            config["transformer"], (Mapping, TransformerConfig)
        ):
            raw = config["transformer"]
            return (
                raw
                if isinstance(raw, TransformerConfig)
                else TransformerConfig.from_dict(raw)
            )
        return TransformerConfig.from_dict(config)
    raise TypeError(
        f"Expected TransformerConfig, ModelConfig, or Mapping, got {type(config).__name__}."
    )


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module with Post-LN residual connection.

    Preserves the exact parameter names (`w_qs`, `w_ks`, `w_vs`, `attention`,
    `layer_norm`, `fc`, `dropout`) and head-major permutation order of
    `original_fastspeech2/transformer/SubLayers.py`, while adding `mask=None`
    safety (DEC-07 / B-07) and shape/mask validation.
    """

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_head = _validate_positive_int(n_head, "n_head")
        self.d_model = _validate_positive_int(d_model, "d_model")
        self.d_k = _validate_positive_int(d_k, "d_k")
        self.d_v = _validate_positive_int(d_v, "d_v")
        dropout_val = _validate_dropout(dropout, "dropout")

        self.w_qs = nn.Linear(self.d_model, self.n_head * self.d_k)
        self.w_ks = nn.Linear(self.d_model, self.n_head * self.d_k)
        self.w_vs = nn.Linear(self.d_model, self.n_head * self.d_v)

        self.attention = ScaledDotProductAttention(
            temperature=float(np.power(self.d_k, 0.5))
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.fc = nn.Linear(self.n_head * self.d_v, self.d_model)

        self.dropout = nn.Dropout(dropout_val)

    @classmethod
    def from_config(
        cls,
        config: TransformerConfig | ModelConfig | Mapping[str, Any],
        *,
        is_decoder: bool = False,
    ) -> MultiHeadAttention:
        """Instantiate `MultiHeadAttention` from a `TransformerConfig`, `ModelConfig`, or dict."""
        t_cfg = _extract_transformer_config(config)
        n_head = t_cfg.decoder_head if is_decoder else t_cfg.encoder_head
        d_model = t_cfg.decoder_hidden if is_decoder else t_cfg.encoder_hidden
        dropout = t_cfg.decoder_dropout if is_decoder else t_cfg.encoder_dropout
        d_k = d_v = d_model // n_head
        return cls(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention with residual connection and layer normalization.

        Args:
            q: Query tensor `[B, L_q, d_model]`.
            k: Key tensor `[B, L_k, d_model]`.
            v: Value tensor `[B, L_v, d_model]` (`L_v == L_k`).
            mask: Optional boolean mask `[B, L_q, L_k]` (`True = PAD`).

        Returns:
            2-tuple `(output, attn)`:
              - `output`: `[B, L_q, d_model]`
              - `attn`:   `[n_head * B, L_q, L_k]`
        """
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                f"MultiHeadAttention expects 3D tensors (q, k, v), got shapes "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
            )

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, dim_q = q.size()
        sz_b_k, len_k, dim_k = k.size()
        sz_b_v, len_v, dim_v = v.size()

        if sz_b == 0 or len_q == 0 or len_k == 0:
            raise ValueError(
                f"MultiHeadAttention requires non-empty batch and sequence lengths, "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}."
            )
        if sz_b != sz_b_k or sz_b != sz_b_v:
            raise ValueError(
                f"Batch size mismatch across q ({sz_b}), k ({sz_b_k}), v ({sz_b_v})."
            )
        if len_k != len_v:
            raise ValueError(
                f"Key sequence length ({len_k}) must match value sequence length ({len_v})."
            )
        if dim_q != self.d_model or dim_k != self.d_model or dim_v != self.d_model:
            raise ValueError(
                f"Input hidden dimensions ({dim_q}, {dim_k}, {dim_v}) must match d_model ({self.d_model})."
            )

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            if mask.shape != (sz_b, len_q, len_k):
                raise ValueError(
                    f"MultiHeadAttention mask must have shape {(sz_b, len_q, len_k)}, "
                    f"got {tuple(mask.shape)}."
                )
            if mask.dtype != torch.bool:
                mask = mask.bool()
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x lq x lk

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Two-layer 1D-convolutional position-wise feed-forward network with Post-LN residual.

    Preserves the exact parameter names (`w_1`, `w_2`, `layer_norm`, `dropout`) of
    `original_fastspeech2/transformer/SubLayers.py`, while validating odd positive
    kernel sizes so `(k - 1) // 2` same-padding always preserves sequence length `L`.
    """

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        kernel_size: Sequence[int] = (9, 1),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_in = _validate_positive_int(d_in, "d_in")
        self.d_hid = _validate_positive_int(d_hid, "d_hid")
        if (
            not isinstance(kernel_size, (list, tuple))
            or len(kernel_size) != 2
        ):
            raise ValueError(
                f"'kernel_size' must be a 2-element sequence of odd positive integers, "
                f"got {kernel_size!r}."
            )
        k0 = _validate_odd_positive_int(kernel_size[0], "kernel_size[0]")
        k1 = _validate_odd_positive_int(kernel_size[1], "kernel_size[1]")
        self.kernel_size = (k0, k1)
        dropout_val = _validate_dropout(dropout, "dropout")

        # Use Conv1D (position-wise across sequence dimension)
        self.w_1 = nn.Conv1d(
            self.d_in,
            self.d_hid,
            kernel_size=k0,
            padding=(k0 - 1) // 2,
        )
        self.w_2 = nn.Conv1d(
            self.d_hid,
            self.d_in,
            kernel_size=k1,
            padding=(k1 - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(self.d_in)
        self.dropout = nn.Dropout(dropout_val)

    @classmethod
    def from_config(
        cls,
        config: TransformerConfig | ModelConfig | Mapping[str, Any],
        *,
        is_decoder: bool = False,
    ) -> PositionwiseFeedForward:
        """Instantiate `PositionwiseFeedForward` from a `TransformerConfig`, `ModelConfig`, or dict."""
        t_cfg = _extract_transformer_config(config)
        d_in = t_cfg.decoder_hidden if is_decoder else t_cfg.encoder_hidden
        dropout = t_cfg.decoder_dropout if is_decoder else t_cfg.encoder_dropout
        return cls(
            d_in=d_in,
            d_hid=t_cfg.conv_filter_size,
            kernel_size=t_cfg.conv_kernel_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D convolutional FFN to `x` (`[B, L, d_in] -> [B, L, d_in]`)."""
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            raise ValueError(
                f"PositionwiseFeedForward expects a 3D tensor [B, L, d_in], got "
                f"{type(x).__name__} with shape {getattr(x, 'shape', None)}."
            )
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError(
                f"PositionwiseFeedForward requires non-empty batch and sequence length, "
                f"got shape {tuple(x.shape)}."
            )
        if x.size(2) != self.d_in:
            raise ValueError(
                f"Input feature dimension ({x.size(2)}) must match d_in ({self.d_in})."
            )

        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


__all__ = [
    "MultiHeadAttention",
    "PositionwiseFeedForward",
]

