"""Feed-Forward Transformer block (`FFTBlock`) and backward-compatible PostNet exports.

Implements the specifications and approved corrections in:
- research/COMMON_ARCHITECTURE_SPECIFICATION.md (Sections 2.4, 2.9)
- research/IMPLEMENTATION_DECISIONS.md (DEC-05, DEC-07, DEC-14, DEF-01)
- research/POSTNET_ANALYSIS.md
- research/ATTENTION_FFT_DEEP_ANALYSIS.md

Note:
    `PostNet` and `ConvNorm` implementations have been decoupled into
    :mod:`common.model.postnet`. They are re-exported here to maintain 100%
    backward compatibility for existing imports from `common.transformer.layers`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from common.config.model import ModelConfig, TransformerConfig
from common.model.postnet import ConvNorm, PostNet
from common.transformer.sublayers import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    _extract_transformer_config,
)


class FFTBlock(nn.Module):
    """Feed-Forward Transformer (`FFTBlock`) combining `MultiHeadAttention` and `PositionwiseFeedForward`.

    Preserves the exact submodule names (`slf_attn`, `pos_ffn`) and interleaved
    post-attention and post-FFN padding zeroing (`masked_fill(mask.unsqueeze(-1), 0)`)
    of `original_fastspeech2/transformer/Layers.py`, while guarding `if mask is not None:`
    so `mask=None` works cleanly (DEC-07 / B-07).
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_k: int,
        d_v: int,
        d_inner: int,
        kernel_size: Sequence[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    @classmethod
    def from_config(
        cls,
        config: TransformerConfig | ModelConfig | Mapping[str, Any],
        *,
        is_decoder: bool = False,
    ) -> FFTBlock:
        """Instantiate an `FFTBlock` from a `TransformerConfig`, `ModelConfig`, or dict."""
        t_cfg = _extract_transformer_config(config)
        d_model = t_cfg.decoder_hidden if is_decoder else t_cfg.encoder_hidden
        n_head = t_cfg.decoder_head if is_decoder else t_cfg.encoder_head
        dropout = t_cfg.decoder_dropout if is_decoder else t_cfg.encoder_dropout
        d_k = d_v = d_model // n_head
        return cls(
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            d_inner=t_cfg.conv_filter_size,
            kernel_size=t_cfg.conv_kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        enc_input: torch.Tensor,
        mask: torch.Tensor | None = None,
        slf_attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one Feed-Forward Transformer block.

        Args:
            enc_input: Input hidden tensor `[B, L, d_model]`.
            mask: Optional 2D padding mask `[B, L]` (`True = PAD`) used to zero out
                padded query positions after self-attention and after the Conv1D FFN.
            slf_attn_mask: Optional 3D key-padding mask `[B, L, L]` (`True = PAD`).

        Returns:
            2-tuple `(enc_output, enc_slf_attn)`:
              - `enc_output`:   `[B, L, d_model]` (`0.0` at padded positions when `mask` is provided).
              - `enc_slf_attn`: `[n_head * B, L, L]` attention weight tensor.
        """
        if not isinstance(enc_input, torch.Tensor) or enc_input.dim() != 3:
            raise ValueError(
                f"FFTBlock expects a 3D input tensor [B, L, d_model], got "
                f"{type(enc_input).__name__} with shape {getattr(enc_input, 'shape', None)}."
            )
        batch_size, seq_len, _ = enc_input.shape

        if mask is not None:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"FFTBlock 'mask' must have shape {(batch_size, seq_len)}, "
                    f"got {tuple(mask.shape)}."
                )
            if mask.dtype != torch.bool:
                mask = mask.bool()

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


__all__ = [
    "FFTBlock",
    "ConvNorm",
    "PostNet",
]
