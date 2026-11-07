"""FastSpeech 2 Transformer Encoder and Decoder modules.

Implements the specifications and approved corrections in:
- research/COMMON_ARCHITECTURE_SPECIFICATION.md (Sections 2.1, 2.5)
- research/IMPLEMENTATION_DECISIONS.md (DEC-01, DEC-06, DEC-09, DEC-14)
- research/TRANSFORMER_ANALYSIS.md
- research/ATTENTION_FFT_DEEP_ANALYSIS.md
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from common.config.base import (
    _validate_positive_int,
)
from common.config.model import ModelConfig, TransformerConfig
from common.transformer.constants import PAD
from common.transformer.layers import FFTBlock
from common.transformer.modules import get_sinusoid_encoding_table
from common.transformer.sublayers import _extract_transformer_config


class Encoder(nn.Module):
    """FastSpeech 2 Transformer Encoder.

    Transforms discrete phoneme token IDs `[B, L_src]` into contextual hidden
    representations `[B, L_src, d_model]` using token embeddings, sinusoidal positional
    encoding, and a stack of `FFTBlock` layers.

    Decoupled from any global English vocabulary import (DEC-06); accepts language-specific
    `n_src_vocab` via constructor argument or `ModelConfig`.
    """

    def __init__(
        self,
        config: TransformerConfig | ModelConfig | Mapping[str, Any],
        n_src_vocab: int | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()

        # Resolve language-specific vocabulary size (DEC-06)
        resolved_vocab: int | None = n_src_vocab
        if resolved_vocab is None:
            if isinstance(config, ModelConfig):
                resolved_vocab = config.n_src_vocab
            elif isinstance(config, Mapping):
                if "n_src_vocab" in config and config["n_src_vocab"] is not None:
                    resolved_vocab = config["n_src_vocab"]
                elif (
                    "model" in config
                    and isinstance(config["model"], Mapping)
                    and "n_src_vocab" in config["model"]
                ):
                    resolved_vocab = config["model"]["n_src_vocab"]
                elif (
                    "transformer" in config
                    and isinstance(config["transformer"], Mapping)
                    and "n_src_vocab" in config["transformer"]
                ):
                    resolved_vocab = config["transformer"]["n_src_vocab"]

        if resolved_vocab is None:
            raise ValueError(
                "Encoder requires 'n_src_vocab' to be specified either via constructor "
                "argument or inside 'config.n_src_vocab'. Global symbol imports are prohibited (DEC-06)."
            )
        self.n_src_vocab = _validate_positive_int(
            resolved_vocab, "n_src_vocab", min_value=2
        )

        # Resolve max sequence length
        resolved_max_seq_len: int = 1000
        if max_seq_len is not None:
            resolved_max_seq_len = max_seq_len
        elif isinstance(config, ModelConfig):
            resolved_max_seq_len = config.max_seq_len
        elif isinstance(config, Mapping):
            if "max_seq_len" in config and config["max_seq_len"] is not None:
                resolved_max_seq_len = config["max_seq_len"]
            elif (
                "model" in config
                and isinstance(config["model"], Mapping)
                and "max_seq_len" in config["model"]
            ):
                resolved_max_seq_len = config["model"]["max_seq_len"]
        self.max_seq_len = _validate_positive_int(
            resolved_max_seq_len, "max_seq_len", min_value=1
        )

        # Extract transformer hyperparameters
        t_cfg = _extract_transformer_config(config)
        self.d_model = t_cfg.encoder_hidden
        self.n_head = t_cfg.encoder_head
        self.d_k = self.d_model // self.n_head
        self.d_v = self.d_model // self.n_head
        self.d_inner = t_cfg.conv_filter_size
        self.kernel_size = t_cfg.conv_kernel_size
        self.dropout = t_cfg.encoder_dropout
        self.n_layers = t_cfg.encoder_layer

        # Submodules (preserves exact state_dict keys: src_word_emb, position_enc, layer_stack)
        self.src_word_emb = nn.Embedding(
            self.n_src_vocab, self.d_model, padding_idx=PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(
                self.max_seq_len + 1, self.d_model
            ).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model,
                    self.n_head,
                    self.d_k,
                    self.d_v,
                    self.d_inner,
                    self.kernel_size,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

    @classmethod
    def from_config(
        cls,
        config: TransformerConfig | ModelConfig | Mapping[str, Any],
        n_src_vocab: int | None = None,
        max_seq_len: int | None = None,
    ) -> Encoder:
        """Instantiate `Encoder` from a configuration object or mapping."""
        return cls(config, n_src_vocab=n_src_vocab, max_seq_len=max_seq_len)

    def forward(
        self,
        src_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attns: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode input token IDs into contextual hidden states.

        Args:
            src_seq: LongTensor `[B, L_src]` containing token IDs in `[0, n_src_vocab - 1]`.
            mask: Optional BoolTensor `[B, L_src]` where `True` indicates a `<PAD>` position.
            return_attns: If `True`, return `(enc_output, enc_slf_attn_list)`.
                If `False` (default), return `enc_output` (DEC-09).

        Returns:
            - If `return_attns=False`: `enc_output` `[B, L_src, d_model]`.
            - If `return_attns=True`: `(enc_output, enc_slf_attn_list)` where each attention
              tensor in `enc_slf_attn_list` has shape `[n_head * B, L_src, L_src]`.
        """
        if not isinstance(src_seq, torch.Tensor) or src_seq.dim() != 2:
            raise ValueError(
                f"'src_seq' must be a 2D LongTensor [B, L_src], got "
                f"{type(src_seq).__name__} with shape {getattr(src_seq, 'shape', None)}."
            )
        batch_size, max_len = src_seq.shape
        if batch_size < 1:
            raise ValueError(
                f"'src_seq' batch size must be >= 1, got shape {tuple(src_seq.shape)}."
            )
        if max_len < 1:
            raise ValueError(
                f"'src_seq' sequence length must be >= 1, got shape {tuple(src_seq.shape)}."
            )

        # Validate token ID bounds
        if (src_seq < 0).any() or (src_seq >= self.n_src_vocab).any():
            min_id = int(src_seq.min().item())
            max_id = int(src_seq.max().item())
            raise ValueError(
                f"Token IDs in 'src_seq' must be in range [0, {self.n_src_vocab - 1}], "
                f"but got min={min_id}, max={max_id}."
            )

        # Validate mask
        if mask is not None:
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dim() != 2
                or mask.shape != src_seq.shape
            ):
                raise ValueError(
                    f"'mask' shape {getattr(mask, 'shape', None)} must match 'src_seq' shape {tuple(src_seq.shape)}."
                )
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.device != src_seq.device:
                mask = mask.to(device=src_seq.device)
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        else:
            slf_attn_mask = None

        # Positional encoding addition
        if not self.training and max_len > self.max_seq_len:
            # Inference dynamic expansion (DEC-01)
            pos_enc = (
                get_sinusoid_encoding_table(
                    max_len,
                    self.d_model,
                    device=src_seq.device,
                    dtype=self.position_enc.dtype,
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            enc_output = self.src_word_emb(src_seq) + pos_enc
        elif self.training and max_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({max_len}) exceeds max_seq_len ({self.max_seq_len}) during training."
            )
        else:
            pos_enc = (
                self.position_enc[:, :max_len, :]
                .to(device=src_seq.device, dtype=self.src_word_emb.weight.dtype)
                .expand(batch_size, -1, -1)
            )
            enc_output = self.src_word_emb(src_seq) + pos_enc

        # FFTBlock stack
        enc_slf_attn_list: list[torch.Tensor] = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list.append(enc_slf_attn)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    """FastSpeech 2 Transformer Decoder.

    Transforms continuous representations `[B, L_mel, d_model]` from the Variance Adaptor
    into decoded acoustic hidden states using sinusoidal positional encoding and a stack
    of `FFTBlock` layers.

    Preserves original FastSpeech 2 training sequence truncation behavior (`min(L_mel, max_seq_len)`)
    and supports dynamic sinusoidal table expansion during inference when `L_mel > max_seq_len`.
    """

    def __init__(
        self,
        config: TransformerConfig | ModelConfig | Mapping[str, Any],
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()

        # Resolve max sequence length
        resolved_max_seq_len: int = 1000
        if max_seq_len is not None:
            resolved_max_seq_len = max_seq_len
        elif isinstance(config, ModelConfig):
            resolved_max_seq_len = config.max_seq_len
        elif isinstance(config, Mapping):
            if "max_seq_len" in config and config["max_seq_len"] is not None:
                resolved_max_seq_len = config["max_seq_len"]
            elif (
                "model" in config
                and isinstance(config["model"], Mapping)
                and "max_seq_len" in config["model"]
            ):
                resolved_max_seq_len = config["model"]["max_seq_len"]
        self.max_seq_len = _validate_positive_int(
            resolved_max_seq_len, "max_seq_len", min_value=1
        )

        # Extract transformer hyperparameters
        t_cfg = _extract_transformer_config(config)
        self.d_model = t_cfg.decoder_hidden
        self.n_head = t_cfg.decoder_head
        self.d_k = self.d_model // self.n_head
        self.d_v = self.d_model // self.n_head
        self.d_inner = t_cfg.conv_filter_size
        self.kernel_size = t_cfg.conv_kernel_size
        self.dropout = t_cfg.decoder_dropout
        self.n_layers = t_cfg.decoder_layer

        # Submodules (preserves exact state_dict keys: position_enc, layer_stack)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(
                self.max_seq_len + 1, self.d_model
            ).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model,
                    self.n_head,
                    self.d_k,
                    self.d_v,
                    self.d_inner,
                    self.kernel_size,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

    @classmethod
    def from_config(
        cls,
        config: TransformerConfig | ModelConfig | Mapping[str, Any],
        max_seq_len: int | None = None,
    ) -> Decoder:
        """Instantiate `Decoder` from a configuration object or mapping."""
        return cls(config, max_seq_len=max_seq_len)

    def forward(
        self,
        enc_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attns: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]
    ):
        """Decode acoustic hidden states.

        Args:
            enc_seq: FloatTensor `[B, L_mel, d_model]`.
            mask: Optional BoolTensor `[B, L_mel]` where `True` indicates a `<PAD>` position.
            return_attns: If `True`, return `(dec_output, mask, dec_slf_attn_list)`.
                If `False` (default), return `(dec_output, mask)` (DEC-09).

        Returns:
            - If `return_attns=False`: `(dec_output, mask)` where `dec_output` has shape
              `[B, L_dec, d_model]` and `mask` has shape `[B, L_dec]`.
            - If `return_attns=True`: `(dec_output, mask, dec_slf_attn_list)`.
        """
        if not isinstance(enc_seq, torch.Tensor) or enc_seq.dim() != 3:
            raise ValueError(
                f"'enc_seq' must be a 3D FloatTensor [B, L_mel, d_model], got "
                f"{type(enc_seq).__name__} with shape {getattr(enc_seq, 'shape', None)}."
            )
        batch_size, orig_len, d_hid = enc_seq.shape
        if batch_size < 1:
            raise ValueError(
                f"'enc_seq' batch size must be >= 1, got shape {tuple(enc_seq.shape)}."
            )
        if orig_len < 1:
            raise ValueError(
                f"'enc_seq' sequence length must be >= 1, got shape {tuple(enc_seq.shape)}."
            )
        if d_hid != self.d_model:
            raise ValueError(
                f"'enc_seq' feature dimension ({d_hid}) must match decoder hidden dimension ({self.d_model})."
            )

        if mask is not None:
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dim() != 2
                or mask.shape != (batch_size, orig_len)
            ):
                raise ValueError(
                    f"'mask' shape {getattr(mask, 'shape', None)} must match 'enc_seq' shape {(batch_size, orig_len)}."
                )
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.device != enc_seq.device:
                mask = mask.to(device=enc_seq.device)

        # Handle max_seq_len truncation vs. dynamic expansion
        if not self.training and orig_len > self.max_seq_len:
            # Inference long-utterance expansion: preserve full length (DEC-01)
            max_len = orig_len
            if mask is not None:
                slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            else:
                slf_attn_mask = None

            pos_enc = (
                get_sinusoid_encoding_table(
                    max_len,
                    self.d_model,
                    device=enc_seq.device,
                    dtype=self.position_enc.dtype,
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            dec_output = enc_seq + pos_enc
        else:
            # Training truncation or normal-length inference
            max_len = min(orig_len, self.max_seq_len)
            if mask is not None:
                mask = mask[:, :max_len]
                slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            else:
                slf_attn_mask = None

            pos_enc = (
                self.position_enc[:, :max_len, :]
                .to(device=enc_seq.device, dtype=enc_seq.dtype)
                .expand(batch_size, -1, -1)
            )
            dec_output = enc_seq[:, :max_len, :] + pos_enc

        # FFTBlock stack
        dec_slf_attn_list: list[torch.Tensor] = []
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list.append(dec_slf_attn)

        if return_attns:
            return dec_output, mask, dec_slf_attn_list
        return dec_output, mask


__all__ = [
    "Encoder",
    "Decoder",
    "get_sinusoid_encoding_table",
]

