"""Duration, Pitch, and Energy Variance Predictor."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from common.config import ModelConfig, VariancePredictorConfig
from common.config.base import (
    _validate_dropout,
    _validate_odd_positive_int,
    _validate_positive_int,
)
from common.model.conv import Conv


class VariancePredictor(nn.Module):
    """Duration, Pitch, and Energy Predictor.

    Consists of 2 Conv1d layers with LayerNorm, ReLU, and Dropout, followed by
    a Linear projection to predict a 1D scalar contour `[B, L_src]`.
    """

    def __init__(
        self,
        input_size: int
        | ModelConfig
        | VariancePredictorConfig
        | Mapping[str, Any] = 256,
        filter_size: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Resolve configuration if passed as first argument
        if isinstance(input_size, ModelConfig):
            cfg = input_size
            self.input_size = cfg.transformer.encoder_hidden
            self.filter_size = cfg.variance_predictor.filter_size
            self.kernel = cfg.variance_predictor.kernel_size
            self.dropout = cfg.variance_predictor.dropout
        elif isinstance(input_size, VariancePredictorConfig):
            cfg_vp = input_size
            self.input_size = filter_size if filter_size != 256 else 256
            self.filter_size = cfg_vp.filter_size
            self.kernel = cfg_vp.kernel_size
            self.dropout = cfg_vp.dropout
        elif isinstance(input_size, Mapping):
            m = input_size
            # Check for model_config structure
            if "transformer" in m and isinstance(m["transformer"], Mapping):
                self.input_size = int(
                    m["transformer"].get("encoder_hidden", 256)
                )
            else:
                self.input_size = int(m.get("input_size", 256))

            if "variance_predictor" in m and isinstance(
                m["variance_predictor"], Mapping
            ):
                vp_m = m["variance_predictor"]
                self.filter_size = int(vp_m.get("filter_size", 256))
                self.kernel = int(vp_m.get("kernel_size", 3))
                self.dropout = float(vp_m.get("dropout", 0.5))
            else:
                self.filter_size = int(m.get("filter_size", 256))
                self.kernel = int(m.get("kernel_size", 3))
                self.dropout = float(m.get("dropout", 0.5))
        else:
            self.input_size = _validate_positive_int(input_size, "input_size")
            self.filter_size = _validate_positive_int(
                filter_size, "filter_size"
            )
            self.kernel = _validate_odd_positive_int(kernel_size, "kernel_size")
            self.dropout = _validate_dropout(dropout, "dropout")

        self.conv_output_size = self.filter_size

        # Two 1D convolutions with LayerNorm, ReLU, and Dropout
        # DEC-11: use (kernel - 1) // 2 padding in BOTH conv1d_1 and conv1d_2
        padding = (self.kernel - 1) // 2

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=padding,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=padding,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig | VariancePredictorConfig | Mapping[str, Any],
        input_size: int = 256,
    ) -> VariancePredictor:
        """Create a VariancePredictor from configuration."""
        if isinstance(config, ModelConfig):
            return cls(config)
        elif isinstance(config, VariancePredictorConfig):
            return cls(
                input_size=input_size,
                filter_size=config.filter_size,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            )
        elif isinstance(config, Mapping):
            return cls(config)
        raise TypeError(f"Unsupported config type: {type(config).__name__}")

    def forward(
        self, encoder_output: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict scalar variance contour (duration, pitch, or energy).

        Args:
            encoder_output: FloatTensor `[B, L_src, input_size]`.
            mask: Optional BoolTensor `[B, L_src]` where `True` indicates a `<PAD>` token.

        Returns:
            FloatTensor `[B, L_src]`. When `mask` is provided, padded positions are `0.0`.
        """
        if (
            not isinstance(encoder_output, torch.Tensor)
            or encoder_output.dim() != 3
        ):
            raise ValueError(
                f"VariancePredictor expects 3D FloatTensor [B, L, input_size], got "
                f"{type(encoder_output).__name__} with shape {getattr(encoder_output, 'shape', None)}."
            )
        batch_size, seq_len, in_dim = encoder_output.shape
        if batch_size < 1 or seq_len < 1:
            raise ValueError(
                f"VariancePredictor inputs cannot be empty, got shape {tuple(encoder_output.shape)}."
            )
        if in_dim != self.input_size:
            raise ValueError(
                f"Input feature dimension ({in_dim}) does not match "
                f"VariancePredictor input_size ({self.input_size})."
            )

        if mask is not None:
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dim() != 2
                or mask.shape != (batch_size, seq_len)
            ):
                raise ValueError(
                    f"VariancePredictor 'mask' shape {getattr(mask, 'shape', None)} must match "
                    f"input shape {(batch_size, seq_len)}."
                )
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.device != encoder_output.device:
                mask = mask.to(device=encoder_output.device)

            # Mask inputs to prevent padded tokens from bleeding across convolution receptive field
            encoder_output = encoder_output.masked_fill(mask.unsqueeze(-1), 0.0)

        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


__all__ = ["VariancePredictor"]

