"""Acoustic Mel-Spectrogram Refinement Network (PostNet) for FastSpeech 2.

Implements the 5-layer 1D convolutional PostNet predicting residual corrections
to decoder Mel-spectrograms:

    postnet_mel = postnet(mel, mask) + mel

Features:
    - 5-layer 1D ConvNorm + BatchNorm1d hierarchy matching reference FastSpeech 2
    - Mask-Aware BatchNorm ensuring zeroed padded frames cannot bias batch statistics
    - Exact parameter names and shapes preserved for 100% checkpoint compatibility
    - Dynamic device support for CPU, Apple Silicon MPS, and CUDA
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config.base import (
    _validate_dropout,
    _validate_odd_positive_int,
    _validate_positive_int,
)
from common.config.model import ModelConfig, PostNetConfig
from common.config.preprocess import MelConfig, PreprocessConfig


class ConvNorm(nn.Module):
    """1D Convolution wrapper with automatic odd-kernel same-padding.

    Preserves exact parameter naming (`self.conv`) for checkpoint compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
    ) -> None:
        super().__init__()
        in_channels = _validate_positive_int(in_channels, "in_channels")
        out_channels = _validate_positive_int(out_channels, "out_channels")
        kernel_size = _validate_positive_int(kernel_size, "kernel_size")
        stride = _validate_positive_int(stride, "stride")
        dilation = _validate_positive_int(dilation, "dilation")

        if padding is None:
            if kernel_size % 2 != 1:
                raise ValueError(
                    f"ConvNorm automatic same-padding requires an odd kernel_size, got {kernel_size}."
                )
            padding = int(dilation * (kernel_size - 1) // 2)
        else:
            padding = _validate_positive_int(padding, "padding", min_value=0)

        self.w_init_gain = w_init_gain
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply 1D convolution to `signal` `[B, C, L]`."""
        if not isinstance(signal, torch.Tensor) or signal.dim() != 3:
            raise ValueError(
                f"ConvNorm expects a 3D FloatTensor [B, C, L], got "
                f"{type(signal).__name__} with shape {getattr(signal, 'shape', None)}."
            )
        return self.conv(signal)


class PostNet(nn.Module):
    """5-layer 1D convolutional PostNet predicting residual Mel-spectrogram corrections.

    Preserves the exact module hierarchy and state_dict keys (`self.convolutions.0`..`4`)
    of the original FastSpeech 2 repository:
      - Layers 0..3: ConvNorm(k=5, p=2) + BatchNorm1d(embedding_dim) + Tanh + Dropout
      - Layer 4:     ConvNorm(k=5, p=2) + BatchNorm1d(n_mel_channels) + Dropout (linear, no Tanh)

    The forward pass outputs the residual tensor ΔMel [B, L, n_mel_channels].
    Refined prediction is obtained via: `postnet_mel = self.postnet(mel) + mel`.
    """

    def __init__(
        self,
        n_mel_channels: int = 80,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolutions: int = 5,
        postnet_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_mel_channels = _validate_positive_int(
            n_mel_channels, "n_mel_channels"
        )
        self.postnet_embedding_dim = _validate_positive_int(
            postnet_embedding_dim, "postnet_embedding_dim"
        )
        self.postnet_kernel_size = _validate_odd_positive_int(
            postnet_kernel_size, "postnet_kernel_size"
        )
        self.postnet_n_convolutions = _validate_positive_int(
            postnet_n_convolutions, "postnet_n_convolutions", min_value=2
        )
        self.postnet_dropout = _validate_dropout(
            postnet_dropout, "postnet_dropout"
        )

        self.convolutions = nn.ModuleList()

        # Layer 0: n_mel_channels -> postnet_embedding_dim
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    self.n_mel_channels,
                    self.postnet_embedding_dim,
                    kernel_size=self.postnet_kernel_size,
                    stride=1,
                    padding=(self.postnet_kernel_size - 1) // 2,
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(self.postnet_embedding_dim),
            )
        )

        # Layers 1 to (n_convolutions - 2): postnet_embedding_dim -> postnet_embedding_dim
        for _ in range(1, self.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        self.postnet_embedding_dim,
                        self.postnet_embedding_dim,
                        kernel_size=self.postnet_kernel_size,
                        stride=1,
                        padding=(self.postnet_kernel_size - 1) // 2,
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(self.postnet_embedding_dim),
                )
            )

        # Layer (n_convolutions - 1): postnet_embedding_dim -> n_mel_channels (linear)
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    self.postnet_embedding_dim,
                    self.n_mel_channels,
                    kernel_size=self.postnet_kernel_size,
                    stride=1,
                    padding=(self.postnet_kernel_size - 1) // 2,
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(self.n_mel_channels),
            )
        )

    @classmethod
    def from_config(
        cls,
        config: PostNetConfig | ModelConfig | PreprocessConfig | Mapping[str, Any],
        n_mel_channels: int | None = None,
    ) -> PostNet:
        """Instantiate `PostNet` from a configuration object or mapping."""
        # 1. Resolve n_mel_channels
        resolved_mels: int = 80
        if n_mel_channels is not None:
            resolved_mels = n_mel_channels
        elif isinstance(config, PreprocessConfig):
            resolved_mels = config.preprocessing.mel.n_mel_channels
        elif isinstance(config, MelConfig):
            resolved_mels = config.n_mel_channels
        elif isinstance(config, Mapping):
            if "n_mel_channels" in config and config["n_mel_channels"] is not None:
                resolved_mels = int(config["n_mel_channels"])
            elif (
                "preprocessing" in config
                and isinstance(config["preprocessing"], Mapping)
                and "mel" in config["preprocessing"]
                and isinstance(config["preprocessing"]["mel"], Mapping)
                and "n_mel_channels" in config["preprocessing"]["mel"]
            ):
                resolved_mels = int(
                    config["preprocessing"]["mel"]["n_mel_channels"]
                )
            elif (
                "mel" in config
                and isinstance(config["mel"], Mapping)
                and "n_mel_channels" in config["mel"]
            ):
                resolved_mels = int(config["mel"]["n_mel_channels"])

        # 2. Resolve PostNet hyperparameters
        p_dim = 512
        p_kernel = 5
        p_convs = 5
        p_drop = 0.5

        if isinstance(config, PostNetConfig):
            p_dim = config.postnet_embedding_dim
            p_kernel = config.postnet_kernel_size
            p_convs = config.postnet_n_convolutions
            p_drop = config.postnet_dropout
        elif isinstance(config, ModelConfig):
            p_dim = config.postnet.postnet_embedding_dim
            p_kernel = config.postnet.postnet_kernel_size
            p_convs = config.postnet.postnet_n_convolutions
            p_drop = config.postnet.postnet_dropout
        elif isinstance(config, Mapping):
            postnet_raw = config.get("postnet")
            if isinstance(postnet_raw, PostNetConfig):
                p_dim = postnet_raw.postnet_embedding_dim
                p_kernel = postnet_raw.postnet_kernel_size
                p_convs = postnet_raw.postnet_n_convolutions
                p_drop = postnet_raw.postnet_dropout
            elif isinstance(postnet_raw, Mapping):
                p_dim = postnet_raw.get("postnet_embedding_dim", 512)
                p_kernel = postnet_raw.get("postnet_kernel_size", 5)
                p_convs = postnet_raw.get("postnet_n_convolutions", 5)
                p_drop = postnet_raw.get("postnet_dropout", 0.5)
            elif "postnet_embedding_dim" in config:
                p_dim = config.get("postnet_embedding_dim", 512)
                p_kernel = config.get("postnet_kernel_size", 5)
                p_convs = config.get("postnet_n_convolutions", 5)
                p_drop = config.get("postnet_dropout", 0.5)

        return cls(
            n_mel_channels=resolved_mels,
            postnet_embedding_dim=p_dim,
            postnet_kernel_size=p_kernel,
            postnet_n_convolutions=p_convs,
            postnet_dropout=p_drop,
        )

    @staticmethod
    def _masked_batch_norm(
        x: torch.Tensor,
        mask_t: torch.Tensor,
        bn: nn.BatchNorm1d,
    ) -> torch.Tensor:
        """Compute batch normalization strictly over valid (unmasked) frames.

        Args:
            x: FloatTensor `[B, C, L]`, output of 1D convolution.
            mask_t: BoolTensor `[B, 1, L]` where `True` indicates a `<PAD>` frame.
            bn: Target `nn.BatchNorm1d` layer providing affine parameters and running stats.

        Returns:
            Normalized FloatTensor `[B, C, L]` with valid frames normalized by true
            sample statistics and padded positions strictly zeroed out.
        """
        valid_mask = (~mask_t).float()
        k = valid_mask.sum(dim=(0, 2), keepdim=True)  # [1, 1, 1]
        k_val = k.item()
        if k_val < 1.0:
            raise ValueError(
                "Cannot compute Mask-Aware BatchNorm: all frames in the batch are masked (<PAD>)."
            )

        # 1. Sample mean across valid frames only: [1, C, 1]
        mean = (x * valid_mask).sum(dim=(0, 2), keepdim=True) / k

        # 2. Biased variance across valid frames for current batch normalization: [1, C, 1]
        var_biased = (((x - mean) * valid_mask) ** 2).sum(dim=(0, 2), keepdim=True) / k

        # 3. Unbiased variance for updating running_var (Bessel's correction N / (N - 1))
        if k_val > 1.0:
            var_unbiased = var_biased * (k_val / (k_val - 1.0))
        else:
            var_unbiased = var_biased

        # 4. Update running statistics buffer
        if bn.track_running_stats:
            with torch.no_grad():
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked.add_(1)
                    if bn.momentum is None:
                        # Cumulative moving average
                        exponential_average_factor = 1.0 / float(
                            bn.num_batches_tracked.item()
                        )
                    else:
                        exponential_average_factor = bn.momentum
                else:
                    exponential_average_factor = (
                        bn.momentum if bn.momentum is not None else 0.1
                    )

                if (
                    exponential_average_factor is not None
                    and exponential_average_factor > 0.0
                ):
                    if bn.running_mean is not None:
                        bn.running_mean.copy_(
                            (1.0 - exponential_average_factor) * bn.running_mean
                            + exponential_average_factor * mean.view(-1)
                        )
                    if bn.running_var is not None:
                        bn.running_var.copy_(
                            (1.0 - exponential_average_factor) * bn.running_var
                            + exponential_average_factor * var_unbiased.view(-1)
                        )

        # 5. Normalization with eps and affine parameters
        inv_std = torch.rsqrt(var_biased + bn.eps)
        x_hat = (x - mean) * inv_std
        if bn.affine and bn.weight is not None and bn.bias is not None:
            w = bn.weight.view(1, -1, 1)
            b = bn.bias.view(1, -1, 1)
            out = x_hat * w + b
        else:
            out = x_hat

        # 6. Ensure padded positions are strictly zeroed
        return out.masked_fill(mask_t, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the residual Mel correction tensor ΔMel.

        Args:
            x: FloatTensor `[B, L, n_mel_channels]`.
            mask: Optional BoolTensor `[B, L]` where `True` indicates a `<PAD>` frame.

        Returns:
            Residual FloatTensor `[B, L, n_mel_channels]`. When `mask` is provided,
            padded positions are strictly `0.0`.
        """
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            raise ValueError(
                f"PostNet expects a 3D FloatTensor [B, L, n_mel_channels], got "
                f"{type(x).__name__} with shape {getattr(x, 'shape', None)}."
            )
        batch_size, seq_len, n_mels = x.shape
        if batch_size < 1:
            raise ValueError(
                f"PostNet batch size must be >= 1, got shape {tuple(x.shape)}."
            )
        if seq_len < 1:
            raise ValueError(
                f"PostNet sequence length must be >= 1, got shape {tuple(x.shape)}."
            )
        if n_mels != self.n_mel_channels:
            raise ValueError(
                f"Input Mel channel dimension ({n_mels}) does not match "
                f"PostNet n_mel_channels ({self.n_mel_channels})."
            )

        if mask is not None:
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dim() != 2
                or mask.shape != (batch_size, seq_len)
            ):
                raise ValueError(
                    f"PostNet 'mask' shape {getattr(mask, 'shape', None)} must match "
                    f"input shape {(batch_size, seq_len)}."
                )
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.device != x.device:
                mask = mask.to(device=x.device)
            mask_t = mask.unsqueeze(1)  # [B, 1, L]
        else:
            mask_t = None

        # [B, L, C] -> [B, C, L]
        x_t = x.contiguous().transpose(1, 2)
        if mask_t is not None:
            x_t = x_t.masked_fill(mask_t, 0.0)

        # 1. Intermediate convolutional layers with Tanh activation
        for i in range(len(self.convolutions) - 1):
            conv = self.convolutions[i][0]
            bn = self.convolutions[i][1]
            x_t = conv(x_t)
            if self.training and mask_t is not None:
                x_t = self._masked_batch_norm(x_t, mask_t, bn)
            else:
                x_t = bn(x_t)
                if mask_t is not None:
                    x_t = x_t.masked_fill(mask_t, 0.0)
            x_t = torch.tanh(x_t)
            if self.postnet_dropout > 0.0:
                x_t = F.dropout(x_t, self.postnet_dropout, self.training)
            if mask_t is not None:
                x_t = x_t.masked_fill(mask_t, 0.0)

        # 2. Final convolutional layer (linear, no Tanh)
        conv = self.convolutions[-1][0]
        bn = self.convolutions[-1][1]
        x_t = conv(x_t)
        if self.training and mask_t is not None:
            x_t = self._masked_batch_norm(x_t, mask_t, bn)
        else:
            x_t = bn(x_t)
            if mask_t is not None:
                x_t = x_t.masked_fill(mask_t, 0.0)
        if self.postnet_dropout > 0.0:
            x_t = F.dropout(x_t, self.postnet_dropout, self.training)
        if mask_t is not None:
            x_t = x_t.masked_fill(mask_t, 0.0)

        # [B, C, L] -> [B, L, C]
        return x_t.contiguous().transpose(1, 2)


__all__ = ["ConvNorm", "PostNet"]

