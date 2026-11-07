"""1D Convolution wrapper matching original FastSpeech 2 transpose semantics."""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv(nn.Module):
    """1D Convolution wrapper matching original FastSpeech 2 transpose semantics.

    Accepts 3D tensors shaped `[B, L, C_in]` along sequence length `L`, transposes to
    `[B, C_in, L]` for standard `nn.Conv1d`, and transposes back to `[B, L, C_out]`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        w_init: str = "linear",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor `[B, L, C_in]`.

        Returns:
            FloatTensor `[B, L, C_out]`.
        """
        x_t = x.contiguous().transpose(1, 2)
        out = self.conv(x_t)
        return out.contiguous().transpose(1, 2)


__all__ = ["Conv"]

