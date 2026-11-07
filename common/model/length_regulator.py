"""Length Regulator for phoneme-to-frame sequence expansion."""

from __future__ import annotations

import torch
import torch.nn as nn

from common.utils.pad import expand, pad


class LengthRegulator(nn.Module):
    """Length Regulator for expanding phoneme representations according to durations."""

    def __init__(self) -> None:
        super().__init__()

    def expand(
        self, batch: torch.Tensor, predicted: torch.Tensor
    ) -> torch.Tensor:
        """Expand a single utterance's token embeddings according to per-token durations.

        Args:
            batch: FloatTensor `[L_src, H]` representing phoneme hidden states.
            predicted: 1D Tensor `[L_src]` of non-negative integer frame counts.

        Returns:
            FloatTensor `[T, H]` where `T = sum(predicted)`. Zero-duration phonemes
            produce 0 frames and are dropped from the expanded representation.
        """
        return expand(batch, predicted)

    def LR(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform length regulation across a batch of utterances.

        Args:
            x: FloatTensor `[B, L_src, H]` from Encoder.
            duration: LongTensor or FloatTensor `[B, L_src]` containing non-negative durations.
            max_len: Optional target maximum sequence length (frames) for padding.

        Returns:
            A tuple of `(output, mel_len)`:
            - `output`: FloatTensor `[B, L_mel, H]` padded to `max_len` (or `max(mel_len)`).
            - `mel_len`: LongTensor `[B]` containing the actual expanded frame count per sample,
              allocated on `x.device` (DEC-01).
        """
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            raise ValueError(
                f"LengthRegulator expects 3D FloatTensor [B, L_src, H], got "
                f"{type(x).__name__} with shape {getattr(x, 'shape', None)}."
            )
        batch_size, seq_len, hidden_dim = x.shape
        if batch_size < 1 or seq_len < 1:
            raise ValueError(
                f"LengthRegulator input tensors cannot be empty, got shape {tuple(x.shape)}."
            )

        if not isinstance(duration, torch.Tensor) or duration.dim() != 2:
            raise ValueError(
                f"LengthRegulator expects 2D duration tensor [B, L_src], got "
                f"{type(duration).__name__} with shape {getattr(duration, 'shape', None)}."
            )
        if duration.shape != (batch_size, seq_len):
            raise ValueError(
                f"Duration shape {tuple(duration.shape)} does not match "
                f"input shape {(batch_size, seq_len)}."
            )
        if torch.any(duration < 0).item():
            raise ValueError(
                f"LengthRegulator received negative duration values: "
                f"{duration[duration < 0].tolist()}. Durations must be non-negative."
            )

        # Convert durations to long tensor on x.device
        if duration.dtype == torch.bool or duration.is_complex():
            raise ValueError(
                f"LengthRegulator duration dtype {duration.dtype} is not supported."
            )
        if duration.is_floating_point():
            # The LengthRegulator never rounds or truncates. Float tensors are accepted
            # only when every value is already an exact non-negative integer.
            if not torch.isfinite(duration).all().item() or torch.any(
                duration != torch.round(duration)
            ).item():
                raise ValueError(
                    "LengthRegulator received non-integral or non-finite float durations; "
                    "convert log-durations to integer frame counts before expansion "
                    "(see VarianceAdaptor.log_duration_to_frames)."
                )
            duration_int = duration.to(device=x.device, dtype=torch.long)
        else:
            duration_int = duration.to(device=x.device, dtype=torch.long)

        out_list: list[torch.Tensor] = []
        mel_lens: list[int] = []

        for b in range(batch_size):
            expanded = self.expand(x[b], duration_int[b])
            out_list.append(expanded)
            mel_lens.append(int(expanded.shape[0]))

        actual_max = max(mel_lens) if mel_lens else 0

        # Edge case: All durations in batch are zero and max_len is None or 0
        if actual_max == 0 and (max_len is None or max_len == 0):
            output = torch.zeros(
                batch_size, 0, hidden_dim, device=x.device, dtype=x.dtype
            )
        else:
            output = pad(out_list, mel_max_length=max_len)

        # DEC-01: Allocate mel_len on x.device dynamically (replaces global device variable)
        mel_len_tensor = torch.tensor(
            mel_lens, dtype=torch.long, device=x.device
        )

        return output, mel_len_tensor

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass delegating to `LR`."""
        return self.LR(x, duration, max_len)


__all__ = ["LengthRegulator"]

