"""Sequence padding mask utilities for FastSpeech 2.

Implements the mask generation specification:
- research/COMMON_ARCHITECTURE_SPECIFICATION.md (Part 1, Part 4 B-01)
- research/IMPLEMENTATION_DECISIONS.md (DEC-01)

Key guarantees:
- Follows the FastSpeech 2 convention: `True = PAD`, `False = valid`.
- Direct allocation on `lengths.device` avoiding any global device assumptions.
"""

from __future__ import annotations

import torch


def get_mask_from_lengths(
    lengths: torch.Tensor, max_len: int | None = None
) -> torch.Tensor:
    """Create a boolean padding mask from a 1D tensor of sequence lengths.

    Follows the FastSpeech 2 convention:
    - ``True``  = padded position (masked out)
    - ``False`` = valid token / frame position

    The returned mask is always allocated directly on ``lengths.device`` (DEC-01 / B-01).

    Args:
        lengths: 1D integer :class:`torch.Tensor` of shape ``[B]`` with strictly positive lengths.
        max_len: Optional maximum sequence length ($\ge \max(\text{lengths})$). Defaults to
            ``int(torch.max(lengths).item())``.

    Returns:
        Boolean :class:`torch.Tensor` of shape ``[B, max_len]`` on ``lengths.device``.

    Raises:
        TypeError: If ``lengths`` is not a :class:`torch.Tensor` or does not have an integer dtype.
        ValueError: If ``lengths`` is not 1D, is empty, contains values $\le 0$, or if ``max_len``
            is invalid or smaller than ``max(lengths)``.
    """
    if not isinstance(lengths, torch.Tensor):
        raise TypeError(
            f"'lengths' must be a torch.Tensor, got {type(lengths).__name__}."
        )
    if lengths.ndim != 1:
        raise ValueError(
            f"'lengths' must be a 1D tensor of shape [B], got {lengths.ndim}D tensor with shape {tuple(lengths.shape)}."
        )
    if lengths.numel() == 0:
        raise ValueError("'lengths' tensor cannot be empty.")
    if (
        lengths.dtype == torch.bool
        or lengths.is_floating_point()
        or lengths.is_complex()
    ):
        raise TypeError(
            f"'lengths' must have an integer dtype (e.g. torch.int64 or torch.int32), got {lengths.dtype}."
        )
    if torch.any(lengths <= 0).item():
        raise ValueError(
            f"All sequence lengths in 'lengths' must be strictly positive (> 0), got {lengths.tolist()}."
        )

    actual_max = int(torch.max(lengths).item())
    if max_len is None:
        target_max_len = actual_max
    else:
        if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len <= 0:
            raise ValueError(
                f"'max_len' must be a positive integer, got {max_len!r}."
            )
        if max_len < actual_max:
            raise ValueError(
                f"'max_len' ({max_len}) cannot be smaller than the maximum sequence length "
                f"in 'lengths' ({actual_max})."
            )
        target_max_len = max_len

    batch_size = lengths.shape[0]
    ids = (
        torch.arange(0, target_max_len, device=lengths.device, dtype=lengths.dtype)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    mask = ids >= lengths.unsqueeze(1).expand(-1, target_max_len)
    return mask


__all__ = ["get_mask_from_lengths"]

