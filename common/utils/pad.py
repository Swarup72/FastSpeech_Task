"""Sequence padding and duration-based expansion utilities for FastSpeech 2.

Implements padding and expansion specifications:
- research/COMMON_ARCHITECTURE_SPECIFICATION.md (Part 1, Part 4 B-01, Part 7.13)
- research/IMPLEMENTATION_DECISIONS.md (DEC-01, DEC-08, DEC-13)

Key guarantees:
- Zero-padding for 1D and 2D PyTorch tensors and NumPy arrays.
- Preserves input tensor dtypes and devices.
- Duration-based repetition for phoneme-to-frame alignment.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def pad(
    input_ele: Sequence[torch.Tensor], mel_max_length: int | None = None
) -> torch.Tensor:
    """Zero-pad a sequence of 1D ``[L_i]`` or 2D ``[L_i, D]`` tensors to a common length.

    Preserves the ``dtype`` and ``device`` of the input tensors.

    Args:
        input_ele: Non-empty sequence (``list`` or ``tuple``) of 1D or 2D :class:`torch.Tensor`
            instances sharing the same ``ndim``, ``dtype``, ``device``, and (if 2D) trailing
            feature dimension ``D``.
        mel_max_length: Optional target sequence length ($\ge \max_i L_i$). If ``None``, pads
            to $\max_i L_i$.

    Returns:
        Stacked :class:`torch.Tensor` of shape ``[B, max_len]`` (for 1D inputs) or
        ``[B, max_len, D]`` (for 2D inputs) on the input tensors' device.

    Raises:
        TypeError: If ``input_ele`` is not a sequence of :class:`torch.Tensor` objects.
        ValueError: If ``input_ele`` is empty, contains tensors with unsupported dimensions
            or mismatched shapes/dtypes/devices, or if ``mel_max_length`` is invalid.
    """
    if isinstance(input_ele, (torch.Tensor, np.ndarray, str, bytes)) or not isinstance(
        input_ele, Sequence
    ):
        raise TypeError(
            f"'input_ele' must be a list or tuple of torch.Tensor objects, got {type(input_ele).__name__}."
        )
    if len(input_ele) == 0:
        raise ValueError("'input_ele' cannot be empty.")

    first = input_ele[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(
            f"Elements of 'input_ele' must be torch.Tensor objects, got {type(first).__name__} at index 0."
        )
    if first.ndim not in (1, 2):
        raise ValueError(
            f"pad() supports 1D [L] or 2D [L, D] tensors, got {first.ndim}D tensor at index 0."
        )

    expected_ndim = first.ndim
    expected_dtype = first.dtype
    expected_device = first.device
    expected_feat_dim = first.size(1) if expected_ndim == 2 else None

    lengths: list[int] = []
    for idx, tensor in enumerate(input_ele):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Element at index {idx} of 'input_ele' is not a torch.Tensor (got {type(tensor).__name__})."
            )
        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"Inconsistent tensor dimensions in 'input_ele': index 0 is {expected_ndim}D, "
                f"but index {idx} is {tensor.ndim}D."
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"Inconsistent tensor dtypes in 'input_ele': index 0 has {expected_dtype}, "
                f"but index {idx} has {tensor.dtype}."
            )
        if tensor.device != expected_device:
            raise ValueError(
                f"Inconsistent tensor devices in 'input_ele': index 0 is on {expected_device}, "
                f"but index {idx} is on {tensor.device}."
            )
        if expected_ndim == 2 and tensor.size(1) != expected_feat_dim:
            raise ValueError(
                f"Inconsistent feature dimensions in 'input_ele': index 0 has shape[1]={expected_feat_dim}, "
                f"but index {idx} has shape[1]={tensor.size(1)}."
            )
        lengths.append(int(tensor.size(0)))

    actual_max = max(lengths)
    if mel_max_length is not None:
        if (
            isinstance(mel_max_length, bool)
            or not isinstance(mel_max_length, int)
            or mel_max_length <= 0
        ):
            raise ValueError(
                f"'mel_max_length' must be a positive integer, got {mel_max_length!r}."
            )
        if mel_max_length < actual_max:
            raise ValueError(
                f"'mel_max_length' ({mel_max_length}) cannot be smaller than the longest "
                f"tensor in 'input_ele' ({actual_max})."
            )
        max_len = mel_max_length
    else:
        if actual_max <= 0:
            raise ValueError(
                "Cannot pad tensors when all sequences have length 0 and 'mel_max_length' is None."
            )
        max_len = actual_max

    out_list: list[torch.Tensor] = []
    for batch in input_ele:
        pad_amount = max_len - int(batch.size(0))
        if expected_ndim == 1:
            one_batch_padded = (
                F.pad(batch, (0, pad_amount), "constant", 0)
                if pad_amount > 0
                else batch
            )
        else:
            one_batch_padded = (
                F.pad(batch, (0, 0, 0, pad_amount), "constant", 0)
                if pad_amount > 0
                else batch
            )
        out_list.append(one_batch_padded)

    return torch.stack(out_list, dim=0)


def pad_1D(inputs: Sequence[np.ndarray], PAD: int | float = 0) -> np.ndarray:
    """Pad a non-empty sequence of 1D NumPy arrays to the maximum length in the batch."""
    if not isinstance(inputs, Sequence) or len(inputs) == 0:
        raise ValueError("'inputs' must be a non-empty sequence of 1D NumPy arrays.")
    for idx, x in enumerate(inputs):
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError(
                f"Element at index {idx} of 'inputs' must be a 1D np.ndarray."
            )
    max_len = max(len(x) for x in inputs)
    if max_len <= 0:
        raise ValueError("Cannot pad 1D arrays when maximum length is 0.")
    return np.stack(
        [
            np.pad(x, (0, max_len - x.shape[0]), mode="constant", constant_values=PAD)
            for x in inputs
        ]
    )


def pad_2D(
    inputs: Sequence[np.ndarray], maxlen: int | None = None
) -> np.ndarray:
    """Pad a non-empty sequence of 2D NumPy arrays along axis 0 to ``maxlen`` (or max batch length)."""
    if not isinstance(inputs, Sequence) or len(inputs) == 0:
        raise ValueError("'inputs' must be a non-empty sequence of 2D NumPy arrays.")
    first_feat = None
    for idx, x in enumerate(inputs):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise ValueError(
                f"Element at index {idx} of 'inputs' must be a 2D np.ndarray."
            )
        if first_feat is None:
            first_feat = x.shape[1]
        elif x.shape[1] != first_feat:
            raise ValueError(
                f"Inconsistent feature dimension in pad_2D: index 0 has {first_feat}, "
                f"index {idx} has {x.shape[1]}."
            )
    actual_max = max(x.shape[0] for x in inputs)
    if maxlen is not None:
        if isinstance(maxlen, bool) or not isinstance(maxlen, int) or maxlen < actual_max:
            raise ValueError(
                f"'maxlen' ({maxlen}) must be an integer >= max sequence length ({actual_max})."
            )
        target_max = maxlen
    else:
        if actual_max <= 0:
            raise ValueError("Cannot pad 2D arrays when maximum length is 0.")
        target_max = actual_max

    return np.stack(
        [
            np.pad(
                x,
                ((0, target_max - x.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            for x in inputs
        ]
    )


def expand(
    values: torch.Tensor | np.ndarray,
    durations: torch.Tensor | np.ndarray | Sequence[int | float],
) -> torch.Tensor | np.ndarray:
    """Expand a 1D ``[L]`` or 2D ``[L, D]`` sequence along axis 0 according to ``durations``.

    Used for expanding phoneme-level pitch/energy contours to frame level as well as
    repeating per-token representations by integer frame counts. Preserves sequence order,
    ``dtype``, and (for PyTorch tensors) ``values.device``.

    Args:
        values: 1D ``[L]`` or 2D ``[L, D]`` :class:`torch.Tensor` or :class:`numpy.ndarray`.
        durations: 1D sequence/array/tensor of non-negative frame counts of length ``L``.

    Returns:
        Expanded :class:`torch.Tensor` (on ``values.device`` with ``values.dtype``) if ``values``
        is a Tensor, or expanded :class:`numpy.ndarray` (with ``values.dtype``) if ``values`` is
        a NumPy array.

    Raises:
        TypeError: If ``values`` or ``durations`` has an invalid type or boolean dtype.
        ValueError: If ``values`` or ``durations`` is empty, not 1D/2D, has mismatched lengths,
            or if ``durations`` contains negative values.
    """
    if not isinstance(values, (torch.Tensor, np.ndarray)):
        raise TypeError(
            f"'values' must be a torch.Tensor or np.ndarray, got {type(values).__name__}."
        )
    if values.ndim not in (1, 2):
        raise ValueError(
            f"'values' must be a 1D [L] or 2D [L, D] tensor/array, got {values.ndim}D."
        )
    seq_len = int(values.shape[0])
    if seq_len == 0:
        raise ValueError("'values' cannot be empty along dimension 0.")

    # Normalize and validate durations
    if isinstance(durations, torch.Tensor):
        if durations.ndim != 1:
            raise ValueError(
                f"'durations' tensor must be 1D, got {durations.ndim}D."
            )
        if durations.dtype == torch.bool or durations.is_complex():
            raise TypeError(
                f"'durations' tensor cannot have dtype {durations.dtype}."
            )
        if int(durations.shape[0]) != seq_len:
            raise ValueError(
                f"Length mismatch between 'values' ({seq_len}) and 'durations' ({ int(durations.shape[0]) })."
            )
        if torch.any(durations < 0).item():
            raise ValueError(
                f"'durations' cannot contain negative values, got {durations.tolist()}."
            )
        if isinstance(values, torch.Tensor):
            dur_tensor = durations.to(device=values.device, dtype=torch.long)
            return torch.repeat_interleave(values, dur_tensor, dim=0)
        dur_np = durations.detach().cpu().to(dtype=torch.long).numpy()
        return np.repeat(values, dur_np, axis=0)

    if isinstance(durations, np.ndarray):
        if durations.ndim != 1:
            raise ValueError(
                f"'durations' array must be 1D, got {durations.ndim}D."
            )
        if durations.dtype == np.bool_ or np.iscomplexobj(durations):
            raise TypeError(
                f"'durations' array cannot have dtype {durations.dtype}."
            )
        if int(durations.shape[0]) != seq_len:
            raise ValueError(
                f"Length mismatch between 'values' ({seq_len}) and 'durations' ({int(durations.shape[0])})."
            )
        if np.any(durations < 0):
            raise ValueError(
                f"'durations' cannot contain negative values, got {durations.tolist()}."
            )
        dur_np = durations.astype(np.int64)
        if isinstance(values, torch.Tensor):
            dur_tensor = torch.from_numpy(dur_np).to(
                device=values.device, dtype=torch.long
            )
            return torch.repeat_interleave(values, dur_tensor, dim=0)
        return np.repeat(values, dur_np, axis=0)

    if isinstance(durations, Sequence) and not isinstance(durations, (str, bytes)):
        if len(durations) != seq_len:
            raise ValueError(
                f"Length mismatch between 'values' ({seq_len}) and 'durations' ({len(durations)})."
            )
        int_durs: list[int] = []
        for idx, d in enumerate(durations):
            if isinstance(d, bool) or not isinstance(d, (int, float, np.number)):
                raise TypeError(
                    f"Duration at index {idx} must be a non-negative number, got {d!r}."
                )
            if d < 0:
                raise ValueError(
                    f"Duration at index {idx} cannot be negative, got {d}."
                )
            int_durs.append(int(d))
        if isinstance(values, torch.Tensor):
            dur_tensor = torch.tensor(
                int_durs, device=values.device, dtype=torch.long
            )
            return torch.repeat_interleave(values, dur_tensor, dim=0)
        return np.repeat(values, np.asarray(int_durs, dtype=np.int64), axis=0)

    raise TypeError(
        f"'durations' must be a 1D Tensor, ndarray, or sequence of numbers, got {type(durations).__name__}."
    )


__all__ = ["pad", "pad_1D", "pad_2D", "expand"]

