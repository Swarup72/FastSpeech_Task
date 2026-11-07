"""Device resolution and batch/tensor device transfer utilities for FastSpeech 2.

Implements the shared utility specifications in:
- research/COMMON_ARCHITECTURE_SPECIFICATION.md (Part 1, Part 4 B-01, Part 7.13)
- research/IMPLEMENTATION_DECISIONS.md (DEC-01, DEC-08, DEC-13)

Key guarantees:
- No module-global `device` variable or hardcoded `.cuda()` calls (DEC-01).
- No top-level `matplotlib` or `scipy` imports (DEC-13).
- Explicit device availability checks in `resolve_device` / `to_device` (never silently falls back).
- Supports CPU, CUDA, and Apple Silicon MPS.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def resolve_device(device: str | torch.device) -> torch.device:
    """Validate and resolve an explicit PyTorch target device without silent fallback.

    Args:
        device: Target device specification (e.g., `"cpu"`, `"mps"`, `"mps:0"`, `"cuda"`,
            `"cuda:0"`, or a :class:`torch.device` instance).

    Returns:
        Resolved :class:`torch.device`.

    Raises:
        TypeError: If ``device`` is not a ``str`` or :class:`torch.device`.
        ValueError: If ``device`` is an empty or malformed device string.
        RuntimeError: If the explicitly requested device is unavailable on the current host.
    """
    if isinstance(device, str):
        if not device.strip():
            raise ValueError("Device string cannot be empty.")
        try:
            target = torch.device(device)
        except RuntimeError as exc:
            raise ValueError(f"Invalid device specification '{device}': {exc}") from exc
    elif isinstance(device, torch.device):
        target = device
    else:
        raise TypeError(
            f"Expected 'device' to be a str or torch.device, got {type(device).__name__}."
        )

    if target.type == "cpu":
        return target

    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device '{target}' is not available on this system."
            )
        if target.index is not None:
            count = torch.cuda.device_count()
            if target.index < 0 or target.index >= count:
                raise RuntimeError(
                    f"Requested CUDA device index {target.index} is out of range "
                    f"(available CUDA devices: {count})."
                )
        return target

    if target.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                f"Requested MPS device '{target}' is not available on this system."
            )
        if target.index is not None and target.index != 0:
            raise RuntimeError(
                f"Requested MPS device index {target.index} is invalid "
                f"(Apple Silicon MPS supports 'mps' or 'mps:0')."
            )
        return target

    raise RuntimeError(
        f"Requested device '{target}' (type='{target.type}') is not supported or available."
    )


def _to_long_tensor(value: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device=device, dtype=torch.long)
    raise TypeError(
        f"Expected torch.Tensor or np.ndarray for integer batch field, got {type(value).__name__}."
    )


def _to_float_tensor(value: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    if isinstance(value, np.ndarray):
        # Convert float64 NumPy arrays to float32 before moving to device (required for MPS compatibility)
        arr = value.astype(np.float32, copy=False) if value.dtype != np.float32 else value
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    raise TypeError(
        f"Expected torch.Tensor or np.ndarray for float batch field, got {type(value).__name__}."
    )


def _is_fastspeech2_batch_tuple(data: Sequence[Any], expected_len: int) -> bool:
    if len(data) != expected_len:
        return False
    # In FastSpeech 2 batches, index 0 is `ids` (list/tuple of str) and index 1 is `raw_texts` (list/tuple of str),
    # while index 2 (`speakers`) is a Tensor, ndarray, or None (in single-speaker mode).
    return (
        isinstance(data[0], (list, tuple))
        and isinstance(data[1], (list, tuple))
        and (data[2] is None or isinstance(data[2], (torch.Tensor, np.ndarray)))
    )


def to_device(data: Any, device: str | torch.device) -> Any:
    """Move a tensor, NumPy array, or FastSpeech 2 batch tuple to ``device``.

    Supports:
    - Single :class:`torch.Tensor`: moved to ``device`` while preserving ``dtype`` and values.
    - Single :class:`numpy.ndarray`: converted to a :class:`torch.Tensor` on ``device`` preserving ``dtype``.
    - 12-element FastSpeech 2 training/validation batch tuple:
      ``(ids, raw_texts, speakers, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len, pitches, energies, durations)``
    - 6-element FastSpeech 2 inference batch tuple:
      ``(ids, raw_texts, speakers, texts, src_lens, max_src_len)``
    - Non-empty sequence (``tuple`` / ``list``) of :class:`torch.Tensor` or :class:`numpy.ndarray` objects.

    Raises:
        TypeError: If ``data`` or ``device`` has an invalid type.
        ValueError: If ``data`` is an empty sequence/array or has an unsupported batch structure.
        RuntimeError: If the requested ``device`` is unavailable.
    """
    target_device = resolve_device(device)

    if isinstance(data, torch.Tensor):
        return data.to(device=target_device)

    if isinstance(data, np.ndarray):
        if data.size == 0:
            raise ValueError("Cannot move an empty NumPy array to device.")
        return torch.from_numpy(data).to(device=target_device)

    if isinstance(data, (tuple, list)):
        if len(data) == 0:
            raise ValueError("Cannot move an empty sequence to device.")

        if _is_fastspeech2_batch_tuple(data, 12):
            (
                ids,
                raw_texts,
                speakers,
                texts,
                src_lens,
                max_src_len,
                mels,
                mel_lens,
                max_mel_len,
                pitches,
                energies,
                durations,
            ) = data

            speakers_t = (
                _to_long_tensor(speakers, target_device)
                if speakers is not None
                else None
            )
            texts_t = _to_long_tensor(texts, target_device)
            src_lens_t = _to_long_tensor(src_lens, target_device)
            mels_t = _to_float_tensor(mels, target_device)
            mel_lens_t = _to_long_tensor(mel_lens, target_device)
            pitches_t = _to_float_tensor(pitches, target_device)
            energies_t = _to_float_tensor(energies, target_device)
            durations_t = _to_long_tensor(durations, target_device)

            return (
                ids,
                raw_texts,
                speakers_t,
                texts_t,
                src_lens_t,
                int(max_src_len),
                mels_t,
                mel_lens_t,
                int(max_mel_len),
                pitches_t,
                energies_t,
                durations_t,
            )

        if _is_fastspeech2_batch_tuple(data, 6):
            (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

            speakers_t = (
                _to_long_tensor(speakers, target_device)
                if speakers is not None
                else None
            )
            texts_t = _to_long_tensor(texts, target_device)
            src_lens_t = _to_long_tensor(src_lens, target_device)

            return (
                ids,
                raw_texts,
                speakers_t,
                texts_t,
                src_lens_t,
                int(max_src_len),
            )

        if all(isinstance(elem, (torch.Tensor, np.ndarray)) for elem in data):
            moved = [to_device(elem, target_device) for elem in data]
            return tuple(moved) if isinstance(data, tuple) else moved

        raise ValueError(
            f"Unsupported batch structure of length {len(data)} passed to to_device(). "
            f"Expected a Tensor, a sequence of Tensors, a 6-element inference batch, "
            f"or a 12-element training batch."
        )

    raise TypeError(
        f"Unsupported data type '{type(data).__name__}' passed to to_device()."
    )


__all__ = ["resolve_device", "to_device"]

