"""Backward-compatible facade for FastSpeech 2 utilities.

This module preserves 100% backward compatibility for imports from
``common.utils.tools`` by re-exporting functions from their modularized homes:
- Device resolution & batch transfer: :mod:`common.utils.device`
- Padding mask generation: :mod:`common.utils.mask`
- Variable-length sequence padding & expansion: :mod:`common.utils.pad`
"""

from __future__ import annotations

from common.utils.device import (
    _is_fastspeech2_batch_tuple,
    _to_float_tensor,
    _to_long_tensor,
    resolve_device,
    to_device,
)
from common.utils.mask import get_mask_from_lengths
from common.utils.pad import expand, pad, pad_1D, pad_2D

__all__ = [
    "resolve_device",
    "to_device",
    "get_mask_from_lengths",
    "pad",
    "pad_1D",
    "pad_2D",
    "expand",
    "_to_long_tensor",
    "_to_float_tensor",
    "_is_fastspeech2_batch_tuple",
]
