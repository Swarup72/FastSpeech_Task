"""Shared FastSpeech 2 building blocks: Conv, VariancePredictor, and LengthRegulator.

Implementation based on `original_fastspeech2/model/modules.py` with fixes for:
- DEC-01: Module-global `device` removed; all dynamically allocated tensors (such as `mel_len`)
  inherit device from inputs (`x.device`), ensuring seamless CPU and Apple Silicon MPS operation.
- DEC-11: VariancePredictor uses `padding = (kernel - 1) // 2` in both `conv1d_1` and `conv1d_2`
  (replacing hardcoded `padding=1` in `conv1d_2`), preserving sequence length for arbitrary odd kernels.
- DEC-14: Module hierarchy and state_dict parameter names are 100% preserved.

This module acts as a backward-compatible facade re-exporting components from their
dedicated modules:
- Conv -> `common.model.conv`
- VariancePredictor -> `common.model.variance_predictor`
- LengthRegulator -> `common.model.length_regulator`
- VarianceAdaptor -> `common.model.variance_adaptor`
"""

from __future__ import annotations

from common.model.conv import Conv
from common.model.length_regulator import LengthRegulator
from common.model.variance_adaptor import VarianceAdaptor
from common.model.variance_predictor import VariancePredictor

__all__ = [
    "VarianceAdaptor",
    "Conv",
    "VariancePredictor",
    "LengthRegulator",
]
