# Refactoring Phase 1: Utilities Modularization Report

**Date**: November 4, 2026  
**Status**: Completed  
**Baseline Test Status**: **126 / 126 Passed** (100% pass rate in 5.808s)  
**Scope**: `common/utils/` and `tests/test_utils.py`  

---

## 1. Objective and Summary

Phase 1 of the codebase reorganization addressed the monolithic `common/utils/tools.py` file (**616 lines**). This file previously bundled device resolution, batch transfer, padding mask generation, 1D/2D array padding, and duration expansion into a single unspecialized module.

The utility functions have been decoupled into three dedicated, single-responsibility modules:
- [`common/utils/device.py`](file:///Users/swaruppal/FastSpeech2_Project/common/utils/device.py) (230 lines): Hardware device resolution and recursive batch transfer.
- [`common/utils/mask.py`](file:///Users/swaruppal/FastSpeech2_Project/common/utils/mask.py) (89 lines): FastSpeech 2 boolean padding mask generation.
- [`common/utils/pad.py`](file:///Users/swaruppal/FastSpeech2_Project/common/utils/pad.py) (317 lines): Sequence zero-padding (1D/2D) and phoneme-to-frame duration expansion.

[`common/utils/tools.py`](file:///Users/swaruppal/FastSpeech2_Project/common/utils/tools.py) was converted into a lightweight, 33-line backward-compatible facade re-exporting all symbols. No function signatures, contracts, or behaviors were changed.

---

## 2. Files Created and Modified

| File | Status | Lines | Primary Responsibility |
| :--- | :---: | :---: | :--- |
| `common/utils/device.py` | **Created** | 230 | `resolve_device`, `to_device`, `_to_long_tensor`, `_to_float_tensor`, `_is_fastspeech2_batch_tuple` |
| `common/utils/mask.py` | **Created** | 89 | `get_mask_from_lengths` |
| `common/utils/pad.py` | **Created** | 317 | `pad`, `pad_1D`, `pad_2D`, `expand` |
| `common/utils/tools.py` | **Modified** | 33 | Facade re-exporting all functions from `device`, `mask`, and `pad` |
| `common/utils/__init__.py` | **Modified** | 17 | Package exports directly referencing dedicated modules |
| `tests/test_utils.py` | **Modified** | 377 | Added `test_modular_import_identity_and_backward_compatibility` |

---

## 3. Function Migration Map

| Function | Old Location | New Canonical Location | Backward-Compatible Facade |
| :--- | :--- | :--- | :--- |
| `resolve_device` | `common/utils/tools.py` | `common/utils/device.py` | `common.utils.tools.resolve_device` |
| `to_device` | `common/utils/tools.py` | `common/utils/device.py` | `common.utils.tools.to_device` |
| `_to_long_tensor` | `common/utils/tools.py` | `common/utils/device.py` | `common.utils.tools._to_long_tensor` |
| `_to_float_tensor` | `common/utils/tools.py` | `common/utils/device.py` | `common.utils.tools._to_float_tensor` |
| `_is_fastspeech2_batch_tuple` | `common/utils/tools.py` | `common/utils/device.py` | `common.utils.tools._is_fastspeech2_batch_tuple` |
| `get_mask_from_lengths` | `common/utils/tools.py` | `common/utils/mask.py` | `common.utils.tools.get_mask_from_lengths` |
| `pad` | `common/utils/tools.py` | `common/utils/pad.py` | `common.utils.tools.pad` |
| `pad_1D` | `common/utils/tools.py` | `common/utils/pad.py` | `common.utils.tools.pad_1D` |
| `pad_2D` | `common/utils/tools.py` | `common/utils/pad.py` | `common.utils.tools.pad_2D` |
| `expand` | `common/utils/tools.py` | `common/utils/pad.py` | `common.utils.tools.expand` |

---

## 4. Import Syntax & Compatibility

Both legacy imports and new modular imports resolve to the exact same underlying function objects (`assert facade.fn is module.fn`):

### Device Management
```python
# Legacy / Facade Import (continues working seamlessly):
from common.utils.tools import resolve_device, to_device

# Preferred Modular Import:
from common.utils.device import resolve_device, to_device
```

### Mask Generation
```python
# Legacy / Facade Import:
from common.utils.tools import get_mask_from_lengths

# Preferred Modular Import:
from common.utils.mask import get_mask_from_lengths
```

### Padding & Expansion
```python
# Legacy / Facade Import:
from common.utils.tools import pad, pad_1D, pad_2D, expand

# Preferred Modular Import:
from common.utils.pad import pad, pad_1D, pad_2D, expand
```

### Top-Level Package Import
```python
# Unchanged top-level convenience imports:
from common.utils import (
    resolve_device,
    to_device,
    get_mask_from_lengths,
    pad,
    pad_1D,
    pad_2D,
    expand,
)
```

---

## 5. Verification & Test Execution Results

### A. Dedicated Utility Tests
Ran `tests/test_utils.py` including the new import identity verification test:
```bash
.venv/bin/python -m unittest tests/test_utils.py -v
```
```
test_expand_1d_and_2d_and_errors (test_utils.TestDeviceAndTensorUtils) ... ok
test_modular_import_identity_and_backward_compatibility (test_utils.TestDeviceAndTensorUtils) ... ok
test_mps_device_support_if_available (test_utils.TestDeviceAndTensorUtils) ... ok
test_pad_1d_and_2d_numpy (test_utils.TestDeviceAndTensorUtils) ... ok
test_pad_pytorch_tensors (test_utils.TestDeviceAndTensorUtils) ... ok
test_padding_mask_empty_and_invalid_raises (test_utils.TestDeviceAndTensorUtils) ... ok
test_padding_mask_single_token (test_utils.TestDeviceAndTensorUtils) ... ok
test_padding_mask_variable_length_and_true_is_pad_convention (test_utils.TestDeviceAndTensorUtils) ... ok
test_to_device_cpu_operation (test_utils.TestDeviceAndTensorUtils) ... ok
test_unavailable_and_invalid_devices_raise_errors (test_utils.TestDeviceAndTensorUtils) ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.285s
OK
```

### B. Full Test Suite Execution
Ran all 10 test suites across the repository:
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```
```
Ran 126 tests in 5.808s
OK
```

### Test Suite Breakdown:
- `tests/test_config.py`: 22 passed
- `tests/test_fastspeech2.py`: 11 passed
- `tests/test_fastspeech2_loss.py`: 12 passed
- `tests/test_models.py`: 13 passed
- `tests/test_postnet.py`: 8 passed
- `tests/test_training_engine.py`: 11 passed
- `tests/test_transformer.py`: 10 passed
- `tests/test_utils.py`: 10 passed (including modular import identity test)
- `tests/test_variance_adaptor.py`: 19 passed
- `tests/test_variance_components.py`: 13 passed

Total: **126 passed, 0 failed, 0 errors, 0 skipped**.

---

## 6. Compatibility & Regression Checks

1. **State Dictionary Compatibility**: FastSpeech 2 parameters and state dictionary keys are unaffected (confirmed via `test_fastspeech2.py` and `test_models.py`).
2. **Device Portability**: Execution on CPU and native Apple Silicon MPS verified (confirmed via `test_mps_device_support_if_available` and `test_09_device_compatibility_cpu_and_mps`).
3. **External Repository Integrity**: `original_fastspeech2/` checked with `git status` — completely untouched (0 modified, 0 untracked).
4. **No Code Duplication**: Implementations exist strictly within `device.py`, `mask.py`, and `pad.py`; `tools.py` contains only import statements and re-exports.

---

## 7. Next Steps

Phase 1 is complete and verified. As instructed, execution has stopped. Awaiting review and explicit approval before proceeding to **Phase 2: PostNet Decoupling from Transformer Layers**.

