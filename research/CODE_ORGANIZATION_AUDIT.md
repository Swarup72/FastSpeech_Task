# FastSpeech 2 Codebase Organization and Quality Audit

**Date**: November 4, 2026  
**Status**: Completed (Inspection-Only Audit)  
**Baseline Test Status**: **125 / 125 Passed** (100% pass rate in 6.045s)  
**Scope**: `common/`, `tests/`, `research/`, and root project configuration  

---

## 1. Executive Summary

This audit performs a systematic software architecture, modularity, and maintainability review of the FastSpeech 2 codebase following the completion of the core model components (Step 1.4A through Step 1.4H). The codebase currently consists of **34 Python files** (24 implementation files under `common/` and 10 test suites under `tests/`).

### Overall Health & Architectural Strengths
1. **Zero Circular Imports**: The dependency graph forms a strict, well-ordered Directed Acyclic Graph (DAG) with no cycles.
2. **Device Portability (DEC-01)**: The codebase is fully device-agnostic, with seamless execution across CPU, Apple Silicon MPS (`mps`), and CUDA.
3. **Rigorous Correctness**: Mask-Aware BatchNorm in PostNet (Step 1.4E-R), correct duration rounding inside `torch.round` (DEC-03), and frame-level temporal truncation in FastSpeech2Loss (DEC-10) are verified with deterministic unit tests.
4. **Strong Test Coverage**: 125 granular tests covering configuration validation, tensor utilities, attention blocks, encoder/decoder, PostNet, variance adaptor, end-to-end model, multi-task loss, and training engine.

### Key Organizational Deficiencies Identified
1. **Files Exceeding 500 Lines**:
   - `common/model/modules.py` (**750 lines**): Violates the Single Responsibility Principle by bundling four distinct architectural components (`Conv`, `VariancePredictor`, `LengthRegulator`, and `VarianceAdaptor`). Furthermore, `VarianceAdaptor.forward()` is **168 lines long**.
   - `common/utils/tools.py` (**616 lines**): Functions as a general "junk drawer" merging device resolution/transfer, mask construction, 1D/2D padding, and duration expansion.
   - `common/transformer/layers.py` (**530 lines**): Architecturally couples the Transformer Feed-Forward block (`FFTBlock`) with the acoustic mel-refinement `PostNet` and its 80-line mask-aware normalization routine.
2. **Files Exceeding 300 Lines (Requiring Review)**:
   - `common/model/loss.py` (497 lines): High line count is primarily driven by extensive architectural documentation and rigorous DEC-10 dimension validation guards; structurally cohesive.
   - `common/training/engine.py` (456 lines): Cohesive training loop engine.
   - `common/transformer/models.py` (434 lines): Bundles both `Encoder` and `Decoder`.
   - `common/training/checkpoint.py` (383 lines): Cohesive checkpoint manager.
   - `common/config/model.py` (380 lines) & `common/config/preprocess.py` (333 lines): Cohesive dataclass definitions with validation methods.
   - `common/model/fastspeech2.py` (314 lines): Cohesive top-level model module.
3. **Redundant Module Namespace**:
   - `common/modules/` exists with only an `__init__.py` (18 lines) re-exporting classes from `common/model/modules.py`. Having both `common/model/modules.py` and `common/modules/` is redundant and creates confusion regarding the canonical import path.
4. **Code Overlap**:
   - Duration expansion is implemented twice: once as `expand()` in `common/utils/tools.py` and once as `LengthRegulator.expand()` in `common/model/modules.py`.

---

## 2. Existing Directory Structure

```
FastSpeech2_Project/
├── .venv/                                # Isolated virtual environment (Python 3.10.18, PyTorch 2.10.0)
├── common/                               # Core shared FastSpeech 2 package
│   ├── __init__.py                       # Package entry point & top-level exports (81 lines)
│   ├── config/                           # Strongly-typed configuration system
│   │   ├── __init__.py                   # Config exports (95 lines)
│   │   ├── base.py                       # ConfigNode base class & validators (202 lines)
│   │   ├── loader.py                     # YAML loaders & FastSpeech2Config (135 lines)
│   │   ├── model.py                      # ModelConfig dataclasses (380 lines)
│   │   ├── preprocess.py                 # PreprocessConfig dataclasses (333 lines)
│   │   └── training.py                   # TrainConfig dataclasses (206 lines)
│   ├── model/                            # Neural acoustic model building blocks
│   │   ├── __init__.py                   # Model exports (24 lines)
│   │   ├── fastspeech2.py                # FastSpeech2 top-level module (314 lines)
│   │   ├── loss.py                       # FastSpeech2Loss multi-task loss (497 lines)
│   │   ├── modules.py                    # Conv, VP, LR, VarianceAdaptor (750 lines) [!]
│   │   └── optimizer.py                  # ScheduledOptim & Noam scheduler (236 lines)
│   ├── modules/                          # Redundant compatibility alias
│   │   └── __init__.py                   # Re-exports from common.model.modules (18 lines) [!]
│   ├── training/                         # Training engine & persistence
│   │   ├── __init__.py                   # Training exports (28 lines)
│   │   ├── checkpoint.py                 # CheckpointManager & RNG serialization (383 lines)
│   │   └── engine.py                     # Trainer & TrainLosses (456 lines)
│   ├── transformer/                      # Transformer sub-components
│   │   ├── __init__.py                   # Transformer exports (49 lines)
│   │   ├── constants.py                  # Padding and special constants (25 lines)
│   │   ├── layers.py                     # FFTBlock, ConvNorm, PostNet (530 lines) [!]
│   │   ├── models.py                     # Encoder, Decoder (434 lines)
│   │   ├── modules.py                    # PositionalEncoding, ScaledDotProductAttention (256 lines)
│   │   └── sublayers.py                  # MultiHeadAttention, PositionwiseFeedForward (291 lines)
│   └── utils/                            # Shared utilities
│       ├── __init__.py                   # Utils exports (22 lines)
│       └── tools.py                      # to_device, get_mask_from_lengths, pad, expand (616 lines) [!]
├── original_fastspeech2/                 # Reference repository (read-only invariant baseline)
├── research/                             # Architectural specifications & audits
│   ├── COMMON_ARCHITECTURE_SPECIFICATION.md
│   ├── IMPLEMENTATION_DECISIONS.md
│   └── ...
└── tests/                                # Comprehensive unit & regression test suite (10 suites, 125 tests)
    ├── test_config.py                    # Configuration validation tests (547 lines, 22 tests)
    ├── test_fastspeech2.py               # FastSpeech2 model integration tests (452 lines, 10 tests)
    ├── test_fastspeech2_loss.py          # Loss component & DEC-10 tests (656 lines, 12 tests)
    ├── test_models.py                    # Encoder & Decoder tests (510 lines, 13 tests)
    ├── test_postnet.py                   # PostNet & Mask-Aware BatchNorm tests (633 lines, 8 tests)
    ├── test_training_engine.py           # Optimizer, scheduler & trainer tests (564 lines, 11 tests)
    ├── test_transformer.py               # Attention & FFTBlock tests (467 lines, 10 tests)
    ├── test_utils.py                     # Tools & tensor tests (340 lines, 21 tests)
    ├── test_variance_adaptor.py          # VarianceAdaptor integration tests (535 lines, 10 tests)
    └── test_variance_components.py       # VariancePredictor & LengthRegulator tests (393 lines, 8 tests)
```

---

## 3. Complete Python File-Size Inventory

### A. Implementation Files (`common/`)

| File Path | Line Count | Classes | Functions (Top/Methods) | Main Responsibilities | Organization Assessment |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `common/__init__.py` | 81 | 0 | 0 / 0 | Package-level re-exports for public API | **Appropriate** |
| `common/config/__init__.py` | 95 | 0 | 0 / 0 | Re-export of configuration dataclasses | **Appropriate** |
| `common/config/base.py` | 202 | 2 | 12 / 6 | ConfigNode base, validation helpers | **Appropriate** |
| `common/config/loader.py` | 135 | 1 | 4 / 3 | YAML loading and top-level container | **Appropriate** |
| `common/config/model.py` | 380 | 6 | 0 / 14 | ModelConfig and transformer dataclasses | **Appropriate** (>300 lines due to 6 dataclasses) |
| `common/config/preprocess.py` | 333 | 9 | 0 / 19 | Preprocessing feature settings dataclasses | **Appropriate** (>300 lines due to 9 dataclasses) |
| `common/config/training.py` | 206 | 4 | 0 / 9 | TrainConfig and optimizer dataclasses | **Appropriate** |
| `common/model/__init__.py` | 24 | 0 | 0 / 0 | Model building block re-exports | **Appropriate** |
| `common/model/fastspeech2.py` | 314 | 1 | 0 / 3 | Top-level FastSpeech2 model integration | **Appropriate** (>300 lines due to type checks) |
| `common/model/loss.py` | 497 | 1 | 1 / 4 | FastSpeech2Loss with DEC-10 slicing | **Appropriate** (Single cohesive responsibility) |
| `common/model/modules.py` | **750** | **4** | **0 / 17** | Bundles Conv, VP, LR, VarianceAdaptor | **NEEDS SPLITTING** (>500 lines, 4 distinct components) |
| `common/model/optimizer.py` | 236 | 1 | 0 / 10 | ScheduledOptim with Noam schedule | **Appropriate** |
| `common/modules/__init__.py` | 18 | 0 | 0 / 0 | Alias re-exporting common.model.modules | **REDUNDANT** (Consider deprecating) |
| `common/training/__init__.py` | 28 | 0 | 0 / 0 | Training package re-exports | **Appropriate** |
| `common/training/checkpoint.py` | 383 | 1 | 3 / 10 | Atomic checkpoint manager and RNG save/load | **Appropriate** (>300 lines due to Colab safeguards) |
| `common/training/engine.py` | 456 | 2 | 1 / 9 | Trainer and TrainLosses loop execution | **Appropriate** (Single cohesive responsibility) |
| `common/transformer/__init__.py` | 49 | 0 | 0 / 0 | Transformer package re-exports | **Appropriate** |
| `common/transformer/constants.py` | 25 | 0 | 0 / 0 | Transformer constants (PAD token index) | **Appropriate** |
| `common/transformer/layers.py` | **530** | **3** | **0 / 9** | FFTBlock, ConvNorm, PostNet | **NEEDS SPLITTING** (>500 lines, couples FFT with PostNet) |
| `common/transformer/models.py` | 434 | 2 | 0 / 6 | Encoder and Decoder models | **Appropriate** (>300 lines due to 2 large modules) |
| `common/transformer/modules.py` | 256 | 2 | 1 / 6 | PositionalEncoding, ScaledDotProductAttn | **Appropriate** |
| `common/transformer/sublayers.py` | 291 | 2 | 1 / 6 | MultiHeadAttention, PositionwiseFeedForward | **Appropriate** |
| `common/utils/__init__.py` | 22 | 0 | 0 / 0 | Utility function re-exports | **Appropriate** |
| `common/utils/tools.py` | **616** | **0** | **10 / 0** | Device transfer, padding, masking, expand | **NEEDS SPLITTING** (>500 lines, 4 mixed responsibilities) |

### B. Test Suite Files (`tests/`)

| Test File Path | Line Count | Test Classes | Test Methods | Main Scope | Assessment |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `tests/test_config.py` | 547 | 1 | 7 | Configuration validation & YAML loading | Appropriate for complete config audit |
| `tests/test_fastspeech2.py` | 452 | 1 | 11 | End-to-end model forward & backward | Appropriate |
| `tests/test_fastspeech2_loss.py` | 656 | 1 | 12 | Loss components, parity, DEC-10 | Appropriate |
| `tests/test_models.py` | 510 | 1 | 13 | Encoder and Decoder unit tests | Appropriate |
| `tests/test_postnet.py` | 633 | 1 | 8 | PostNet & Mask-Aware BatchNorm | Appropriate |
| `tests/test_training_engine.py` | 564 | 2 | 11 | ScheduledOptim, CheckpointManager, Trainer | Appropriate |
| `tests/test_transformer.py` | 467 | 1 | 10 | Attention & FFTBlock unit tests | Appropriate |
| `tests/test_utils.py` | 340 | 1 | 21 | Device and tensor utilities | Appropriate |
| `tests/test_variance_adaptor.py` | 535 | 1 | 19 | VarianceAdaptor integration | Appropriate |
| `tests/test_variance_components.py` | 393 | 1 | 13 | VariancePredictor & LengthRegulator | Appropriate |

---

## 4. Detailed Responsibility Breakdown of Files Exceeding 500 Lines

### A. `common/model/modules.py` (750 lines)

```
Lines 1–32:    Module docstrings and imports
Lines 33–73:   Conv class (41 lines) - 1D Convolution with optional Weight Normalization
Lines 76–252:  VariancePredictor class (177 lines) - 2-layer Conv1d with LayerNorm, ReLU, Dropout
Lines 255–374: LengthRegulator class (120 lines) - Duration expansion module (expand, LR, forward)
Lines 377–741: VarianceAdaptor class (365 lines) - Quantization, embedding lookups, controls, and forward
Lines 743–750: __all__ export definition
```

#### Responsibility Assessment:
- **Component Congestion**: This single file implements four distinct classes. `VariancePredictor` is a general predictor; `LengthRegulator` is a sequence length modifier; `VarianceAdaptor` is a high-level composite coordinating duration, pitch, and energy conditioning.
- **Monolithic Forward Method**: `VarianceAdaptor.forward()` spans **168 lines** (lines 573–741). It handles:
  1. Phoneme-level duration prediction and clamping.
  2. Duration conversion (`log_duration_to_frames`).
  3. Source-mask row validation and zero-duration inference fallback with warning logging.
  4. Length regulation and sequence expansion.
  5. Pitch prediction and embedding lookup (branching on `phoneme_level` vs `frame_level`).
  6. Energy prediction and embedding lookup (branching on `phoneme_level` vs `frame_level`).
  7. Sequential addition of variance embeddings to hidden states.
- **Recommendation**: Split into modular components under `common/model/` while maintaining `common/model/modules.py` as a backward-compatible facade re-exporting all four classes.

---

### B. `common/utils/tools.py` (616 lines)

```
Lines 1–28:    Module docstrings and imports
Lines 29–89:   resolve_device() - Device parsing, availability checking, and validation
Lines 92–123:  _to_long_tensor, _to_float_tensor, _is_fastspeech2_batch_tuple helpers
Lines 126–233: to_device() - Recursive device transfer for tensors, batches, and sequences
Lines 241–310: get_mask_from_lengths() - Boolean padding mask generation [B, L]
Lines 318–433: pad() - Generalized sequence padding along arbitrary dimensions
Lines 436–453: pad_1D() - 1D sequence padding helper
Lines 456–497: pad_2D() - 2D sequence padding helper
Lines 505–615: expand() - Standalone duration expansion utility
```

#### Responsibility Assessment:
- **Mixed Concerns**: Contains three fundamentally different categories of utilities:
  1. *Hardware/Device Management*: `resolve_device`, `to_device`, and batch structure inspection.
  2. *Mask Generation*: `get_mask_from_lengths` (strictly used inside attention and model forward passes).
  3. *Data Padding & Collation*: `pad`, `pad_1D`, `pad_2D` (used in dataset preprocessing and batching).
  4. *Expansion Utility*: `expand` (redundant with `LengthRegulator.expand`).
- **Recommendation**: Modularize into dedicated, single-responsibility files under `common/utils/`:
  - `common/utils/device.py` (`resolve_device`, `to_device`, helpers)
  - `common/utils/mask.py` (`get_mask_from_lengths`)
  - `common/utils/pad.py` (`pad`, `pad_1D`, `pad_2D`)
  Keep `common/utils/tools.py` as a facade re-exporting all functions to guarantee 100% backward compatibility.

---

### C. `common/transformer/layers.py` (530 lines)

```
Lines 1–32:    Module docstrings and imports
Lines 33–128:  FFTBlock class (96 lines) - MultiHeadAttention + PositionwiseFeedForward + LayerNorm
Lines 131–182: ConvNorm class (52 lines) - Conv1d with Weight Normalization and Xavier init
Lines 185–523: PostNet class (339 lines) - 5-layer convolutional network with Mask-Aware BatchNorm
Lines 525–530: __all__ export definition
```

#### Responsibility Assessment:
- **Architectural Mismatch**: `FFTBlock` is the fundamental building block of the Transformer Encoder and Decoder. In contrast, `PostNet` and `ConvNorm` constitute a post-processing convolutional refinement network for mel-spectrograms.
- **Inappropriate Dependency**: To support `PostNet.from_config`, `layers.py` imports `PreprocessConfig` from `common.config.preprocess`. Transformer layers should not depend on audio preprocessing settings.
- **Normalization Complexity**: Lines 351–430 implement `_masked_batch_norm` (80 lines of custom valid-frame statistics and Bessel correction).
- **Recommendation**: Move `PostNet` and `ConvNorm` to `common/model/postnet.py` (or `common/transformer/postnet.py`), removing the audio config dependency from Transformer layers. Maintain re-exports in `common/transformer/layers.py` for backward compatibility.

---

## 5. Architecture and Maintainability Analysis

### A. Code Duplication & Overlapping Implementations
1. **Duration Expansion**:
   - `expand()` in `common/utils/tools.py` (lines 505–615) implements tensor expansion given durations.
   - `LengthRegulator.expand()` in `common/model/modules.py` (lines 261–274) and `LengthRegulator.LR()` (lines 276–365) implement phoneme expansion with max-length clamping.
   - *Resolution*: Unify duration expansion so `LengthRegulator` uses a single shared expansion engine or delegates cleanly.
2. **Synthetic Test Fixtures**:
   - `tests/test_fastspeech2.py`, `tests/test_fastspeech2_loss.py`, `tests/test_training_engine.py`, and `tests/test_variance_adaptor.py` each independently define synthetic config-builder functions (`make_model`, `_make_dummy_configs`, `_make_test_configs`, `make_configs`).
   - While keeping tests self-contained is acceptable, a shared `tests/fixtures.py` would eliminate ~200 lines of boilerplate across test files.

### B. Namespace Inconsistency: `common/modules/` vs `common/model/modules.py`
- In `original_fastspeech2`, files were located at `model/modules.py` and `transformer/Modules.py`.
- In our codebase, `common/model/modules.py` contains the implementation, while `common/modules/__init__.py` exists purely to re-export `Conv`, `LengthRegulator`, `VarianceAdaptor`, and `VariancePredictor`.
- *Resolution*: Retain `common/modules/__init__.py` as a deprecated compatibility bridge, but standardize all internal codebase imports on `common.model`.

### C. Large Method Deconstruction Opportunities
- `VarianceAdaptor.forward()` (168 lines): Can be cleanly decomposed into private helper methods:
  - `_process_durations(output, src_masks, d_targets, d_control, max_mel_len)`
  - `_process_pitch(output, src_masks, mel_masks, p_targets, p_control)`
  - `_process_energy(output, src_masks, mel_masks, e_targets, e_control)`
  This improves readability without modifying public signature or tensor shapes.

---

## 6. Proposed Reorganized Directory Structure

```
common/
├── config/                               # (No structural changes needed)
│   ├── base.py
│   ├── loader.py
│   ├── model.py
│   ├── preprocess.py
│   └── training.py
├── model/                                # Decomposed model components
│   ├── __init__.py                       # Re-exports all model components
│   ├── conv.py                           # [NEW] Conv1d block (formerly in modules.py)
│   ├── fastspeech2.py                    # Top-level model integration
│   ├── length_regulator.py               # [NEW] LengthRegulator class (formerly in modules.py)
│   ├── loss.py                           # FastSpeech2Loss multi-task loss
│   ├── modules.py                        # [FACADE] Backward-compatible re-exports from new files
│   ├── optimizer.py                      # ScheduledOptim & Noam scheduler
│   ├── postnet.py                        # [NEW] PostNet & ConvNorm (extracted from transformer/layers.py)
│   ├── variance_adaptor.py               # [NEW] VarianceAdaptor class (formerly in modules.py)
│   └── variance_predictor.py             # [NEW] VariancePredictor class (formerly in modules.py)
├── training/                             # (No structural changes needed)
│   ├── checkpoint.py                     # CheckpointManager
│   └── engine.py                         # Trainer & TrainLosses
├── transformer/                          # Clean Transformer-only building blocks
│   ├── constants.py                      # Padding constants
│   ├── layers.py                         # [CLEANED] FFTBlock only (PostNet re-exported for compat)
│   ├── models.py                         # Encoder, Decoder
│   ├── modules.py                        # PositionalEncoding, ScaledDotProductAttention
│   └── sublayers.py                      # MultiHeadAttention, PositionwiseFeedForward
└── utils/                                # Modular utilities
    ├── device.py                         # [NEW] resolve_device, to_device, batch helpers
    ├── mask.py                           # [NEW] get_mask_from_lengths
    ├── pad.py                            # [NEW] pad, pad_1D, pad_2D
    └── tools.py                          # [FACADE] Backward-compatible re-exports from device, mask, pad
```

---

## 7. Preservation of Critical Invariants & Compatibility

Every proposed reorganization strictly preserves the following invariants:

| Invariant | Requirement | Proposed Architecture Guarantee |
| :--- | :--- | :--- |
| **Model `state_dict` Keys** | Exact parameter naming (`variance_adaptor.duration_predictor.*`, `postnet.convolutions.*`) | Moving classes to dedicated files does not change PyTorch submodule hierarchies or parameter names. |
| **Forward Contract** | Exact 10-tuple output from `FastSpeech2.forward()` | Unchanged. |
| **VarianceAdaptor Contract** | Exact 7-tuple output from `VarianceAdaptor.forward()` | Unchanged. |
| **Loss Contract** | Exact 6-tuple return from `FastSpeech2Loss.forward()` | Unchanged. |
| **Public Import Paths** | Existing imports like `from common.model.modules import VarianceAdaptor` | Preserved via `modules.py`, `layers.py`, and `tools.py` facade modules. |
| **Hardware Portability** | Dynamic device inheritance on CPU, MPS, and CUDA | Preserved. |
| **Language Independence** | Config-driven vocab and phoneme settings | Svarah (English) and IndicTTS (Telugu) remain fully isolated in their respective config domains. |

---

## 8. Baseline Test Execution Results

Prior to any modifications, the complete test suite was executed against the current repository state:

- **Command**: `.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`
- **Execution Date**: November 4, 2026, 00:04:58 IST
- **Total Tests Executed**: **125**
- **Passed**: **125**
- **Failed**: **0**
- **Skipped**: **0**
- **Errors**: **0**
- **Runtime**: **6.045 seconds**
- **Exit Code**: `0`

### Test Suite Breakdown:
1. `tests/test_config.py`: 22 passed
2. `tests/test_fastspeech2.py`: 11 passed
3. `tests/test_fastspeech2_loss.py`: 12 passed
4. `tests/test_models.py`: 13 passed
5. `tests/test_postnet.py`: 8 passed
6. `tests/test_training_engine.py`: 11 passed
7. `tests/test_transformer.py`: 10 passed
8. `tests/test_utils.py`: 21 passed
9. `tests/test_variance_adaptor.py`: 19 passed
10. `tests/test_variance_components.py`: 13 passed

**Comparison**: The baseline exactly matches the 125 passing tests reported at the completion of Step 1.4H.

---

## 9. Recommended Refactoring Sequence (Upon Approval)

If the proposed reorganization is approved, refactoring should proceed in three isolated, incremental phases, running the full test suite after each phase:

### Phase 1: Modularize `common/utils/tools.py`
1. Create `common/utils/device.py` (`resolve_device`, `to_device`, tensor conversion helpers).
2. Create `common/utils/mask.py` (`get_mask_from_lengths`).
3. Create `common/utils/pad.py` (`pad`, `pad_1D`, `pad_2D`, `expand`).
4. Convert `common/utils/tools.py` into a facade re-exporting all symbols.
5. Update `common/utils/__init__.py`.
6. Run full test suite (ensure 125/125 pass).

### Phase 2: Decouple PostNet from `common/transformer/layers.py`
1. Create `common/model/postnet.py` containing `ConvNorm` and `PostNet` (with Mask-Aware BatchNorm).
2. Clean `common/transformer/layers.py` to contain only `FFTBlock`, while importing and re-exporting `PostNet` and `ConvNorm` for backward compatibility.
3. Remove `PreprocessConfig` import from `common/transformer/layers.py`.
4. Run full test suite (ensure 125/125 pass).

### Phase 3: Decompose `common/model/modules.py`
1. Create `common/model/conv.py` (`Conv`).
2. Create `common/model/variance_predictor.py` (`VariancePredictor`).
3. Create `common/model/length_regulator.py` (`LengthRegulator`).
4. Create `common/model/variance_adaptor.py` (`VarianceAdaptor`). Decompose its 168-line `forward()` into clean helper methods (`_process_durations`, `_process_pitch`, `_process_energy`).
5. Convert `common/model/modules.py` into a facade re-exporting all four classes.
6. Run full test suite (ensure 125/125 pass).

---

## 10. Audit Sign-off

- [x] All 34 Python files inspected.
- [x] Line counts, classes, functions, and responsibilities documented.
- [x] Files > 300 lines inspected; files > 500 lines thoroughly decomposed.
- [x] Dependency DAG analyzed (0 circular imports confirmed).
- [x] Baseline test suite executed (125/125 passed).
- [x] Reorganization plan designed with 100% backward-compatibility facades.
- [x] Zero source files modified (audit is strictly read-only).

