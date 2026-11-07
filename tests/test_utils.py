"""Unit tests for shared device and tensor utilities (`common.utils.tools`)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from common.utils import (
    expand,
    get_mask_from_lengths,
    pad,
    pad_1D,
    pad_2D,
    resolve_device,
    to_device,
)


class TestDeviceAndTensorUtils(unittest.TestCase):
    """Tests for `to_device`, `resolve_device`, `get_mask_from_lengths`, `pad`, and `expand`."""

    # ------------------------------------------------------------------
    # 1, 2, 3. Padding Masks: Variable-Length, True=PAD, Single-Token
    # ------------------------------------------------------------------

    def test_padding_mask_variable_length_and_true_is_pad_convention(self) -> None:
        """Verify variable-length masks follow `True = PAD, False = valid`."""
        lengths = torch.tensor([5, 3, 1], dtype=torch.long)
        mask = get_mask_from_lengths(lengths)

        self.assertEqual(mask.shape, (3, 5))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.device.type, "cpu")

        expected = torch.tensor(
            [
                [False, False, False, False, False],  # len 5: 0 padded
                [False, False, False, True, True],    # len 3: 2 padded
                [False, True, True, True, True],      # len 1: 4 padded
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask, expected))

        # Explicit max_len larger than max(lengths)
        mask_padded = get_mask_from_lengths(lengths, max_len=7)
        self.assertEqual(mask_padded.shape, (3, 7))
        self.assertTrue(torch.all(mask_padded[:, 5:]))
        self.assertTrue(torch.equal(mask_padded[:, :5], expected))

    def test_padding_mask_single_token_sequence(self) -> None:
        """Verify single-token sequence (`lengths=[1]`) produces `[[False]]`."""
        lengths = torch.tensor([1], dtype=torch.int64)
        mask = get_mask_from_lengths(lengths)

        self.assertEqual(mask.shape, (1, 1))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(mask[0, 0].item())

    # ------------------------------------------------------------------
    # 4, 5, 6. Sequence Padding (`pad`): Shapes, Values, Dtype & Device
    # ------------------------------------------------------------------

    def test_pad_shapes_values_and_dtype_preservation_cpu(self) -> None:
        """Verify 1D and 2D tensor padding shapes, zero-fill values, and dtype preservation."""
        for dtype in (torch.float32, torch.float16, torch.int64, torch.int32, torch.bool):
            t1 = torch.ones((2, 4), dtype=dtype)
            t2 = torch.ones((5, 4), dtype=dtype)
            t3 = torch.zeros((0, 4), dtype=dtype)

            padded_2d = pad([t1, t2, t3])
            self.assertEqual(padded_2d.shape, (3, 5, 4))
            self.assertEqual(padded_2d.dtype, dtype)
            self.assertEqual(padded_2d.device.type, "cpu")

            # Valid positions preserved, padded positions strictly 0 / False
            self.assertTrue(torch.equal(padded_2d[0, :2], t1))
            self.assertTrue(torch.all(padded_2d[0, 2:] == 0))
            self.assertTrue(torch.equal(padded_2d[1], t2))
            self.assertTrue(torch.all(padded_2d[2] == 0))

            # 1D tensors with explicit mel_max_length
            v1 = torch.ones((3,), dtype=dtype)
            v2 = torch.ones((1,), dtype=dtype)
            padded_1d = pad([v1, v2], mel_max_length=6)
            self.assertEqual(padded_1d.shape, (2, 6))
            self.assertEqual(padded_1d.dtype, dtype)
            self.assertTrue(torch.all(padded_1d[0, :3] == 1))
            self.assertTrue(torch.all(padded_1d[0, 3:] == 0))
            self.assertTrue(torch.all(padded_1d[1, :1] == 1))
            self.assertTrue(torch.all(padded_1d[1, 1:] == 0))

        # Also verify NumPy pad_1D and pad_2D helpers
        arr_1d = pad_1D([np.array([1, 2]), np.array([3, 4, 5])], PAD=0)
        self.assertEqual(arr_1d.shape, (2, 3))
        self.assertEqual(arr_1d[0, 2], 0)

        arr_2d = pad_2D([np.ones((2, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32)], maxlen=5)
        self.assertEqual(arr_2d.shape, (2, 5, 3))
        self.assertTrue(np.all(arr_2d[0, 2:] == 0.0))

    # ------------------------------------------------------------------
    # 7. Expansion Behaviour (`expand`)
    # ------------------------------------------------------------------

    def test_expand_preserves_order_dtype_and_device(self) -> None:
        """Verify `expand` repeats elements in order, drops 0-duration tokens, and preserves dtype/device."""
        # 1D Tensor expansion with zero-duration token
        vals_1d = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
        durs = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        expanded_1d = expand(vals_1d, durs)
        self.assertIsInstance(expanded_1d, torch.Tensor)
        assert isinstance(expanded_1d, torch.Tensor)
        self.assertEqual(expanded_1d.dtype, torch.float32)
        self.assertEqual(expanded_1d.device.type, "cpu")
        self.assertEqual(expanded_1d.tolist(), [10.0, 10.0, 30.0, 30.0, 30.0, 40.0])

        # 2D Tensor expansion (e.g. hidden states [L_src, H])
        vals_2d = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
        expanded_2d = expand(vals_2d, [1, 2, 0])
        assert isinstance(expanded_2d, torch.Tensor)
        self.assertEqual(expanded_2d.shape, (3, 2))
        self.assertEqual(expanded_2d.dtype, torch.int64)
        self.assertEqual(expanded_2d.tolist(), [[1, 2], [3, 4], [3, 4]])

        # All-zero durations produce empty 0-length sequence with preserved trailing dim and dtype
        all_zero = expand(vals_2d, [0, 0, 0])
        assert isinstance(all_zero, torch.Tensor)
        self.assertEqual(all_zero.shape, (0, 2))
        self.assertEqual(all_zero.dtype, torch.int64)

        # NumPy 1D array expansion (used in contour plotting)
        np_vals = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        np_durs = np.array([2, 1, 0], dtype=np.int64)
        np_expanded = expand(np_vals, np_durs)
        self.assertIsInstance(np_expanded, np.ndarray)
        assert isinstance(np_expanded, np.ndarray)
        self.assertEqual(np_expanded.dtype, np.float32)
        self.assertEqual(np_expanded.tolist(), [1.5, 1.5, 2.5])

    # ------------------------------------------------------------------
    # 8 & 9. Invalid Sequence Lengths & Empty Input Handling
    # ------------------------------------------------------------------

    def test_invalid_sequence_lengths_and_empty_inputs(self) -> None:
        """Verify invalid lengths, mismatched shapes, and empty inputs raise clear errors."""
        # get_mask_from_lengths invalid inputs
        with self.assertRaises(ValueError):
            get_mask_from_lengths(torch.tensor([], dtype=torch.long))  # Empty
        with self.assertRaises(ValueError):
            get_mask_from_lengths(torch.tensor([3, 0, 2], dtype=torch.long))  # Zero length
        with self.assertRaises(ValueError):
            get_mask_from_lengths(torch.tensor([3, -1, 2], dtype=torch.long))  # Negative length
        with self.assertRaises(ValueError):
            get_mask_from_lengths(torch.ones((2, 2), dtype=torch.long))  # 2D tensor
        with self.assertRaises(TypeError):
            get_mask_from_lengths(torch.tensor([3.0, 2.0]))  # Float tensor
        with self.assertRaises(TypeError):
            get_mask_from_lengths(torch.tensor([True, False]))  # Bool tensor
        with self.assertRaises(ValueError):
            get_mask_from_lengths(torch.tensor([4, 2], dtype=torch.long), max_len=3)  # max_len < max

        # pad invalid inputs
        with self.assertRaises(ValueError):
            pad([])  # Empty list
        with self.assertRaises(TypeError):
            pad(torch.ones((2, 3)))  # Passing a single Tensor instead of Sequence[Tensor]
        with self.assertRaises(ValueError):
            pad([torch.zeros((0, 4)), torch.zeros((0, 4))])  # All 0 length with mel_max_length=None
        with self.assertRaises(ValueError):
            pad([torch.ones((4, 8)), torch.ones((2, 8))], mel_max_length=3)  # mel_max_length < max
        with self.assertRaises(ValueError):
            pad([torch.ones((3, 8)), torch.ones((3, 4))])  # Mismatched feature dim D
        with self.assertRaises(ValueError):
            pad([torch.ones((3, 8), dtype=torch.float32), torch.ones((3, 8), dtype=torch.int64)])  # Mismatched dtype
        with self.assertRaises(ValueError):
            pad([torch.ones((2,)), torch.ones((2, 4))])  # Mismatched ndim

        # expand invalid inputs
        with self.assertRaises(ValueError):
            expand(torch.tensor([]), torch.tensor([]))  # Empty values
        with self.assertRaises(ValueError):
            expand(torch.tensor([1.0, 2.0]), torch.tensor([1, 2, 3]))  # Length mismatch
        with self.assertRaises(ValueError):
            expand(torch.tensor([1.0, 2.0]), torch.tensor([2, -1]))  # Negative duration
        with self.assertRaises(TypeError):
            expand(torch.tensor([1.0, 2.0]), torch.tensor([True, False]))  # Bool duration

        # to_device empty inputs
        with self.assertRaises(ValueError):
            to_device([], "cpu")
        with self.assertRaises(ValueError):
            to_device(np.array([]), "cpu")

    # ------------------------------------------------------------------
    # 10. CPU Operation (`to_device` on Tensors, 6-Tuple, and 12-Tuple)
    # ------------------------------------------------------------------

    def test_to_device_cpu_operation(self) -> None:
        """Verify `to_device` on CPU for standalone tensors, 6-tuple inference batches, and 12-tuple training batches."""
        # Standalone tensor preserves dtype
        for dt in (torch.float32, torch.float16, torch.int64, torch.bool):
            t = torch.ones((2, 3), dtype=dt)
            moved = to_device(t, "cpu")
            self.assertEqual(moved.device.type, "cpu")
            self.assertEqual(moved.dtype, dt)
            self.assertTrue(torch.equal(moved, t))

        # 6-element inference batch
        infer_batch = (
            ["utt_1", "utt_2"],
            ["hello", "world"],
            np.array([0, 1], dtype=np.int64),
            np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int64),
            np.array([3, 2], dtype=np.int64),
            3,
        )
        moved_infer = to_device(infer_batch, "cpu")
        self.assertEqual(len(moved_infer), 6)
        self.assertEqual(moved_infer[0], ["utt_1", "utt_2"])
        self.assertEqual(moved_infer[2].dtype, torch.long)
        self.assertEqual(moved_infer[3].shape, (2, 3))
        self.assertEqual(moved_infer[4].tolist(), [3, 2])
        self.assertEqual(moved_infer[5], 3)

        # 12-element training batch (note: energies passed as float64 ndarray to test float32 cast)
        train_batch = (
            ["utt_1", "utt_2"],
            ["hello", "world"],
            np.array([0, 1], dtype=np.int64),
            np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int64),
            np.array([3, 2], dtype=np.int64),
            3,
            np.ones((2, 10, 80), dtype=np.float32),
            np.array([10, 7], dtype=np.int64),
            10,
            np.ones((2, 3), dtype=np.float64),
            np.ones((2, 3), dtype=np.float64),
            np.array([[4, 3, 3], [4, 3, 0]], dtype=np.int64),
        )
        moved_train = to_device(train_batch, torch.device("cpu"))
        self.assertEqual(len(moved_train), 12)
        self.assertEqual(moved_train[6].dtype, torch.float32)  # mels
        self.assertEqual(moved_train[9].dtype, torch.float32)  # pitches
        self.assertEqual(moved_train[10].dtype, torch.float32)  # energies
        self.assertEqual(moved_train[11].dtype, torch.long)    # durations

    # ------------------------------------------------------------------
    # 11. Apple Silicon MPS Operation
    # ------------------------------------------------------------------

    def test_mps_operations_when_available(self) -> None:
        """Execute actual tensor operations (`to_device`, `get_mask_from_lengths`, `pad`, `expand`) on MPS."""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available on this host.")

        mps_device = resolve_device("mps")
        self.assertEqual(mps_device.type, "mps")

        # 1. to_device with 12-element training batch on MPS
        train_batch = (
            ["utt_1", "utt_2"],
            ["namaste", "telugu"],
            np.array([0, 0], dtype=np.int64),
            np.array([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=np.int64),
            np.array([4, 2], dtype=np.int64),
            4,
            np.ones((2, 8, 80), dtype=np.float32),
            np.array([8, 5], dtype=np.int64),
            8,
            np.ones((2, 4), dtype=np.float64),
            np.ones((2, 4), dtype=np.float64),
            np.array([[2, 2, 2, 2], [3, 2, 0, 0]], dtype=np.int64),
        )
        moved_batch = to_device(train_batch, mps_device)
        for idx in (2, 3, 4, 6, 7, 9, 10, 11):
            self.assertEqual(moved_batch[idx].device.type, "mps")

        # 2. get_mask_from_lengths on MPS (verifies DEC-01 / B-01 fix: no CPU/MPS mismatch!)
        src_lens_mps = moved_batch[4]
        src_mask_mps = get_mask_from_lengths(src_lens_mps, max_len=4)
        self.assertEqual(src_mask_mps.device.type, "mps")
        self.assertEqual(src_mask_mps.dtype, torch.bool)
        self.assertEqual(
            src_mask_mps.cpu().tolist(),
            [[False, False, False, False], [False, False, True, True]],
        )

        # 3. expand on MPS (2D hidden states)
        hidden_0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=mps_device)
        dur_0 = torch.tensor([2, 0, 1], dtype=torch.long, device=mps_device)
        exp_0 = expand(hidden_0, dur_0)
        assert isinstance(exp_0, torch.Tensor)
        self.assertEqual(exp_0.device.type, "mps")
        self.assertEqual(exp_0.shape, (3, 2))
        self.assertEqual(exp_0.cpu().tolist(), [[1.0, 2.0], [1.0, 2.0], [5.0, 6.0]])

        # 4. pad on MPS
        exp_1 = torch.ones((5, 2), dtype=torch.float32, device=mps_device)
        padded_mps = pad([exp_0, exp_1], mel_max_length=6)
        self.assertEqual(padded_mps.device.type, "mps")
        self.assertEqual(padded_mps.dtype, torch.float32)
        self.assertEqual(padded_mps.shape, (2, 6, 2))
        self.assertTrue(torch.all(padded_mps[0, 3:] == 0.0))

    # ------------------------------------------------------------------
    # 12. Unavailable / Invalid Device Error Handling (No Silent Fallback)
    # ------------------------------------------------------------------

    def test_unavailable_and_invalid_devices_raise_errors(self) -> None:
        """Verify requesting an unavailable or invalid device raises an error instead of falling back."""
        # CUDA is not available on this Apple Silicon Mac
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError) as ctx:
                resolve_device("cuda")
            self.assertIn("not available", str(ctx.exception))

            with self.assertRaises(RuntimeError):
                to_device(torch.ones(2, 2), "cuda:0")

        # Out-of-range MPS index (mps:2 does not exist on single-SoC Apple Silicon)
        with self.assertRaises(RuntimeError):
            resolve_device("mps:2")

        # Invalid device string or empty string
        with self.assertRaises(ValueError):
            resolve_device("")
        with self.assertRaises(ValueError):
            resolve_device("not_a_real_device")

        # Invalid type
        with self.assertRaises(TypeError):
            resolve_device(123)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 13. Phase 1 Modularization Import Parity & Object Identity
    # ------------------------------------------------------------------

    def test_modular_import_identity_and_backward_compatibility(self) -> None:
        """Verify that modular imports from device, mask, pad match tools.py and common.utils."""
        import importlib
        import common.utils as pkg_utils
        import common.utils.tools as facade_tools
        from common.utils.device import (
            _is_fastspeech2_batch_tuple,
            _to_float_tensor,
            _to_long_tensor,
            resolve_device as dev_resolve_device,
            to_device as dev_to_device,
        )
        from common.utils.mask import get_mask_from_lengths as mask_get_mask
        from common.utils.pad import (
            expand as pad_expand,
            pad as pad_pad,
            pad_1D as pad_pad_1D,
            pad_2D as pad_pad_2D,
        )

        mod_device = importlib.import_module("common.utils.device")
        mod_mask = importlib.import_module("common.utils.mask")
        mod_pad = importlib.import_module("common.utils.pad")

        # Check resolve_device & to_device
        self.assertIs(facade_tools.resolve_device, dev_resolve_device)
        self.assertIs(facade_tools.resolve_device, mod_device.resolve_device)
        self.assertIs(pkg_utils.resolve_device, dev_resolve_device)
        self.assertIs(facade_tools.to_device, dev_to_device)
        self.assertIs(facade_tools.to_device, mod_device.to_device)
        self.assertIs(pkg_utils.to_device, dev_to_device)

        # Check get_mask_from_lengths
        self.assertIs(facade_tools.get_mask_from_lengths, mask_get_mask)
        self.assertIs(facade_tools.get_mask_from_lengths, mod_mask.get_mask_from_lengths)
        self.assertIs(pkg_utils.get_mask_from_lengths, mask_get_mask)

        # Check pad, pad_1D, pad_2D, expand
        self.assertIs(facade_tools.pad, pad_pad)
        self.assertIs(facade_tools.pad, mod_pad.pad)
        self.assertIs(pkg_utils.pad, pad_pad)

        self.assertIs(facade_tools.pad_1D, pad_pad_1D)
        self.assertIs(facade_tools.pad_1D, mod_pad.pad_1D)
        self.assertIs(pkg_utils.pad_1D, pad_pad_1D)

        self.assertIs(facade_tools.pad_2D, pad_pad_2D)
        self.assertIs(facade_tools.pad_2D, mod_pad.pad_2D)
        self.assertIs(pkg_utils.pad_2D, pad_pad_2D)

        self.assertIs(facade_tools.expand, pad_expand)
        self.assertIs(facade_tools.expand, mod_pad.expand)
        self.assertIs(pkg_utils.expand, pad_expand)

        # Private helpers re-exported on facade
        self.assertIs(facade_tools._to_long_tensor, _to_long_tensor)
        self.assertIs(facade_tools._to_float_tensor, _to_float_tensor)
        self.assertIs(facade_tools._is_fastspeech2_batch_tuple, _is_fastspeech2_batch_tuple)


if __name__ == "__main__":
    unittest.main()

