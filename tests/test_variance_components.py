"""Unit tests for VariancePredictor and LengthRegulator (`common.model.modules`).

Covers all requirements specified in Stage 2 — Step 1.4F:
1. VariancePredictor initialization from direct arguments, ModelConfig, VariancePredictorConfig, and dict.
2. VariancePredictor output shapes [B, L_src].
3. VariancePredictor padding mask behavior (masked positions strictly zero).
4. VariancePredictor forward and backward autograd gradient flow on CPU.
5. VariancePredictor forward and backward on Apple Silicon MPS (if available).
6. VariancePredictor state_dict keys and shapes compatibility with original FastSpeech 2.
7. LengthRegulator expansion correctness and token order preservation.
8. LengthRegulator zero-duration phonemes handling (omitted from output).
9. LengthRegulator different expanded lengths within a batch with correct zero-padding.
10. LengthRegulator rejection of negative durations and invalid tensor shapes.
11. LengthRegulator all-zero durations edge case.
12. LengthRegulator dynamic device inheritance on CPU and MPS (DEC-01).
13. LengthRegulator gradient backpropagation to input encoder states.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from common.config import ModelConfig, VariancePredictorConfig
from common.model.modules import Conv, LengthRegulator, VariancePredictor
from common.utils import get_mask_from_lengths


class TestVarianceComponents(unittest.TestCase):
    """Comprehensive verification of VariancePredictor and LengthRegulator."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # A. VariancePredictor Tests
    # -------------------------------------------------------------------------

    def test_01_variance_predictor_init_and_config(self) -> None:
        """1. Verify VariancePredictor initialization from args, configs, dicts, and odd kernel padding."""
        # Direct args
        vp_direct = VariancePredictor(
            input_size=256, filter_size=256, kernel_size=3, dropout=0.5
        )
        self.assertEqual(vp_direct.input_size, 256)
        self.assertEqual(vp_direct.filter_size, 256)
        self.assertEqual(vp_direct.kernel, 3)
        self.assertEqual(vp_direct.dropout, 0.5)

        # From VariancePredictorConfig
        vp_cfg = VariancePredictorConfig(
            filter_size=128, kernel_size=5, dropout=0.2
        )
        vp_from_cfg = VariancePredictor(vp_cfg)
        self.assertEqual(vp_from_cfg.filter_size, 128)
        self.assertEqual(vp_from_cfg.kernel, 5)
        self.assertEqual(vp_from_cfg.dropout, 0.2)

        # From ModelConfig
        m_cfg = ModelConfig(
            n_src_vocab=100,
            variance_predictor=VariancePredictorConfig(
                filter_size=384, kernel_size=3, dropout=0.1
            ),
        )
        vp_from_mcfg = VariancePredictor(m_cfg)
        self.assertEqual(vp_from_mcfg.input_size, 256)
        self.assertEqual(vp_from_mcfg.filter_size, 384)

        # DEC-11: Check padding in both conv1d_1 and conv1d_2
        # For kernel=5, padding must be (5 - 1) // 2 = 2 in BOTH layers
        conv1 = vp_from_cfg.conv_layer.conv1d_1.conv
        conv2 = vp_from_cfg.conv_layer.conv1d_2.conv
        self.assertEqual(conv1.padding, (2,))
        self.assertEqual(conv2.padding, (2,))

        # Even kernel sizes must be rejected
        with self.assertRaises(ValueError):
            VariancePredictor(kernel_size=4)

    def test_02_variance_predictor_output_shapes(self) -> None:
        """2. Verify VariancePredictor outputs strictly [B, L_src] across batch sizes and lengths."""
        vp = VariancePredictor(
            input_size=256, filter_size=256, kernel_size=3, dropout=0.0
        )
        vp.eval()

        for b, l in [(1, 10), (2, 35), (4, 80)]:
            x = torch.randn(b, l, 256)
            out = vp(x)
            self.assertEqual(tuple(out.shape), (b, l))
            self.assertFalse(torch.isnan(out).any().item())

    def test_03_variance_predictor_padding_mask_behavior(self) -> None:
        """3. Verify padded positions are strictly zero and padding mask rejects shape mismatches."""
        vp = VariancePredictor(
            input_size=128, filter_size=128, kernel_size=3, dropout=0.0
        )
        vp.eval()

        b, l = 2, 20
        x = torch.randn(b, l, 128)
        mask = torch.zeros(b, l, dtype=torch.bool)
        mask[0, 15:] = True
        mask[1, 12:] = True

        out = vp(x, mask=mask)
        self.assertEqual(tuple(out.shape), (b, l))

        # Padded positions must be strictly 0.0
        self.assertEqual(out[0, 15:].abs().max().item(), 0.0)
        self.assertEqual(out[1, 12:].abs().max().item(), 0.0)

        # Valid positions must be non-zero
        self.assertGreater(out[0, :15].abs().max().item(), 0.0)
        self.assertGreater(out[1, :12].abs().max().item(), 0.0)

        # Mismatched mask shape raises ValueError
        with self.assertRaises(ValueError):
            vp(x, mask=torch.zeros(b, 25, dtype=torch.bool))

    def test_04_variance_predictor_cpu_forward_and_backward(self) -> None:
        """4. Verify CPU forward and backward passes produce finite, non-zero gradients."""
        vp = VariancePredictor(
            input_size=128, filter_size=128, kernel_size=3, dropout=0.1
        )
        vp.train()

        x = torch.randn(2, 15, 128, requires_grad=True)
        mask = torch.zeros(2, 15, dtype=torch.bool)
        mask[1, 10:] = True

        out = vp(x, mask=mask)
        loss = out.pow(2).sum()
        loss.backward()

        # Input gradients
        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any().item())
        self.assertGreater(x.grad.abs().max().item(), 0.0)

        # Trainable parameters gradients
        for name, param in vp.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient in {name}")
                assert param.grad is not None
                self.assertFalse(
                    torch.isnan(param.grad).any().item(),
                    f"NaN gradient in {name}",
                )

    def test_05_variance_predictor_mps_forward_and_backward(self) -> None:
        """5. Verify native Apple Silicon MPS execution when available."""
        if not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            self.skipTest("Apple Silicon MPS backend is not available.")

        mps_dev = torch.device("mps")
        vp = VariancePredictor(
            input_size=128, filter_size=128, kernel_size=3, dropout=0.1
        ).to(mps_dev)
        vp.train()

        x = torch.randn(2, 18, 128, device=mps_dev, requires_grad=True)
        mask = get_mask_from_lengths(
            torch.tensor([18, 12], device=mps_dev), max_len=18
        )

        out = vp(x, mask=mask)
        self.assertEqual(out.device.type, "mps")
        self.assertEqual(out[1, 12:].abs().max().item(), 0.0)

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertEqual(x.grad.device.type, "mps")
        self.assertFalse(torch.isnan(x.grad).any().item())

    def test_06_variance_predictor_state_dict_compatibility(self) -> None:
        """6. Verify exact state_dict keys and parameter shapes match original FastSpeech 2."""
        vp = VariancePredictor(
            input_size=256, filter_size=256, kernel_size=3, dropout=0.5
        )
        state_dict = vp.state_dict()

        expected_shapes = {
            "conv_layer.conv1d_1.conv.weight": (256, 256, 3),
            "conv_layer.conv1d_1.conv.bias": (256,),
            "conv_layer.layer_norm_1.weight": (256,),
            "conv_layer.layer_norm_1.bias": (256,),
            "conv_layer.conv1d_2.conv.weight": (256, 256, 3),
            "conv_layer.conv1d_2.conv.bias": (256,),
            "conv_layer.layer_norm_2.weight": (256,),
            "conv_layer.layer_norm_2.bias": (256,),
            "linear_layer.weight": (1, 256),
            "linear_layer.bias": (1,),
        }

        self.assertEqual(set(state_dict.keys()), set(expected_shapes.keys()))
        for k, expected_shape in expected_shapes.items():
            self.assertEqual(
                tuple(state_dict[k].shape),
                expected_shape,
                f"Shape mismatch for {k}",
            )

    # -------------------------------------------------------------------------
    # B. LengthRegulator Tests
    # -------------------------------------------------------------------------

    def test_07_length_regulator_expansion_correctness(self) -> None:
        """7. Verify single-utterance expansion repeats each phoneme vector by its duration."""
        lr = LengthRegulator()

        # Utterance of 3 tokens, each 4-dimensional
        batch = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
        )  # [3, 4]
        durations = torch.tensor([2, 1, 3], dtype=torch.long)

        expanded = lr.expand(batch, durations)
        # Expected frames: 2 of token 1, 1 of token 2, 3 of token 3 = 6 frames
        self.assertEqual(tuple(expanded.shape), (6, 4))

        expected = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
            ]
        )
        torch.testing.assert_close(expanded, expected)

    def test_08_length_regulator_zero_duration_phonemes(self) -> None:
        """8. Verify zero-duration phonemes are omitted without generating frames or crashing."""
        lr = LengthRegulator()

        batch = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],  # duration 0 -> omitted
                [3.0, 3.0],
                [4.0, 4.0],  # duration 0 -> omitted
            ]
        )
        durations = torch.tensor([2, 0, 1, 0], dtype=torch.long)

        expanded = lr.expand(batch, durations)
        self.assertEqual(tuple(expanded.shape), (3, 2))

        expected = torch.tensor([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0]])
        torch.testing.assert_close(expanded, expected)

    def test_09_length_regulator_different_expanded_lengths_and_padding(
        self,
    ) -> None:
        """9. Verify batched expansion pads shorter samples to max batch length (or max_len)."""
        lr = LengthRegulator()

        # Batch size 2, sequence length 4, hidden dim 8
        x = torch.randn(2, 4, 8)
        durations = torch.tensor(
            [[1, 2, 0, 1], [2, 1, 3, 2]], dtype=torch.long
        )  # sample 0 sum = 4, sample 1 sum = 8

        # A. Default max_len=None: pads to max(mel_lens) = 8
        output, mel_lens = lr(x, durations)
        self.assertEqual(tuple(output.shape), (2, 8, 8))
        self.assertEqual(mel_lens.tolist(), [4, 8])

        # Sample 0 frames 4..7 must be strictly 0.0 (zero-padding)
        self.assertEqual(output[0, 4:].abs().max().item(), 0.0)
        # Sample 0 frames 0..3 must contain valid non-zero representations
        self.assertGreater(output[0, :4].abs().max().item(), 0.0)

        # B. Explicit max_len=12
        output_12, mel_lens_12 = lr(x, durations, max_len=12)
        self.assertEqual(tuple(output_12.shape), (2, 12, 8))
        self.assertEqual(mel_lens_12.tolist(), [4, 8])
        self.assertEqual(output_12[0, 4:].abs().max().item(), 0.0)
        self.assertEqual(output_12[1, 8:].abs().max().item(), 0.0)

    def test_10_length_regulator_invalid_inputs_and_negative_durations(
        self,
    ) -> None:
        """10. Verify invalid shapes, empty inputs, and negative durations raise ValueError."""
        lr = LengthRegulator()

        x = torch.randn(2, 4, 8)
        durations = torch.tensor([[1, 2, 1, 1], [2, 1, 1, 2]], dtype=torch.long)

        # 1. Negative durations raise ValueError (avoid silently altering durations)
        neg_durations = torch.tensor(
            [[1, -1, 1, 1], [2, 1, 1, 2]], dtype=torch.long
        )
        with self.assertRaises(ValueError) as ctx:
            lr(x, neg_durations)
        self.assertIn("negative duration values", str(ctx.exception))

        # 2. Shape mismatch between x and durations
        with self.assertRaises(ValueError):
            lr(x, torch.tensor([[1, 2, 1], [2, 1, 1]], dtype=torch.long))

        # 3. 2D x instead of 3D
        with self.assertRaises(ValueError):
            lr(torch.randn(4, 8), durations)

        # 4. max_len smaller than expanded length raises ValueError
        with self.assertRaises(ValueError):
            lr(x, durations, max_len=3)  # expanded length is 5 > 3

    def test_11_length_regulator_all_zero_durations_edge_case(self) -> None:
        """11. Verify all-zero durations across all utterances produces deterministic empty/zero tensor."""
        lr = LengthRegulator()

        x = torch.randn(2, 3, 16)
        all_zero_durations = torch.zeros(2, 3, dtype=torch.long)

        # When max_len is None: returns [B, 0, H]
        out, mel_lens = lr(x, all_zero_durations, max_len=None)
        self.assertEqual(tuple(out.shape), (2, 0, 16))
        self.assertEqual(mel_lens.tolist(), [0, 0])

        # When max_len=10: returns [B, 10, H] of zeros
        out_10, mel_lens_10 = lr(x, all_zero_durations, max_len=10)
        self.assertEqual(tuple(out_10.shape), (2, 10, 16))
        self.assertEqual(out_10.abs().max().item(), 0.0)
        self.assertEqual(mel_lens_10.tolist(), [0, 0])

    def test_12_length_regulator_cpu_and_mps_device_preservation(self) -> None:
        """12. Verify DEC-01: dynamic device inheritance on CPU and MPS without global device."""
        lr = LengthRegulator()

        # CPU
        x_cpu = torch.randn(2, 3, 8)
        dur_cpu = torch.tensor([[1, 2, 1], [2, 1, 0]], dtype=torch.long)
        out_cpu, mel_len_cpu = lr(x_cpu, dur_cpu)
        self.assertEqual(out_cpu.device.type, "cpu")
        self.assertEqual(mel_len_cpu.device.type, "cpu")

        # MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_dev = torch.device("mps")
            x_mps = x_cpu.to(mps_dev)
            dur_mps = dur_cpu.to(mps_dev)
            out_mps, mel_len_mps = lr(x_mps, dur_mps)
            self.assertEqual(out_mps.device.type, "mps")
            self.assertEqual(mel_len_mps.device.type, "mps")
            torch.testing.assert_close(out_mps.cpu(), out_cpu)

    def test_13_length_regulator_gradient_propagation(self) -> None:
        """13. Verify gradient backpropagates from expanded output to input encoder hidden states."""
        lr = LengthRegulator()

        # [1, 3, 4]
        x = torch.randn(1, 3, 4, requires_grad=True)
        # Token 0 duration 2, token 1 duration 0, token 2 duration 3
        durations = torch.tensor([[2, 0, 3]], dtype=torch.long)

        out, mel_len = lr(x, durations)
        self.assertEqual(tuple(out.shape), (1, 5, 4))

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        # Gradients on token 0 (repeated 2 times) must be 2.0
        torch.testing.assert_close(
            x.grad[0, 0], torch.tensor([2.0, 2.0, 2.0, 2.0])
        )
        # Gradients on token 1 (repeated 0 times) must be strictly 0.0
        torch.testing.assert_close(
            x.grad[0, 1], torch.tensor([0.0, 0.0, 0.0, 0.0])
        )
        # Gradients on token 2 (repeated 3 times) must be 3.0
        torch.testing.assert_close(
            x.grad[0, 2], torch.tensor([3.0, 3.0, 3.0, 3.0])
        )

    # -------------------------------------------------------------------------
    # C. Phase 3 Modularization & Parity Tests
    # -------------------------------------------------------------------------

    def test_14_phase3_component_import_identity(self) -> None:
        """14. Verify Conv, VariancePredictor, and LengthRegulator import identity across all paths."""
        # Canonical paths
        from common.model.conv import Conv as CanonicalConv
        from common.model.length_regulator import LengthRegulator as CanonicalLR
        from common.model.variance_predictor import VariancePredictor as CanonicalVP

        # Facade module path
        from common.model.modules import (
            Conv as FacadeConv,
            LengthRegulator as FacadeLR,
            VariancePredictor as FacadeVP,
        )

        # Model package path
        from common.model import (
            Conv as ModelConv,
            LengthRegulator as ModelLR,
            VariancePredictor as ModelVP,
        )

        # Legacy modules package path
        from common.modules import (
            Conv as ModulesConv,
            LengthRegulator as ModulesLR,
            VariancePredictor as ModulesVP,
        )

        # Top-level common export path
        from common import (
            Conv as CommonConv,
            LengthRegulator as CommonLR,
            VariancePredictor as CommonVP,
        )

        # Conv object identity
        self.assertIs(CanonicalConv, FacadeConv)
        self.assertIs(CanonicalConv, ModelConv)
        self.assertIs(CanonicalConv, ModulesConv)
        self.assertIs(CanonicalConv, CommonConv)

        # VariancePredictor object identity
        self.assertIs(CanonicalVP, FacadeVP)
        self.assertIs(CanonicalVP, ModelVP)
        self.assertIs(CanonicalVP, ModulesVP)
        self.assertIs(CanonicalVP, CommonVP)

        # LengthRegulator object identity
        self.assertIs(CanonicalLR, FacadeLR)
        self.assertIs(CanonicalLR, ModelLR)
        self.assertIs(CanonicalLR, ModulesLR)
        self.assertIs(CanonicalLR, CommonLR)

    def test_15_phase3_isolated_submodules(self) -> None:
        """15. Verify isolated canonical modules can be instantiated and executed independently."""
        from common.model.conv import Conv
        from common.model.length_regulator import LengthRegulator
        from common.model.variance_predictor import VariancePredictor

        # Conv isolated execution
        conv = Conv(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        x_conv = torch.randn(2, 7, 16)
        out_conv = conv(x_conv)
        self.assertEqual(tuple(out_conv.shape), (2, 7, 32))

        # VariancePredictor isolated execution
        vp = VariancePredictor(input_size=16, filter_size=32, kernel_size=3, dropout=0.0)
        x_vp = torch.randn(2, 7, 16)
        mask = torch.tensor([[False] * 5 + [True] * 2, [False] * 7])
        out_vp = vp(x_vp, mask=mask)
        self.assertEqual(tuple(out_vp.shape), (2, 7))
        self.assertTrue(torch.all(out_vp[0, 5:] == 0.0))

        # LengthRegulator isolated execution
        lr = LengthRegulator()
        x_lr = torch.randn(2, 3, 16)
        dur = torch.tensor([[1, 2, 0], [0, 1, 1]], dtype=torch.long)
        out_lr, mel_len = lr(x_lr, dur)
        self.assertEqual(tuple(out_lr.shape), (2, 3, 16))
        self.assertEqual(mel_len.tolist(), [3, 2])


if __name__ == "__main__":
    unittest.main()

