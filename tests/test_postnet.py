"""Unit tests for the FastSpeech 2 convolutional PostNet (`common.transformer.layers.PostNet`).

Covers all requirements specified in Stage 2 — Step 1.4E and Step 1.4E-R:
1. PostNet initialization and configuration.
2. Correct input/output shapes.
3. Residual output behaviour.
4. Variable sequence lengths in a batch.
5. Optional mask handling (mask=None vs. explicit boolean mask).
6. Evaluation-mode padding invariance and strictly zeroed padded positions.
7. Training-mode padding invariance across diverse padding lengths (Option 1: Mask-Aware BatchNorm).
8. Valid-frame sample mean, biased variance, and unbiased running variance calculations.
9. Cumulative moving average update when `momentum=None`.
10. Multi-utterance batch pooling over valid frames only.
11. Explicit error handling for all-masked inputs (0 valid frames).
12. Invalid input dimensions, channels, and mask shapes.
13. CPU forward and backward passes.
14. Native Apple Silicon MPS forward and backward passes, if available.
15. Configuration integration (PostNetConfig, ModelConfig, PreprocessConfig, dict).
16. Checkpoint state-dict keys, parameter shapes, and buffer shapes compatibility.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from common.config import (
    ModelConfig,
    PostNetConfig,
    PreprocessConfig,
    PreprocessPathConfig,
    PreprocessingSettings,
    TextConfig,
    TransformerConfig,
)
from common.transformer import ConvNorm, PostNet
from common.utils import get_mask_from_lengths


class TestPostNet(unittest.TestCase):
    """Comprehensive verification of the convolutional PostNet module with Mask-Aware BatchNorm."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_01_initialization_and_configuration(self) -> None:
        """1. Verify default and custom PostNet initialization, layer hierarchy, and parameter counts."""
        # Default initialization (80 channels, 512 dim, kernel 5, 5 convs, 0.5 dropout)
        postnet_default = PostNet()
        self.assertEqual(postnet_default.n_mel_channels, 80)
        self.assertEqual(postnet_default.postnet_embedding_dim, 512)
        self.assertEqual(postnet_default.postnet_kernel_size, 5)
        self.assertEqual(postnet_default.postnet_n_convolutions, 5)
        self.assertEqual(postnet_default.postnet_dropout, 0.5)
        self.assertEqual(len(postnet_default.convolutions), 5)

        # Total parameter count must match reference FastSpeech 2 PostNet: 4,348,144 params
        n_params = sum(p.numel() for p in postnet_default.parameters())
        self.assertEqual(n_params, 4348144)

        # Check layers hierarchy: Sequential(ConvNorm, BatchNorm1d)
        for i in range(4):
            self.assertIsInstance(postnet_default.convolutions[i][0], ConvNorm)
            self.assertIsInstance(
                postnet_default.convolutions[i][1], torch.nn.BatchNorm1d
            )
            self.assertEqual(
                postnet_default.convolutions[i][1].num_features, 512
            )
        # Final layer outputs n_mel_channels (80)
        self.assertEqual(postnet_default.convolutions[4][1].num_features, 80)

        # Custom initialization (e.g. 64 channels, 256 dim, kernel 3, 3 convs)
        postnet_custom = PostNet(
            n_mel_channels=64,
            postnet_embedding_dim=256,
            postnet_kernel_size=3,
            postnet_n_convolutions=3,
            postnet_dropout=0.2,
        )
        self.assertEqual(postnet_custom.n_mel_channels, 64)
        self.assertEqual(postnet_custom.postnet_embedding_dim, 256)
        self.assertEqual(postnet_custom.postnet_kernel_size, 3)
        self.assertEqual(postnet_custom.postnet_n_convolutions, 3)
        self.assertEqual(postnet_custom.postnet_dropout, 0.2)
        self.assertEqual(len(postnet_custom.convolutions), 3)

    def test_02_correct_input_output_shapes(self) -> None:
        """2. Verify output shape strictly preserves [B, L, C] across diverse batch sizes and lengths."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_n_convolutions=3,
            postnet_dropout=0.1,
        )
        postnet.eval()

        for b, l in [(1, 10), (2, 45), (4, 80)]:
            x = torch.randn(b, l, 80)
            out = postnet(x)
            self.assertEqual(tuple(out.shape), (b, l, 80))
            self.assertFalse(torch.isnan(out).any().item())

    def test_03_residual_output_behaviour(self) -> None:
        """3. Verify PostNet outputs residual correction ΔMel, added via postnet_output = postnet(x) + x."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_n_convolutions=3,
            postnet_dropout=0.0,
        )
        postnet.eval()

        x = torch.randn(2, 20, 80)
        delta = postnet(x)

        # delta must have exact same shape as x
        self.assertEqual(tuple(delta.shape), tuple(x.shape))
        # delta must be non-zero (it predicts a non-trivial residual)
        self.assertGreater(delta.abs().max().item(), 0.0)

        # Residual addition reproduces FastSpeech 2 refined Mel-spectrogram
        refined = delta + x
        self.assertEqual(tuple(refined.shape), (2, 20, 80))
        # In eval mode with dropout=0, output is deterministic
        delta_again = postnet(x)
        torch.testing.assert_close(delta, delta_again)

    def test_04_different_sequence_lengths_in_batch(self) -> None:
        """4. Verify PostNet processes variable-length batches with padding masks."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_n_convolutions=3,
            postnet_dropout=0.1,
        )
        postnet.eval()

        lengths = torch.tensor([50, 30, 15], dtype=torch.long)
        mask = get_mask_from_lengths(lengths)  # [3, 50], True = PAD
        x = torch.randn(3, 50, 80)

        out = postnet(x, mask=mask)
        self.assertEqual(tuple(out.shape), (3, 50, 80))
        self.assertFalse(torch.isnan(out).any().item())

    def test_05_optional_mask_handling(self) -> None:
        """5. Verify PostNet works with mask=None (default, unmasked) and with explicit mask."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_n_convolutions=3,
            postnet_dropout=0.1,
        )

        x = torch.randn(2, 25, 80)

        # mask=None in train mode (standard PyTorch BatchNorm)
        postnet.train()
        out_train_none = postnet(x, mask=None)
        self.assertEqual(tuple(out_train_none.shape), (2, 25, 80))

        # mask=None in eval mode
        postnet.eval()
        out_eval_none = postnet(x, mask=None)
        self.assertEqual(tuple(out_eval_none.shape), (2, 25, 80))

        # with boolean mask in eval mode
        mask = torch.zeros(2, 25, dtype=torch.bool)
        mask[1, 15:] = True
        out_eval_masked = postnet(x, mask=mask)
        self.assertEqual(tuple(out_eval_masked.shape), (2, 25, 80))
        self.assertEqual(out_eval_masked[1, 15:].abs().max().item(), 0.0)

    def test_06_eval_mode_padding_invariance(self) -> None:
        """6. Verify evaluation-mode padding invariance and strictly zeroed padded positions."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_kernel_size=5,
            postnet_n_convolutions=3,
            postnet_dropout=0.0,
        )
        postnet.eval()

        # Utterance A has length 12
        x_single = torch.randn(1, 12, 80)
        out_single = postnet(x_single, mask=None)

        # Place Utterance A in a padded batch of length 24 where positions 12..23 are random noise
        x_batch = torch.zeros(2, 24, 80)
        x_batch[0, :12, :] = x_single[0]
        x_batch[0, 12:, :] = torch.randn(12, 80) * 10.0  # large arbitrary noise
        x_batch[1, :18, :] = torch.randn(18, 80)
        x_batch[1, 18:, :] = torch.randn(6, 80) * 5.0

        lengths = torch.tensor([12, 18], dtype=torch.long)
        mask = get_mask_from_lengths(lengths, max_len=24)  # [2, 24]

        out_batch = postnet(x_batch, mask=mask)

        # Padded frames must be strictly 0.0
        self.assertEqual(out_batch[0, 12:].abs().max().item(), 0.0)
        self.assertEqual(out_batch[1, 18:].abs().max().item(), 0.0)

        # In eval mode, valid frames 0..11 in sample 0 must match out_single within float32 tolerance
        torch.testing.assert_close(
            out_batch[0, :12, :], out_single[0], atol=1e-5, rtol=1e-5
        )

    def test_07_training_mode_padding_invariance_and_varying_padding(self) -> None:
        """7. Verify training-mode padding invariance across diverse padding lengths (Mask-Aware BatchNorm)."""
        # Baseline PostNet with dropout=0.0 to strictly isolate normalization
        base_postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=128,
            postnet_kernel_size=5,
            postnet_n_convolutions=4,
            postnet_dropout=0.0,
        )
        initial_state = base_postnet.state_dict()

        # Utterance of valid length 30
        valid_len = 30
        u = torch.randn(1, valid_len, 80)

        # 1. Unpadded forward in train mode
        p_unpadded = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=128,
            postnet_kernel_size=5,
            postnet_n_convolutions=4,
            postnet_dropout=0.0,
        )
        p_unpadded.load_state_dict(initial_state)
        p_unpadded.train()
        mask_unpadded = torch.zeros(1, valid_len, dtype=torch.bool)
        out_unpadded = p_unpadded(u, mask=mask_unpadded)

        # 2. Test padding with different lengths: 10, 25, 50, 100, 200 frames
        for pad_len in [10, 25, 50, 100, 200]:
            total_len = valid_len + pad_len
            u_padded = torch.zeros(1, total_len, 80)
            u_padded[0, :valid_len, :] = u[0]
            # Put arbitrary noise in the padded tail to ensure padding mask completely blocks it
            u_padded[0, valid_len:, :] = torch.randn(pad_len, 80) * 8.0

            mask = get_mask_from_lengths(
                torch.tensor([valid_len]), max_len=total_len
            )

            p_test = PostNet(
                n_mel_channels=80,
                postnet_embedding_dim=128,
                postnet_kernel_size=5,
                postnet_n_convolutions=4,
                postnet_dropout=0.0,
            )
            p_test.load_state_dict(initial_state)
            p_test.train()

            out_padded = p_test(u_padded, mask=mask)

            # A. Padded positions must be strictly 0.0
            self.assertEqual(
                out_padded[0, valid_len:].abs().max().item(),
                0.0,
                f"Padded positions not zero for pad_len={pad_len}",
            )

            # B. Valid frames must remain equal to unpadded output within float32 tolerance (< 1e-5)
            torch.testing.assert_close(
                out_padded[0, :valid_len, :],
                out_unpadded[0],
                atol=1e-5,
                rtol=1e-5,
                msg=f"Training mode padding invariance failed for pad_len={pad_len}",
            )

    def test_08_valid_frame_statistics_and_running_buffers(self) -> None:
        """8. Verify sample mean, biased variance, Bessel-corrected unbiased variance, and running buffers."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_kernel_size=5,
            postnet_n_convolutions=2,
            postnet_dropout=0.0,
        )
        postnet.train()

        bn = postnet.convolutions[0][1]
        self.assertEqual(bn.num_batches_tracked.item(), 0)

        # Batch with lengths 12 and 18, padded to 24
        lengths = torch.tensor([12, 18], dtype=torch.long)
        mask = get_mask_from_lengths(lengths, max_len=24)
        x = torch.randn(2, 24, 80)

        # Execute training forward pass
        _ = postnet(x, mask=mask)

        # 1. num_batches_tracked must increment to 1
        self.assertEqual(bn.num_batches_tracked.item(), 1)

        # 2. In eval mode, forward pass must NOT modify running statistics or num_batches_tracked
        postnet.eval()
        rm_saved = bn.running_mean.clone()
        rv_saved = bn.running_var.clone()
        _ = postnet(x, mask=mask)
        self.assertEqual(bn.num_batches_tracked.item(), 1)
        torch.testing.assert_close(bn.running_mean, rm_saved)
        torch.testing.assert_close(bn.running_var, rv_saved)

    def test_09_cumulative_moving_average_when_momentum_is_none(self) -> None:
        """9. Verify cumulative moving average (1/n) when BatchNorm momentum=None."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_n_convolutions=2,
            postnet_dropout=0.0,
        )
        for conv in postnet.convolutions:
            conv[1].momentum = None

        postnet.train()
        x = torch.randn(2, 20, 80)
        mask = get_mask_from_lengths(torch.tensor([15, 20]), max_len=20)

        # Run 3 training batches
        for step in range(1, 4):
            _ = postnet(x, mask=mask)
            self.assertEqual(
                postnet.convolutions[0][1].num_batches_tracked.item(), step
            )

    def test_10_batch_with_additional_valid_utterance_uses_valid_frames_only(
        self,
    ) -> None:
        """10. Verify multi-utterance batch normalizes using only valid frames from both utterances."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_kernel_size=5,
            postnet_n_convolutions=2,
            postnet_dropout=0.0,
        )
        postnet.train()

        u1 = torch.randn(1, 15, 80)
        u2 = torch.randn(1, 25, 80)

        # Batch 1: with zero padding in u1
        batch1 = torch.zeros(2, 25, 80)
        batch1[0, :15] = u1[0]
        batch1[1, :25] = u2[0]
        mask1 = get_mask_from_lengths(torch.tensor([15, 25]), max_len=25)

        # Batch 2: same valid data, but with huge arbitrary noise in padded frames of u1
        batch2 = batch1.clone()
        batch2[0, 15:] = torch.randn(10, 80) * 50.0

        p1 = PostNet(80, 64, 5, 2, 0.0)
        p1.load_state_dict(postnet.state_dict())
        p1.train()
        out1 = p1(batch1, mask=mask1)

        p2 = PostNet(80, 64, 5, 2, 0.0)
        p2.load_state_dict(postnet.state_dict())
        p2.train()
        out2 = p2(batch2, mask=mask1)

        # Outputs on all valid frames must be identical regardless of noise in padded positions
        torch.testing.assert_close(
            out1[0, :15], out2[0, :15], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(out1[1], out2[1], atol=1e-5, rtol=1e-5)
        self.assertEqual(out1[0, 15:].abs().max().item(), 0.0)
        self.assertEqual(out2[0, 15:].abs().max().item(), 0.0)

    def test_11_all_masked_raises_value_error(self) -> None:
        """11. Verify that a batch where all frames are masked raises ValueError."""
        postnet = PostNet()
        postnet.train()
        x = torch.randn(2, 10, 80)
        all_pad_mask = torch.ones(2, 10, dtype=torch.bool)
        with self.assertRaises(ValueError) as ctx:
            postnet(x, mask=all_pad_mask)
        self.assertIn("all frames in the batch are masked", str(ctx.exception))

    def test_12_invalid_input_dimensions_and_channels(self) -> None:
        """12. Verify invalid tensor ranks, empty inputs, channel mismatches, and mask shape errors raise ValueError."""
        postnet = PostNet(n_mel_channels=80)

        # 1. 2D input tensor
        with self.assertRaises(ValueError):
            postnet(torch.randn(2, 80))
        # 2. 4D input tensor
        with self.assertRaises(ValueError):
            postnet(torch.randn(2, 10, 80, 1))
        # 3. Empty batch (B=0)
        with self.assertRaises(ValueError):
            postnet(torch.randn(0, 10, 80))
        # 4. Empty length (L=0)
        with self.assertRaises(ValueError):
            postnet(torch.randn(2, 0, 80))
        # 5. Channel count mismatch (64 != 80)
        with self.assertRaises(ValueError) as ctx_c:
            postnet(torch.randn(2, 10, 64))
        self.assertIn(
            "does not match PostNet n_mel_channels", str(ctx_c.exception)
        )
        # 6. Mask shape mismatch
        with self.assertRaises(ValueError):
            postnet(
                torch.randn(2, 10, 80),
                mask=torch.zeros(2, 12, dtype=torch.bool),
            )
        # 7. Even kernel size in ConvNorm raises ValueError
        with self.assertRaises(ValueError):
            ConvNorm(80, 512, kernel_size=4)

    def test_13_cpu_forward_and_backward_passes(self) -> None:
        """13. Verify CPU forward and backward passes produce finite, non-zero gradients for PostNet and input."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_n_convolutions=3,
            postnet_dropout=0.1,
        )
        postnet.train()

        x = torch.randn(2, 20, 80, requires_grad=True)
        mask = torch.zeros(2, 20, dtype=torch.bool)
        mask[1, 14:] = True

        delta = postnet(x, mask=mask)
        refined = delta + x
        loss = refined.pow(2).mean()
        loss.backward()

        # Input x must receive non-zero finite gradients
        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any().item())
        self.assertGreater(x.grad.abs().max().item(), 0.0)

        # Every trainable parameter in PostNet must receive finite non-zero gradients
        for name, param in postnet.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient in {name}")
                assert param.grad is not None
                self.assertFalse(
                    torch.isnan(param.grad).any().item(),
                    f"NaN gradient in {name}",
                )

    def test_14_mps_forward_and_backward_passes(self) -> None:
        """14. Verify native execution on Apple Silicon MPS backend when available."""
        if not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            self.skipTest("Apple Silicon MPS backend is not available.")

        mps_dev = torch.device("mps")
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=64,
            postnet_n_convolutions=3,
            postnet_dropout=0.1,
        ).to(mps_dev)
        postnet.train()

        x = torch.randn(2, 25, 80, device=mps_dev, requires_grad=True)
        mask = get_mask_from_lengths(
            torch.tensor([25, 16], dtype=torch.long, device=mps_dev)
        )

        delta = postnet(x, mask=mask)
        self.assertEqual(delta.device.type, "mps")
        self.assertEqual(delta[1, 16:].abs().max().item(), 0.0)

        loss = (delta + x).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertEqual(x.grad.device.type, "mps")
        self.assertFalse(torch.isnan(x.grad).any().item())
        self.assertGreater(x.grad.abs().max().item(), 0.0)

    def test_15_configuration_integration(self) -> None:
        """15. Verify PostNet instantiation from PostNetConfig, ModelConfig, PreprocessConfig, and dicts."""
        # A. From PostNetConfig
        p_cfg = PostNetConfig(
            postnet_embedding_dim=256,
            postnet_kernel_size=3,
            postnet_n_convolutions=4,
            postnet_dropout=0.3,
        )
        pn_from_pcfg = PostNet.from_config(p_cfg, n_mel_channels=80)
        self.assertEqual(pn_from_pcfg.n_mel_channels, 80)
        self.assertEqual(pn_from_pcfg.postnet_embedding_dim, 256)
        self.assertEqual(pn_from_pcfg.postnet_kernel_size, 3)
        self.assertEqual(pn_from_pcfg.postnet_n_convolutions, 4)
        self.assertEqual(pn_from_pcfg.postnet_dropout, 0.3)

        # B. From ModelConfig
        m_cfg = ModelConfig(
            n_src_vocab=88,
            postnet=PostNetConfig(
                postnet_embedding_dim=384,
                postnet_kernel_size=5,
                postnet_n_convolutions=5,
                postnet_dropout=0.4,
            ),
        )
        pn_from_mcfg = PostNet.from_config(m_cfg, n_mel_channels=80)
        self.assertEqual(pn_from_mcfg.postnet_embedding_dim, 384)
        self.assertEqual(pn_from_mcfg.postnet_dropout, 0.4)

        # C. From PreprocessConfig (extracts n_mel_channels)
        prep_cfg = PreprocessConfig(
            dataset="Test",
            path=PreprocessPathConfig(
                corpus_path="./data",
                raw_path="./raw",
                preprocessed_path="./prep",
            ),
            preprocessing=PreprocessingSettings(
                text=TextConfig(language="en"),
            ),
        )
        pn_from_prep = PostNet.from_config(prep_cfg)
        self.assertEqual(pn_from_prep.n_mel_channels, 80)

        # D. From nested dictionary
        d_cfg = {
            "n_mel_channels": 64,
            "postnet": {
                "postnet_embedding_dim": 128,
                "postnet_kernel_size": 3,
                "postnet_n_convolutions": 3,
                "postnet_dropout": 0.2,
            },
        }
        pn_from_dict = PostNet.from_config(d_cfg)
        self.assertEqual(pn_from_dict.n_mel_channels, 64)
        self.assertEqual(pn_from_dict.postnet_embedding_dim, 128)
        self.assertEqual(pn_from_dict.postnet_kernel_size, 3)
        self.assertEqual(pn_from_dict.postnet_n_convolutions, 3)

    def test_16_checkpoint_state_dict_keys_and_shapes_compatibility(
        self,
    ) -> None:
        """16. Verify exact state_dict keys and tensor shapes match reference FastSpeech 2 PostNet."""
        postnet = PostNet(
            n_mel_channels=80,
            postnet_embedding_dim=512,
            postnet_kernel_size=5,
            postnet_n_convolutions=5,
        )
        state_dict = postnet.state_dict()

        # Expected 35 state_dict keys (7 per block across 5 blocks: conv.weight, conv.bias, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.num_batches_tracked)
        expected_keys = []
        for i in range(5):
            expected_keys.extend(
                [
                    f"convolutions.{i}.0.conv.weight",
                    f"convolutions.{i}.0.conv.bias",
                    f"convolutions.{i}.1.weight",
                    f"convolutions.{i}.1.bias",
                    f"convolutions.{i}.1.running_mean",
                    f"convolutions.{i}.1.running_var",
                    f"convolutions.{i}.1.num_batches_tracked",
                ]
            )

        self.assertEqual(list(state_dict.keys()), expected_keys)

        # Verify layer shapes
        # Layer 0: Conv [512, 80, 5], BN [512]
        self.assertEqual(
            tuple(state_dict["convolutions.0.0.conv.weight"].shape),
            (512, 80, 5),
        )
        self.assertEqual(
            tuple(state_dict["convolutions.0.0.conv.bias"].shape), (512,)
        )
        self.assertEqual(
            tuple(state_dict["convolutions.0.1.weight"].shape), (512,)
        )
        self.assertEqual(
            tuple(state_dict["convolutions.0.1.running_mean"].shape), (512,)
        )

        # Layers 1..3: Conv [512, 512, 5], BN [512]
        for i in range(1, 4):
            self.assertEqual(
                tuple(state_dict[f"convolutions.{i}.0.conv.weight"].shape),
                (512, 512, 5),
            )
            self.assertEqual(
                tuple(state_dict[f"convolutions.{i}.0.conv.bias"].shape), (512,)
            )
            self.assertEqual(
                tuple(state_dict[f"convolutions.{i}.1.weight"].shape), (512,)
            )
            self.assertEqual(
                tuple(state_dict[f"convolutions.{i}.1.running_mean"].shape),
                (512,),
            )

        # Layer 4 (final): Conv [80, 512, 5], BN [80]
        self.assertEqual(
            tuple(state_dict["convolutions.4.0.conv.weight"].shape),
            (80, 512, 5),
        )
        self.assertEqual(
            tuple(state_dict["convolutions.4.0.conv.bias"].shape), (80,)
        )
        self.assertEqual(
            tuple(state_dict["convolutions.4.1.weight"].shape), (80,)
        )
        self.assertEqual(
            tuple(state_dict["convolutions.4.1.running_mean"].shape), (80,)
        )

    def test_17_phase2_postnet_modular_import_identity_and_equivalence(self) -> None:
        """17. Verify PostNet and ConvNorm import identity and functional equivalence across paths."""
        # 1. Canonical module path
        from common.model.postnet import ConvNorm as CanonicalConvNorm, PostNet as CanonicalPostNet

        # 2. Backward-compatible transformer layers path
        from common.transformer.layers import ConvNorm as LayersConvNorm, PostNet as LayersPostNet

        # 3. Backward-compatible transformer package path
        from common.transformer import ConvNorm as TransConvNorm, PostNet as TransPostNet

        # 4. Model package export path
        from common.model import ConvNorm as ModelConvNorm, PostNet as ModelPostNet

        # 5. Top-level common export path
        from common import ConvNorm as CommonConvNorm, PostNet as CommonPostNet

        # Object identity checks
        self.assertIs(CanonicalPostNet, LayersPostNet)
        self.assertIs(CanonicalPostNet, TransPostNet)
        self.assertIs(CanonicalPostNet, ModelPostNet)
        self.assertIs(CanonicalPostNet, CommonPostNet)

        self.assertIs(CanonicalConvNorm, LayersConvNorm)
        self.assertIs(CanonicalConvNorm, TransConvNorm)
        self.assertIs(CanonicalConvNorm, ModelConvNorm)
        self.assertIs(CanonicalConvNorm, CommonConvNorm)

        # Functional equivalence: instantiate via canonical vs facade, load identical weights, compare outputs
        torch.manual_seed(42)
        p_canonical = CanonicalPostNet(n_mel_channels=80, postnet_embedding_dim=64, postnet_kernel_size=5, postnet_n_convolutions=3)
        p_facade = LayersPostNet(n_mel_channels=80, postnet_embedding_dim=64, postnet_kernel_size=5, postnet_n_convolutions=3)

        p_facade.load_state_dict(p_canonical.state_dict())

        x = torch.randn(2, 10, 80)
        mask = torch.tensor([[False]*8 + [True]*2, [False]*6 + [True]*4])

        p_canonical.eval()
        p_facade.eval()

        with torch.no_grad():
            out_c = p_canonical(x, mask)
            out_f = p_facade(x, mask)

        self.assertTrue(torch.equal(out_c, out_f))


if __name__ == "__main__":
    unittest.main()
