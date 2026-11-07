"""Unit tests for shared FastSpeech 2 Transformer building blocks (`common.transformer`).

Covers:
1. Positional encoding shape and values.
2. Positional encoding on CPU.
3. Positional encoding on MPS when available.
4. Multi-head attention output shapes.
5. Attention mask behaviour (`True = PAD`).
6. Single-token attention (`L = 1`).
7. Variable-length batches (using `get_mask_from_lengths`).
8. Padded query positions and padding invariance across Conv1D FFN (`k = 9`).
9. Invalid all-masked inputs and malformed input/mask rejection.
10. Position-wise feed-forward output shapes and odd kernel validation.
11. FFTBlock output shape, `mask=None` safety, and padded-position zeroing.
12. Forward and backward passes (finite non-zero gradients).
13. Different valid attention-head configurations (`n_head = 1, 2, 4, 8`).
14. Device and dtype consistency (CPU and MPS).
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from common.config import TransformerConfig
from common.transformer import (
    BOS,
    BOS_WORD,
    EOS,
    EOS_WORD,
    PAD,
    PAD_WORD,
    UNK,
    UNK_WORD,
    FFTBlock,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    ScaledDotProductAttention,
    get_sinusoid_encoding_table,
)
from common.utils import get_mask_from_lengths, to_device


class TestTransformerBuildingBlocks(unittest.TestCase):
    """Comprehensive verification of `common.transformer` building blocks."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_01_constants_and_positional_encoding_shape_and_values(self) -> None:
        """Verify Transformer constants and sinusoidal positional encoding formula and values."""
        self.assertEqual(PAD, 0)
        self.assertEqual(UNK, 1)
        self.assertEqual(BOS, 2)
        self.assertEqual(EOS, 3)
        self.assertEqual(PAD_WORD, "<blank>")
        self.assertEqual(UNK_WORD, "<unk>")
        self.assertEqual(BOS_WORD, "<s>")
        self.assertEqual(EOS_WORD, "</s>")

        n_pos, d_hid = 50, 256
        table = get_sinusoid_encoding_table(n_pos, d_hid)
        self.assertEqual(tuple(table.shape), (n_pos, d_hid))
        self.assertEqual(table.dtype, torch.float32)
        self.assertEqual(table.device.type, "cpu")
        self.assertFalse(torch.isnan(table).any().item())

        # Position 0: sin(0) = 0 on even dims, cos(0) = 1 on odd dims
        torch.testing.assert_close(
            table[0, 0::2], torch.zeros(d_hid // 2, dtype=torch.float32)
        )
        torch.testing.assert_close(
            table[0, 1::2], torch.ones(d_hid // 2, dtype=torch.float32)
        )

        # Spot-check arbitrary position pos=7, dim=4 (even) and dim=5 (odd)
        angle_7_4 = 7.0 / math.pow(10000.0, 2.0 * (4 // 2) / d_hid)
        self.assertAlmostEqual(
            table[7, 4].item(), math.sin(angle_7_4), places=6
        )
        self.assertAlmostEqual(
            table[7, 5].item(), math.cos(angle_7_4), places=6
        )

        # Verify padding_idx zeroes out the specified position
        padded_table = get_sinusoid_encoding_table(n_pos, d_hid, padding_idx=0)
        self.assertEqual(padded_table[0].abs().max().item(), 0.0)
        self.assertGreater(padded_table[1].abs().max().item(), 0.0)

        # Invalid n_position, d_hid, or padding_idx raise ValueError
        with self.assertRaises(ValueError):
            get_sinusoid_encoding_table(0, 256)
        with self.assertRaises(ValueError):
            get_sinusoid_encoding_table(10, 0)
        with self.assertRaises(ValueError):
            get_sinusoid_encoding_table(10, 16, padding_idx=10)

    def test_02_positional_encoding_on_cpu_and_dynamic_expansion(self) -> None:
        """Verify PositionalEncoding on CPU within max_seq_len and when seq_len > max_seq_len."""
        pos_enc = PositionalEncoding(d_model=256, max_seq_len=20)
        self.assertEqual(tuple(pos_enc.position_enc.shape), (1, 21, 256))
        self.assertFalse(pos_enc.position_enc.requires_grad)

        # Within max_seq_len (L = 12 <= 21)
        x_short = torch.zeros(2, 12, 256, dtype=torch.float32)
        out_short = pos_enc(x_short)
        self.assertEqual(tuple(out_short.shape), (2, 12, 256))
        self.assertEqual(out_short.device.type, "cpu")
        torch.testing.assert_close(
            out_short[0], get_sinusoid_encoding_table(12, 256)
        )

        # Exceeding max_seq_len (L = 35 > 21) triggers dynamic sinusoidal table expansion
        x_long = torch.zeros(2, 35, 256, dtype=torch.float32)
        out_long = pos_enc(x_long)
        self.assertEqual(tuple(out_long.shape), (2, 35, 256))
        torch.testing.assert_close(
            out_long[1], get_sinusoid_encoding_table(35, 256)
        )

    def test_03_positional_encoding_and_transformer_blocks_on_mps(self) -> None:
        """Execute positional encoding, MHA, FFN, and FFTBlock forward/backward on MPS when available."""
        if not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            self.skipTest("Apple Silicon MPS backend is not available.")

        mps_device = torch.device("mps")

        # 1. get_sinusoid_encoding_table directly on MPS
        table_mps = get_sinusoid_encoding_table(32, 256, device=mps_device)
        table_cpu = get_sinusoid_encoding_table(32, 256, device="cpu")
        self.assertEqual(table_mps.device.type, "mps")
        torch.testing.assert_close(table_mps.cpu(), table_cpu)

        # 2. PositionalEncoding module on MPS (both normal and dynamic L > max_seq_len)
        pos_enc = PositionalEncoding(d_model=256, max_seq_len=16).to(mps_device)
        x_mps = torch.randn(2, 24, 256, device=mps_device)
        out_pos = pos_enc(x_mps)
        self.assertEqual(out_pos.device.type, "mps")
        self.assertEqual(tuple(out_pos.shape), (2, 24, 256))

        # 3. Full FFTBlock forward and backward on MPS with variable-length mask
        cfg = TransformerConfig()
        fft_block = FFTBlock.from_config(cfg).to(mps_device)
        fft_block.eval()

        lengths = torch.tensor([6, 3], dtype=torch.long, device=mps_device)
        mask = get_mask_from_lengths(lengths)  # [2, 6] on mps
        slf_attn_mask = mask.unsqueeze(1).expand(-1, 6, -1)  # [2, 6, 6] on mps

        enc_in = torch.randn(
            2, 6, 256, device=mps_device, requires_grad=True
        ).masked_fill(mask.unsqueeze(-1), 0.0)
        enc_in.retain_grad()

        enc_out, attn = fft_block(
            enc_in, mask=mask, slf_attn_mask=slf_attn_mask
        )
        self.assertEqual(enc_out.device.type, "mps")
        self.assertEqual(attn.device.type, "mps")
        self.assertEqual(tuple(enc_out.shape), (2, 6, 256))
        self.assertEqual(tuple(attn.shape), (4, 6, 6))
        self.assertEqual(enc_out[1, 3:].abs().max().item(), 0.0)

        loss = enc_out.sum()
        loss.backward()
        self.assertIsNotNone(enc_in.grad)
        assert enc_in.grad is not None
        self.assertFalse(torch.isnan(enc_in.grad).any().item())

    def test_04_multi_head_attention_output_shapes_and_mask_none(self) -> None:
        """Verify MultiHeadAttention output and attention shapes for self-attention, cross-length, and mask=None."""
        mha = MultiHeadAttention(
            n_head=2, d_model=256, d_k=128, d_v=128, dropout=0.0
        )
        mha.eval()

        # Self-attention with mask=None (DEC-07 / B-07)
        x = torch.randn(3, 10, 256)
        out, attn = mha(x, x, x, mask=None)
        self.assertEqual(tuple(out.shape), (3, 10, 256))
        self.assertEqual(tuple(attn.shape), (2 * 3, 10, 10))
        # Softmax weights along key dimension (dim=2) must sum to 1.0
        torch.testing.assert_close(
            attn.sum(dim=-1), torch.ones(6, 10), atol=1e-5, rtol=1e-5
        )

        # Distinct query and key/value lengths (L_q = 5, L_k = L_v = 8)
        q = torch.randn(2, 5, 256)
        kv = torch.randn(2, 8, 256)
        cross_mask = torch.zeros(2, 5, 8, dtype=torch.bool)
        cross_mask[1, :, 6:] = True
        out_c, attn_c = mha(q, kv, kv, mask=cross_mask)
        self.assertEqual(tuple(out_c.shape), (2, 5, 256))
        self.assertEqual(tuple(attn_c.shape), (4, 5, 8))

    def test_05_attention_mask_behaviour_true_is_pad(self) -> None:
        """Verify that key positions where mask == True receive strictly 0.0 attention weight."""
        mha = MultiHeadAttention(
            n_head=2, d_model=256, d_k=128, d_v=128, dropout=0.0
        )
        mha.eval()

        lengths = torch.tensor([5, 3, 2], dtype=torch.long)
        src_mask = get_mask_from_lengths(lengths)  # [3, 5], True = PAD
        slf_attn_mask = src_mask.unsqueeze(1).expand(-1, 5, -1)  # [3, 5, 5]

        x = torch.randn(3, 5, 256)
        _, attn = mha(x, x, x, mask=slf_attn_mask)  # [6, 5, 5]

        # Head 0 is indices [0, 1, 2]; Head 1 is indices [3, 4, 5]
        for head_idx in range(2):
            base = head_idx * 3
            # Sample 0 (len=5): all keys valid, weights > 0
            self.assertGreater(attn[base + 0].min().item(), 0.0)
            # Sample 1 (len=3): keys 3, 4 must be strictly 0.0
            self.assertEqual(attn[base + 1, :, 3:].abs().max().item(), 0.0)
            # Sample 2 (len=2): keys 2, 3, 4 must be strictly 0.0
            self.assertEqual(attn[base + 2, :, 2:].abs().max().item(), 0.0)

        # Unmasked key weights still sum to 1.0 across every row
        torch.testing.assert_close(
            attn.sum(dim=-1), torch.ones(6, 5), atol=1e-5, rtol=1e-5
        )

    def test_06_single_token_attention_and_fft_block(self) -> None:
        """Verify single-token sequence (L = 1) through MultiHeadAttention, FFN, and FFTBlock."""
        cfg = TransformerConfig()
        fft_block = FFTBlock.from_config(cfg)
        fft_block.eval()

        lengths = torch.tensor([1, 1], dtype=torch.long)
        mask = get_mask_from_lengths(lengths)  # [[False], [False]], shape [2, 1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, 1, -1)  # [2, 1, 1]

        x = torch.randn(2, 1, 256)
        out, attn = fft_block(x, mask=mask, slf_attn_mask=slf_attn_mask)

        self.assertEqual(tuple(out.shape), (2, 1, 256))
        self.assertEqual(tuple(attn.shape), (4, 1, 1))
        self.assertFalse(torch.isnan(out).any().item())
        # With only 1 unmasked key, attention weight must be 1.0 everywhere
        torch.testing.assert_close(attn, torch.ones(4, 1, 1))

    def test_07_variable_length_batches_and_padded_query_positions(self) -> None:
        """Verify variable-length batches keep padded positions strictly zero and prevent Conv1d(k=9) bleed."""
        fft_block = FFTBlock(
            d_model=256,
            n_head=2,
            d_k=128,
            d_v=128,
            d_inner=1024,
            kernel_size=[9, 1],
            dropout=0.0,
        )
        fft_block.eval()

        # Batch of 3 utterances with lengths [7, 4, 2]
        lengths = torch.tensor([7, 4, 2], dtype=torch.long)
        mask = get_mask_from_lengths(lengths)  # [3, 7]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, 7, -1)  # [3, 7, 7]

        x = torch.randn(3, 7, 256)
        # Put large arbitrary noise in padded positions of x; after masking input as Encoder does:
        x_masked = x.masked_fill(mask.unsqueeze(-1), 0.0)

        out, _ = fft_block(x_masked, mask=mask, slf_attn_mask=slf_attn_mask)

        # Padded positions must be strictly 0.0
        self.assertEqual(out[1, 4:].abs().max().item(), 0.0)
        self.assertEqual(out[2, 2:].abs().max().item(), 0.0)
        # Valid positions must be non-zero
        self.assertGreater(out[0, :7].abs().max().item(), 0.0)
        self.assertGreater(out[1, :4].abs().max().item(), 0.0)
        self.assertGreater(out[2, :2].abs().max().item(), 0.0)

        # Padding invariance check: running sample 2 alone at length L=2 without padding
        # must produce the exact same valid hidden states as running it in the padded L=7 batch!
        x_single = x_masked[2:3, :2, :]  # [1, 2, 256]
        mask_single = torch.zeros(1, 2, dtype=torch.bool)
        slf_attn_single = torch.zeros(1, 2, 2, dtype=torch.bool)
        out_single, _ = fft_block(
            x_single, mask=mask_single, slf_attn_mask=slf_attn_single
        )
        torch.testing.assert_close(
            out[2:3, :2, :], out_single, atol=1e-5, rtol=1e-5
        )

    def test_08_all_masked_and_invalid_inputs_handling(self) -> None:
        """Verify DEC-08 / B-06: all-masked key rows produce 0.0 (no NaNs in forward or backward), and invalid shapes raise ValueError."""
        mha = MultiHeadAttention(
            n_head=2, d_model=256, d_k=128, d_v=128, dropout=0.0
        )
        fft_block = FFTBlock(
            d_model=256,
            n_head=2,
            d_k=128,
            d_v=128,
            d_inner=1024,
            kernel_size=[9, 1],
            dropout=0.0,
        )

        # Construct a 2-sample batch where sample 1 has ALL positions masked (True)
        mask = torch.tensor(
            [[False, False, True], [True, True, True]], dtype=torch.bool
        )
        slf_attn_mask = mask.unsqueeze(1).expand(-1, 3, -1)  # [2, 3, 3]

        x = torch.randn(2, 3, 256, requires_grad=True)
        out, attn = fft_block(x, mask=mask, slf_attn_mask=slf_attn_mask)

        # Forward outputs and attention weights must be completely free of NaNs
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isnan(attn).any().item())
        # All-masked sample 1 has strictly 0.0 attention weights and 0.0 output
        self.assertEqual(attn[1].abs().max().item(), 0.0)
        self.assertEqual(attn[3].abs().max().item(), 0.0)
        self.assertEqual(out[1].abs().max().item(), 0.0)

        # Backward pass must also be completely free of NaNs (SoftmaxBackward0 safety)
        loss = out.sum() + attn.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any().item())

        # Invalid / mismatched mask shapes or empty inputs must raise ValueError
        with self.assertRaises(ValueError):
            mha(
                torch.randn(2, 4, 256),
                torch.randn(2, 4, 256),
                torch.randn(2, 4, 256),
                mask=torch.zeros(2, 3, 4, dtype=torch.bool),
            )
        with self.assertRaises(ValueError):
            mha(
                torch.randn(2, 0, 256),
                torch.randn(2, 0, 256),
                torch.randn(2, 0, 256),
            )
        with self.assertRaises(ValueError):
            fft_block(
                torch.randn(2, 4, 256),
                mask=torch.zeros(2, 5, dtype=torch.bool),
            )
        with self.assertRaises(ValueError):
            ScaledDotProductAttention(temperature=0.0)

    def test_09_positionwise_feed_forward_shapes_and_kernel_validation(self) -> None:
        """Verify PositionwiseFeedForward preserves sequence length for odd kernels and rejects even kernels."""
        for k_pair in ([9, 1], [5, 3], [1, 1]):
            ffn = PositionwiseFeedForward(
                d_in=256, d_hid=1024, kernel_size=k_pair, dropout=0.1
            )
            x = torch.randn(2, 15, 256)
            out = ffn(x)
            self.assertEqual(tuple(out.shape), (2, 15, 256))

        # Even kernel sizes must be rejected in __init__
        for bad_kernels in ([8, 1], [9, 2], [0, 1], [9]):
            with self.assertRaises(ValueError):
                PositionwiseFeedForward(
                    d_in=256, d_hid=1024, kernel_size=bad_kernels
                )

    def test_10_fft_block_mask_none_and_forward_backward_gradients(self) -> None:
        """Verify FFTBlock works with mask=None and produces finite gradients for all parameters."""
        fft_block = FFTBlock(
            d_model=256,
            n_head=2,
            d_k=128,
            d_v=128,
            d_inner=1024,
            kernel_size=[9, 1],
            dropout=0.1,
        )
        fft_block.train()

        # Call with default mask=None and slf_attn_mask=None (DEC-07 / B-07)
        x = torch.randn(2, 8, 256, requires_grad=True)
        out, attn = fft_block(x)
        self.assertEqual(tuple(out.shape), (2, 8, 256))
        self.assertEqual(tuple(attn.shape), (4, 8, 8))

        loss = out.pow(2).mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any().item())
        self.assertGreater(x.grad.abs().max().item(), 0.0)

        for name, param in fft_block.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
            assert param.grad is not None
            self.assertFalse(
                torch.isnan(param.grad).any().item(),
                f"NaN gradient in {name}",
            )
            self.assertGreater(
                param.grad.abs().max().item(),
                0.0,
                f"Zero gradient in {name}",
            )

    def test_11_different_valid_attention_head_configurations(self) -> None:
        """Verify FFTBlock and MultiHeadAttention across diverse valid (d_model, n_head) configurations."""
        for d_model, n_head, ffn_dim in [
            (128, 1, 512),
            (256, 2, 1024),
            (256, 4, 1024),
            (384, 6, 1536),
            (512, 8, 2048),
        ]:
            t_cfg = TransformerConfig(
                encoder_hidden=d_model,
                decoder_hidden=d_model,
                encoder_head=n_head,
                decoder_head=n_head,
                conv_filter_size=ffn_dim,
                conv_kernel_size=[9, 1],
            )
            block = FFTBlock.from_config(t_cfg)
            x = torch.randn(2, 6, d_model)
            out, attn = block(x)
            self.assertEqual(tuple(out.shape), (2, 6, d_model))
            self.assertEqual(tuple(attn.shape), (n_head * 2, 6, 6))

    def test_12_device_and_dtype_consistency(self) -> None:
        """Verify dtype preservation (float32 and float64) and integration with `to_device`."""
        for dtype in (torch.float32, torch.float64):
            block = FFTBlock(
                d_model=64,
                n_head=2,
                d_k=32,
                d_v=32,
                d_inner=128,
                kernel_size=[5, 1],
                dropout=0.0,
            ).to(dtype=dtype)
            pos_enc = PositionalEncoding(d_model=64, max_seq_len=20).to(
                dtype=dtype
            )

            x = to_device(torch.randn(2, 5, 64, dtype=dtype), "cpu")
            lengths = to_device(torch.tensor([5, 3], dtype=torch.long), "cpu")
            mask = get_mask_from_lengths(lengths)
            slf_attn_mask = mask.unsqueeze(1).expand(-1, 5, -1)

            x_pos = pos_enc(x).masked_fill(mask.unsqueeze(-1), 0.0)
            out, attn = block(x_pos, mask=mask, slf_attn_mask=slf_attn_mask)

            self.assertEqual(out.dtype, dtype)
            self.assertEqual(attn.dtype, dtype)
            self.assertEqual(out.device.type, "cpu")
            self.assertEqual(attn.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()

