"""Unit tests for FastSpeech 2 Transformer Encoder and Decoder (`common.transformer.models`).

Covers all 16 requirements specified in Stage 2 — Step 1.4D:
1. Encoder initialization with an English vocabulary size.
2. Encoder initialization with a different Telugu vocabulary size.
3. Verify embedding dimensions and vocabulary sizes.
4. Verify invalid token IDs are rejected clearly.
5. Verify Encoder output shapes.
6. Verify variable-length batch handling.
7. Verify padded output positions.
8. Verify Decoder output shapes.
9. Verify Decoder maximum sequence length behaviour.
10. Verify positional encoding expansion.
11. Verify return_attns=False.
12. Verify return_attns=True returns actual attention tensors.
13. Verify CPU forward and backward passes.
14. Verify MPS forward and backward passes when available.
15. Verify configuration integration.
16. Verify invalid input shapes and sequence lengths.
"""

from __future__ import annotations

import unittest

import torch

from common.config import ModelConfig, TransformerConfig
from common.transformer import Decoder, Encoder
from common.utils import get_mask_from_lengths


def _small_transformer_config(**kwargs) -> TransformerConfig:
    """Helper creating a lightweight TransformerConfig for fast testing."""
    defaults = {
        "encoder_layer": 2,
        "encoder_head": 2,
        "encoder_hidden": 64,
        "decoder_layer": 2,
        "decoder_head": 2,
        "decoder_hidden": 64,
        "conv_filter_size": 128,
        "conv_kernel_size": [5, 1],
        "encoder_dropout": 0.1,
        "decoder_dropout": 0.1,
    }
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


class TestEncoderDecoderModels(unittest.TestCase):
    """Comprehensive verification of FastSpeech 2 Encoder and Decoder architectures."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_01_encoder_init_english_vocab(self) -> None:
        """1. Verify Encoder initialization with English vocabulary size (e.g. Svarah/ARPAbet)."""
        cfg = _small_transformer_config()
        enc_en = Encoder(cfg, n_src_vocab=88, max_seq_len=50)

        self.assertEqual(enc_en.n_src_vocab, 88)
        self.assertEqual(enc_en.src_word_emb.num_embeddings, 88)
        self.assertEqual(enc_en.src_word_emb.embedding_dim, 64)
        self.assertEqual(enc_en.d_model, 64)
        self.assertEqual(enc_en.n_layers, 2)
        self.assertEqual(len(enc_en.layer_stack), 2)
        self.assertEqual(enc_en.max_seq_len, 50)

    def test_02_encoder_init_telugu_vocab(self) -> None:
        """2. Verify Encoder initialization with distinct Telugu vocabulary size (e.g. IndicTTS)."""
        cfg = _small_transformer_config()
        enc_te = Encoder(cfg, n_src_vocab=64, max_seq_len=60)

        self.assertEqual(enc_te.n_src_vocab, 64)
        self.assertEqual(enc_te.src_word_emb.num_embeddings, 64)
        self.assertEqual(enc_te.src_word_emb.embedding_dim, 64)
        self.assertEqual(enc_te.max_seq_len, 60)

        # Confirm English and Telugu Encoders coexist in same process with distinct vocabulary sizes
        enc_en = Encoder(cfg, n_src_vocab=88, max_seq_len=50)
        self.assertNotEqual(enc_en.n_src_vocab, enc_te.n_src_vocab)
        self.assertNotEqual(
            enc_en.src_word_emb.num_embeddings,
            enc_te.src_word_emb.num_embeddings,
        )

    def test_03_embedding_dimensions_and_vocabulary_sizes(self) -> None:
        """3. Verify embedding tensor shapes, padding index, and padding initialization."""
        cfg = _small_transformer_config()
        encoder = Encoder(cfg, n_src_vocab=120)

        # Weight shape must match [n_src_vocab, d_model]
        self.assertEqual(tuple(encoder.src_word_emb.weight.shape), (120, 64))
        # Padding index must be 0 (Constants.PAD)
        self.assertEqual(encoder.src_word_emb.padding_idx, 0)
        # Position 0 in embedding weight is initialized to 0.0
        self.assertEqual(
            encoder.src_word_emb.weight[0].abs().max().item(), 0.0
        )
        # Positional encoding parameter shape must match [1, max_seq_len + 1, d_model]
        self.assertEqual(
            tuple(encoder.position_enc.shape), (1, encoder.max_seq_len + 1, 64)
        )
        self.assertFalse(encoder.position_enc.requires_grad)

    def test_04_invalid_token_ids_rejected(self) -> None:
        """4. Verify negative token IDs and out-of-range token IDs raise ValueError clearly."""
        cfg = _small_transformer_config()
        encoder = Encoder(cfg, n_src_vocab=80)

        # Valid range is [0, 79]
        valid_seq = torch.tensor([[0, 1, 10, 79]], dtype=torch.long)
        out = encoder(valid_seq)
        self.assertEqual(tuple(out.shape), (1, 4, 64))

        # Case A: Negative token ID
        bad_negative = torch.tensor([[0, -1, 10]], dtype=torch.long)
        with self.assertRaises(ValueError) as ctx_neg:
            encoder(bad_negative)
        self.assertIn("must be in range [0, 79]", str(ctx_neg.exception))

        # Case B: Out-of-bounds token ID (80 >= 80)
        bad_high = torch.tensor([[0, 80, 10]], dtype=torch.long)
        with self.assertRaises(ValueError) as ctx_high:
            encoder(bad_high)
        self.assertIn("must be in range [0, 79]", str(ctx_high.exception))

    def test_05_encoder_output_shapes(self) -> None:
        """5. Verify Encoder output shapes across different batch sizes and sequence lengths."""
        cfg = _small_transformer_config()
        encoder = Encoder(cfg, n_src_vocab=100)
        encoder.eval()

        for b, l in [(1, 5), (2, 8), (4, 12)]:
            src_seq = torch.randint(0, 100, (b, l), dtype=torch.long)
            out = encoder(src_seq)
            self.assertEqual(tuple(out.shape), (b, l, 64))
            self.assertFalse(torch.isnan(out).any().item())

    def test_06_variable_length_batch_handling(self) -> None:
        """6. Verify Encoder processes variable-length batches using padding masks."""
        cfg = _small_transformer_config()
        encoder = Encoder(cfg, n_src_vocab=50)
        encoder.eval()

        lengths = torch.tensor([8, 5, 2], dtype=torch.long)
        mask = get_mask_from_lengths(lengths)  # [3, 8], True = PAD
        src_seq = torch.randint(1, 50, (3, 8), dtype=torch.long)
        src_seq = src_seq.masked_fill(mask, 0)  # 0 is PAD

        out = encoder(src_seq, mask=mask)
        self.assertEqual(tuple(out.shape), (3, 8, 64))

        # Valid positions must contain finite, non-zero values
        self.assertGreater(out[0, :8].abs().max().item(), 0.0)
        self.assertGreater(out[1, :5].abs().max().item(), 0.0)
        self.assertGreater(out[2, :2].abs().max().item(), 0.0)

    def test_07_padded_output_positions(self) -> None:
        """7. Verify padded output positions in Encoder are strictly 0.0 and padding invariance holds."""
        cfg = _small_transformer_config(encoder_dropout=0.0)
        encoder = Encoder(cfg, n_src_vocab=50)
        encoder.eval()

        lengths = torch.tensor([6, 3], dtype=torch.long)
        mask = get_mask_from_lengths(lengths)
        src_seq = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]], dtype=torch.long
        )

        out = encoder(src_seq, mask=mask)
        # Padded positions in sample 1 (indices 3..5) must be exactly 0.0
        self.assertEqual(out[1, 3:].abs().max().item(), 0.0)

        # Padding invariance: evaluating sample 1 as an unpadded batch of length 3
        # must produce the exact same representations at positions 0..2
        single_seq = torch.tensor([[7, 8, 9]], dtype=torch.long)
        single_out = encoder(single_seq, mask=None)
        torch.testing.assert_close(
            out[1:2, :3, :], single_out, atol=1e-5, rtol=1e-5
        )

    def test_08_decoder_output_shapes(self) -> None:
        """8. Verify Decoder output shapes with and without mask."""
        cfg = _small_transformer_config()
        decoder = Decoder(cfg, max_seq_len=50)
        decoder.eval()

        # Without mask
        enc_seq = torch.randn(2, 10, 64)
        dec_out, ret_mask = decoder(enc_seq, mask=None)
        self.assertEqual(tuple(dec_out.shape), (2, 10, 64))
        self.assertIsNone(ret_mask)

        # With mask
        lengths = torch.tensor([10, 6], dtype=torch.long)
        mask = get_mask_from_lengths(lengths)
        dec_out_m, ret_mask_m = decoder(enc_seq, mask=mask)
        self.assertEqual(tuple(dec_out_m.shape), (2, 10, 64))
        self.assertIsNotNone(ret_mask_m)
        assert ret_mask_m is not None
        self.assertEqual(tuple(ret_mask_m.shape), (2, 10))
        # Padded positions in sample 1 must be 0.0
        self.assertEqual(dec_out_m[1, 6:].abs().max().item(), 0.0)

    def test_09_decoder_max_seq_len_truncation_in_training(self) -> None:
        """9. Verify Decoder truncates training sequences exceeding max_seq_len to max_seq_len."""
        cfg = _small_transformer_config()
        decoder = Decoder(cfg, max_seq_len=10)
        decoder.train()  # Training mode!

        # Input length 15 > max_seq_len (10)
        enc_seq = torch.randn(2, 15, 64)
        mask = torch.zeros(2, 15, dtype=torch.bool)
        mask[1, 12:] = True

        dec_out, ret_mask = decoder(enc_seq, mask=mask)

        # Output and mask must be clamped/truncated to max_seq_len = 10
        self.assertEqual(tuple(dec_out.shape), (2, 10, 64))
        self.assertIsNotNone(ret_mask)
        assert ret_mask is not None
        self.assertEqual(tuple(ret_mask.shape), (2, 10))
        self.assertEqual(ret_mask.shape[1], 10)

    def test_10_positional_encoding_expansion(self) -> None:
        """10. Verify dynamic positional encoding expansion during inference (eval mode) without truncation."""
        cfg = _small_transformer_config()
        encoder = Encoder(cfg, n_src_vocab=50, max_seq_len=10)
        decoder = Decoder(cfg, max_seq_len=10)
        encoder.eval()  # Eval mode!
        decoder.eval()  # Eval mode!

        # Test sequence length 25 > max_seq_len (10)
        src_seq = torch.randint(1, 50, (2, 25), dtype=torch.long)
        enc_out = encoder(src_seq)
        self.assertEqual(tuple(enc_out.shape), (2, 25, 64))

        enc_hidden = torch.randn(2, 25, 64)
        mask = torch.zeros(2, 25, dtype=torch.bool)
        dec_out, ret_mask = decoder(enc_hidden, mask=mask)

        # Full sequence length (25) is preserved (no truncation during inference)
        self.assertEqual(tuple(dec_out.shape), (2, 25, 64))
        self.assertIsNotNone(ret_mask)
        assert ret_mask is not None
        self.assertEqual(tuple(ret_mask.shape), (2, 25))

    def test_11_return_attns_false(self) -> None:
        """11. Verify return_attns=False returns tensor for Encoder and (tensor, mask) for Decoder."""
        cfg = _small_transformer_config()
        encoder = Encoder(cfg, n_src_vocab=50)
        decoder = Decoder(cfg)

        src_seq = torch.randint(1, 50, (2, 6), dtype=torch.long)
        enc_res = encoder(src_seq, return_attns=False)
        # Encoder must return a single Tensor
        self.assertIsInstance(enc_res, torch.Tensor)
        self.assertEqual(tuple(enc_res.shape), (2, 6, 64))

        dec_in = torch.randn(2, 6, 64)
        dec_res = decoder(dec_in, return_attns=False)
        # Decoder must return a 2-tuple (output, mask)
        self.assertIsInstance(dec_res, tuple)
        self.assertEqual(len(dec_res), 2)
        dec_out, _ = dec_res
        self.assertEqual(tuple(dec_out.shape), (2, 6, 64))

    def test_12_return_attns_true(self) -> None:
        """12. Verify return_attns=True returns actual, non-empty attention maps matching specification."""
        cfg = _small_transformer_config(encoder_layer=3, decoder_layer=4, encoder_head=2, decoder_head=2)
        encoder = Encoder(cfg, n_src_vocab=50)
        decoder = Decoder(cfg)
        encoder.eval()
        decoder.eval()

        src_seq = torch.randint(1, 50, (2, 5), dtype=torch.long)
        enc_res = encoder(src_seq, return_attns=True)
        self.assertIsInstance(enc_res, tuple)
        self.assertEqual(len(enc_res), 2)
        enc_out, enc_attns = enc_res
        self.assertEqual(tuple(enc_out.shape), (2, 5, 64))
        self.assertEqual(len(enc_attns), 3)  # 3 encoder layers
        for attn in enc_attns:
            self.assertEqual(tuple(attn.shape), (2 * 2, 5, 5))  # [n_head * B, L, L]
            torch.testing.assert_close(
                attn.sum(dim=-1), torch.ones(4, 5), atol=1e-5, rtol=1e-5
            )

        dec_in = torch.randn(2, 7, 64)
        dec_res = decoder(dec_in, return_attns=True)
        self.assertIsInstance(dec_res, tuple)
        self.assertEqual(len(dec_res), 3)
        dec_out, _, dec_attns = dec_res
        self.assertEqual(tuple(dec_out.shape), (2, 7, 64))
        self.assertEqual(len(dec_attns), 4)  # 4 decoder layers
        for attn in dec_attns:
            self.assertEqual(tuple(attn.shape), (2 * 2, 7, 7))
            torch.testing.assert_close(
                attn.sum(dim=-1), torch.ones(4, 7), atol=1e-5, rtol=1e-5
            )

    def test_13_cpu_forward_and_backward_passes(self) -> None:
        """13. Verify CPU forward and backward passes produce finite, non-zero gradients."""
        cfg = _small_transformer_config()
        encoder = Encoder(cfg, n_src_vocab=50)
        decoder = Decoder(cfg)
        encoder.train()
        decoder.train()

        # Encoder forward + backward
        src_seq = torch.randint(1, 50, (2, 6), dtype=torch.long)
        enc_out = encoder(src_seq)
        enc_loss = enc_out.pow(2).mean()
        enc_loss.backward()

        self.assertIsNotNone(encoder.src_word_emb.weight.grad)
        assert encoder.src_word_emb.weight.grad is not None
        self.assertFalse(
            torch.isnan(encoder.src_word_emb.weight.grad).any().item()
        )
        self.assertGreater(
            encoder.src_word_emb.weight.grad.abs().max().item(), 0.0
        )

        for name, p in encoder.layer_stack.named_parameters():
            self.assertIsNotNone(p.grad, f"Missing grad in {name}")
            assert p.grad is not None
            self.assertFalse(torch.isnan(p.grad).any().item())
            self.assertGreater(p.grad.abs().max().item(), 0.0)

        # Decoder forward + backward
        enc_hidden = torch.randn(2, 6, 64, requires_grad=True)
        dec_out, _ = decoder(enc_hidden)
        dec_loss = dec_out.pow(2).mean()
        dec_loss.backward()

        self.assertIsNotNone(enc_hidden.grad)
        assert enc_hidden.grad is not None
        self.assertFalse(torch.isnan(enc_hidden.grad).any().item())
        self.assertGreater(enc_hidden.grad.abs().max().item(), 0.0)

        for name, p in decoder.layer_stack.named_parameters():
            self.assertIsNotNone(p.grad, f"Missing grad in {name}")
            assert p.grad is not None
            self.assertFalse(torch.isnan(p.grad).any().item())
            self.assertGreater(p.grad.abs().max().item(), 0.0)

    def test_14_mps_forward_and_backward_passes(self) -> None:
        """14. Verify MPS forward and backward passes execute natively on Apple Silicon when available."""
        if not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            self.skipTest("Apple Silicon MPS backend is not available.")

        mps_dev = torch.device("mps")
        cfg = _small_transformer_config()
        encoder = Encoder(cfg, n_src_vocab=40).to(mps_dev)
        decoder = Decoder(cfg).to(mps_dev)
        encoder.train()
        decoder.train()

        # Encoder on MPS
        src_seq = torch.randint(1, 40, (2, 8), dtype=torch.long, device=mps_dev)
        mask = get_mask_from_lengths(
            torch.tensor([8, 5], dtype=torch.long, device=mps_dev)
        )
        enc_out = encoder(src_seq, mask=mask)
        self.assertEqual(enc_out.device.type, "mps")
        self.assertEqual(enc_out[1, 5:].abs().max().item(), 0.0)

        loss_enc = enc_out.sum()
        loss_enc.backward()
        self.assertIsNotNone(encoder.src_word_emb.weight.grad)
        assert encoder.src_word_emb.weight.grad is not None
        self.assertEqual(encoder.src_word_emb.weight.grad.device.type, "mps")
        self.assertFalse(
            torch.isnan(encoder.src_word_emb.weight.grad).any().item()
        )

        # Decoder on MPS
        enc_hidden = torch.randn(2, 8, 64, device=mps_dev, requires_grad=True)
        dec_out, ret_mask = decoder(enc_hidden, mask=mask)
        self.assertEqual(dec_out.device.type, "mps")
        assert ret_mask is not None
        self.assertEqual(ret_mask.device.type, "mps")
        self.assertEqual(dec_out[1, 5:].abs().max().item(), 0.0)

        loss_dec = dec_out.sum()
        loss_dec.backward()
        self.assertIsNotNone(enc_hidden.grad)
        assert enc_hidden.grad is not None
        self.assertEqual(enc_hidden.grad.device.type, "mps")
        self.assertFalse(torch.isnan(enc_hidden.grad).any().item())

    def test_15_configuration_integration(self) -> None:
        """15. Verify initialization from ModelConfig, TransformerConfig, and dictionary mappings."""
        # A. From ModelConfig
        m_cfg = ModelConfig(
            n_src_vocab=92,
            max_seq_len=800,
            transformer=TransformerConfig(
                encoder_layer=3,
                encoder_head=4,
                encoder_hidden=128,
                decoder_layer=5,
                decoder_head=4,
                decoder_hidden=128,
                conv_filter_size=512,
            ),
        )
        enc_m = Encoder.from_config(m_cfg)
        dec_m = Decoder.from_config(m_cfg)
        self.assertEqual(enc_m.n_src_vocab, 92)
        self.assertEqual(enc_m.max_seq_len, 800)
        self.assertEqual(enc_m.d_model, 128)
        self.assertEqual(enc_m.n_layers, 3)
        self.assertEqual(dec_m.max_seq_len, 800)
        self.assertEqual(dec_m.d_model, 128)
        self.assertEqual(dec_m.n_layers, 5)

        # B. From TransformerConfig with explicit n_src_vocab
        t_cfg = TransformerConfig(
            encoder_hidden=64,
            decoder_hidden=64,
            encoder_head=2,
            decoder_head=2,
        )
        enc_t = Encoder(t_cfg, n_src_vocab=55)
        dec_t = Decoder(t_cfg)
        self.assertEqual(enc_t.n_src_vocab, 55)
        self.assertEqual(dec_t.d_model, 64)

        # C. From nested dictionary
        dict_cfg = {
            "n_src_vocab": 77,
            "max_seq_len": 500,
            "transformer": {
                "encoder_layer": 2,
                "encoder_head": 2,
                "encoder_hidden": 64,
                "decoder_layer": 2,
                "decoder_head": 2,
                "decoder_hidden": 64,
                "conv_filter_size": 256,
            },
        }
        enc_d = Encoder(dict_cfg)
        dec_d = Decoder(dict_cfg)
        self.assertEqual(enc_d.n_src_vocab, 77)
        self.assertEqual(enc_d.max_seq_len, 500)
        self.assertEqual(dec_d.max_seq_len, 500)

        # Missing n_src_vocab raises ValueError
        with self.assertRaises(ValueError):
            Encoder(t_cfg)

    def test_16_invalid_input_shapes_and_sequence_lengths(self) -> None:
        """16. Verify invalid tensor ranks, empty inputs, feature mismatches, and training overflows raise ValueError."""
        cfg = _small_transformer_config()
        encoder = Encoder(cfg, n_src_vocab=50, max_seq_len=10)
        decoder = Decoder(cfg, max_seq_len=10)

        # Encoder input errors
        # 1. 1D input tensor
        with self.assertRaises(ValueError):
            encoder(torch.tensor([1, 2, 3], dtype=torch.long))
        # 2. 3D input tensor
        with self.assertRaises(ValueError):
            encoder(torch.randint(0, 50, (2, 3, 4), dtype=torch.long))
        # 3. Empty batch (B=0)
        with self.assertRaises(ValueError):
            encoder(torch.zeros((0, 5), dtype=torch.long))
        # 4. Empty sequence (L=0)
        with self.assertRaises(ValueError):
            encoder(torch.zeros((2, 0), dtype=torch.long))
        # 5. Mask shape mismatch
        with self.assertRaises(ValueError):
            encoder(
                torch.randint(0, 50, (2, 5), dtype=torch.long),
                mask=torch.zeros((2, 6), dtype=torch.bool),
            )
        # 6. Encoder sequence length overflow in training mode
        encoder.train()
        with self.assertRaises(ValueError) as ctx_enc_len:
            encoder(torch.randint(0, 50, (2, 12), dtype=torch.long))
        self.assertIn("exceeds max_seq_len", str(ctx_enc_len.exception))

        # Decoder input errors
        # 7. 2D input tensor
        with self.assertRaises(ValueError):
            decoder(torch.randn(2, 64))
        # 8. Empty batch (B=0)
        with self.assertRaises(ValueError):
            decoder(torch.randn(0, 5, 64))
        # 9. Empty sequence (L=0)
        with self.assertRaises(ValueError):
            decoder(torch.randn(2, 0, 64))
        # 10. Hidden dimension mismatch (128 != 64)
        with self.assertRaises(ValueError):
            decoder(torch.randn(2, 5, 128))
        # 11. Mask shape mismatch
        with self.assertRaises(ValueError):
            decoder(torch.randn(2, 5, 64), mask=torch.zeros((2, 6), dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()

