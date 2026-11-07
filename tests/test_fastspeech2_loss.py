"""Unit and regression tests for FastSpeech2Loss (common.model.loss)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from common.config.preprocess import PreprocessConfig
from common.model.fastspeech2 import FastSpeech2
from common.model.loss import FastSpeech2Loss
from common.utils.tools import get_mask_from_lengths


def _make_dummy_configs(
    pitch_level: str = "phoneme_level",
    energy_level: str = "phoneme_level",
    n_mel_channels: int = 80,
) -> tuple[dict, dict]:
    """Create minimal preprocess_config and model_config dictionaries."""
    preprocess_config = {
        "dataset": "dummy",
        "path": {
            "corpus_path": "/tmp/corpus",
            "raw_path": "/tmp/raw",
            "preprocessed_path": "/tmp/preprocessed",
        },
        "preprocessing": {
            "text": {"language": "en", "text_cleaners": ["english_cleaners"]},
            "val_size": 512,
            "audio": {"sampling_rate": 22050, "max_wav_value": 32768.0},
            "stft": {"filter_length": 1024, "hop_length": 256, "win_length": 1024},
            "mel": {
                "n_mel_channels": n_mel_channels,
                "mel_fmin": 0.0,
                "mel_fmax": 8000.0,
            },
            "pitch": {"feature": pitch_level, "normalization": True},
            "energy": {"feature": energy_level, "normalization": True},
        },
    }
    model_config = {
        "transformer": {
            "encoder_layer": 2,
            "encoder_head": 2,
            "encoder_hidden": 64,
            "decoder_layer": 2,
            "decoder_head": 2,
            "decoder_hidden": 64,
            "fft_conv1d_filter_size": 128,
            "fft_conv1d_kernel_size": [9, 1],
            "dropout": 0.1,
        },
        "variance_predictor": {
            "filter_size": 64,
            "kernel_size": 3,
            "dropout": 0.1,
        },
        "variance_embedding": {
            "n_bins": 16,
        },
        "max_seq_len": 100,
        "multi_speaker": False,
        "n_src_vocab": 50,
    }
    return preprocess_config, model_config


def _build_synthetic_batch_and_predictions(
    batch_size: int = 2,
    src_lens: list[int] | None = None,
    mel_lens: list[int] | None = None,
    n_mels: int = 80,
    pitch_level: str = "phoneme_level",
    energy_level: str = "phoneme_level",
    target_mel_extra_len: int = 0,
    seed: int = 42,
    device: str = "cpu",
):
    """Build a synthetic (inputs, predictions) pair for testing."""
    torch.manual_seed(seed)
    if src_lens is None:
        src_lens = [5, 4]
    if mel_lens is None:
        mel_lens = [12, 10]

    max_src = max(src_lens)
    max_mel = max(mel_lens)
    src_lens_t = torch.tensor(src_lens, dtype=torch.long, device=device)
    mel_lens_t = torch.tensor(mel_lens, dtype=torch.long, device=device)

    src_masks = get_mask_from_lengths(src_lens_t, max_src)  # True = PAD
    mel_masks = get_mask_from_lengths(mel_lens_t, max_mel)  # True = PAD

    # Model predictions
    mel_pred = torch.randn(batch_size, max_mel, n_mels, device=device, requires_grad=True)
    postnet_mel_pred = torch.randn(
        batch_size, max_mel, n_mels, device=device, requires_grad=True
    )
    log_d_pred = torch.randn(batch_size, max_src, device=device, requires_grad=True)

    if pitch_level == "phoneme_level":
        pitch_pred = torch.randn(batch_size, max_src, device=device, requires_grad=True)
    else:
        pitch_pred = torch.randn(batch_size, max_mel, device=device, requires_grad=True)

    if energy_level == "phoneme_level":
        energy_pred = torch.randn(batch_size, max_src, device=device, requires_grad=True)
    else:
        energy_pred = torch.randn(batch_size, max_mel, device=device, requires_grad=True)

    # Targets (potentially with extra frames for DEC-10 testing)
    tgt_mel_len = max_mel + target_mel_extra_len
    mel_targets = torch.randn(batch_size, tgt_mel_len, n_mels, device=device)

    # Synthetic integer durations that sum up to mel_lens for valid tokens
    duration_targets = torch.zeros(batch_size, max_src, dtype=torch.long, device=device)
    for b in range(batch_size):
        v_tokens = src_lens[b]
        target_frames = mel_lens[b]
        # Distribute frames
        base = target_frames // v_tokens
        rem = target_frames % v_tokens
        for i in range(v_tokens):
            duration_targets[b, i] = base + (1 if i < rem else 0)

    if pitch_level == "phoneme_level":
        pitch_targets = torch.randn(batch_size, max_src, device=device)
    else:
        pitch_targets = torch.randn(batch_size, tgt_mel_len, device=device)

    if energy_level == "phoneme_level":
        energy_targets = torch.randn(batch_size, max_src, device=device)
    else:
        energy_targets = torch.randn(batch_size, tgt_mel_len, device=device)

    # 12-element batch tuple matching to_device output
    batch_tuple = (
        [f"id_{i}" for i in range(batch_size)],
        [f"text_{i}" for i in range(batch_size)],
        None,  # speakers
        torch.randint(1, 40, (batch_size, max_src), device=device),  # texts
        src_lens_t,
        max_src,
        mel_targets,
        mel_lens_t,
        tgt_mel_len,
        pitch_targets,
        energy_targets,
        duration_targets,
    )

    # 10-element predictions tuple matching FastSpeech2.forward
    pred_tuple = (
        mel_pred,
        postnet_mel_pred,
        pitch_pred,
        energy_pred,
        log_d_pred,
        duration_targets,  # d_rounded
        src_masks,
        mel_masks,
        src_lens_t,
        mel_lens_t,
    )

    return batch_tuple, pred_tuple


class FastSpeech2LossTest(unittest.TestCase):
    """Test suite for FastSpeech2Loss."""

    def test_01_initialization_modes(self):
        """Verify initialization with config dicts and PreprocessConfig objects."""
        p_cfg, m_cfg = _make_dummy_configs(
            pitch_level="phoneme_level", energy_level="frame_level"
        )
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg)
        self.assertEqual(loss_fn.pitch_feature_level, "phoneme_level")
        self.assertEqual(loss_fn.energy_feature_level, "frame_level")

        # Via PreprocessConfig dataclass
        prep_obj = PreprocessConfig.from_dict(p_cfg)
        loss_fn2 = FastSpeech2Loss(prep_obj)
        self.assertEqual(loss_fn2.pitch_feature_level, "phoneme_level")
        self.assertEqual(loss_fn2.energy_feature_level, "frame_level")

        # Invalid feature level
        invalid_cfg = {"preprocessing": {"pitch": {"feature": "invalid"}, "energy": {"feature": "frame_level"}}}
        with self.assertRaises(ValueError):
            FastSpeech2Loss(invalid_cfg)

    def test_02_individual_loss_components_and_manual_parity(self):
        """Verify each loss component matches exact manual PyTorch calculations."""
        p_cfg, m_cfg = _make_dummy_configs("phoneme_level", "phoneme_level")
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg)

        batch, preds = _build_synthetic_batch_and_predictions(
            batch_size=2,
            src_lens=[4, 3],
            mel_lens=[10, 8],
            pitch_level="phoneme_level",
            energy_level="phoneme_level",
        )

        losses = loss_fn(batch, preds)
        total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss = (
            losses
        )

        # Unpack predictions and batch
        mel_pred, postnet_mel_pred, pitch_pred, energy_pred, log_d_pred = preds[:5]
        src_masks, mel_masks = preds[6], preds[7]
        mel_targets = batch[6]
        pitch_targets = batch[9]
        energy_targets = batch[10]
        duration_targets = batch[11]

        valid_src = ~src_masks
        valid_mel = ~mel_masks

        # Manual L1 mel loss
        valid_mel_3d = valid_mel.unsqueeze(-1).expand_as(mel_pred)
        manual_mel_loss = torch.abs(
            mel_pred[valid_mel_3d] - mel_targets[valid_mel_3d]
        ).mean()
        manual_postnet_loss = torch.abs(
            postnet_mel_pred[valid_mel_3d] - mel_targets[valid_mel_3d]
        ).mean()

        # Manual MSE pitch and energy losses
        manual_pitch_loss = (
            (pitch_pred[valid_src] - pitch_targets[valid_src]) ** 2
        ).mean()
        manual_energy_loss = (
            (energy_pred[valid_src] - energy_targets[valid_src]) ** 2
        ).mean()

        # Manual MSE duration loss
        log_d_target = torch.log(duration_targets[valid_src].float() + 1.0)
        manual_dur_loss = ((log_d_pred[valid_src] - log_d_target) ** 2).mean()

        self.assertAlmostEqual(mel_loss.item(), manual_mel_loss.item(), places=5)
        self.assertAlmostEqual(
            postnet_mel_loss.item(), manual_postnet_loss.item(), places=5
        )
        self.assertAlmostEqual(pitch_loss.item(), manual_pitch_loss.item(), places=5)
        self.assertAlmostEqual(energy_loss.item(), manual_energy_loss.item(), places=5)
        self.assertAlmostEqual(
            duration_loss.item(), manual_dur_loss.item(), places=5
        )

        # Total loss unweighted sum
        manual_total = (
            manual_mel_loss
            + manual_postnet_loss
            + manual_pitch_loss
            + manual_energy_loss
            + manual_dur_loss
        )
        self.assertAlmostEqual(total_loss.item(), manual_total.item(), places=5)

    def test_03_frame_level_features(self):
        """Verify loss computation with frame-level pitch and energy."""
        p_cfg, m_cfg = _make_dummy_configs("frame_level", "frame_level")
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg)

        batch, preds = _build_synthetic_batch_and_predictions(
            batch_size=2,
            src_lens=[4, 3],
            mel_lens=[10, 8],
            pitch_level="frame_level",
            energy_level="frame_level",
        )

        losses = loss_fn(batch, preds)
        total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss = (
            losses
        )

        valid_mel = ~preds[7]
        pitch_pred = preds[2]
        pitch_targets = batch[9]
        manual_pitch_loss = (
            (pitch_pred[valid_mel] - pitch_targets[valid_mel]) ** 2
        ).mean()
        self.assertAlmostEqual(pitch_loss.item(), manual_pitch_loss.item(), places=5)

    def test_04_dec10_temporal_truncation(self):
        """DEC-10 verification: target tensors longer than mel_masks do not cause dimension mismatch."""
        p_cfg, m_cfg = _make_dummy_configs("frame_level", "frame_level")
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg)

        # Ground truth has 5 extra frames (target_mel_extra_len=5)
        batch, preds = _build_synthetic_batch_and_predictions(
            batch_size=2,
            src_lens=[5, 4],
            mel_lens=[15, 12],
            pitch_level="frame_level",
            energy_level="frame_level",
            target_mel_extra_len=5,
        )

        # Target length is 20, prediction length is 15
        self.assertEqual(batch[6].shape[1], 20)
        self.assertEqual(preds[0].shape[1], 15)
        self.assertEqual(batch[9].shape[1], 20)
        self.assertEqual(preds[2].shape[1], 15)

        # Must execute cleanly without RuntimeError
        losses = loss_fn(batch, preds)
        self.assertEqual(len(losses), 6)
        for l in losses:
            self.assertTrue(torch.isfinite(l))

    def test_05_padding_invariance(self):
        """Verify that modifying padded regions of targets or predictions has ZERO effect on any loss."""
        p_cfg, m_cfg = _make_dummy_configs("phoneme_level", "frame_level")
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg)

        batch, preds = _build_synthetic_batch_and_predictions(
            batch_size=2,
            src_lens=[4, 2],
            mel_lens=[12, 6],
            pitch_level="phoneme_level",
            energy_level="frame_level",
        )

        losses_orig = loss_fn(batch, preds)

        # Corrupt padded regions in predictions and targets
        src_masks = preds[6]
        mel_masks = preds[7]

        # Clone and corrupt predictions
        mel_pred_c = preds[0].clone()
        postnet_mel_pred_c = preds[1].clone()
        pitch_pred_c = preds[2].clone()
        energy_pred_c = preds[3].clone()
        log_d_pred_c = preds[4].clone()

        mel_pred_c[mel_masks.unsqueeze(-1).expand_as(mel_pred_c)] = 9999.0
        postnet_mel_pred_c[mel_masks.unsqueeze(-1).expand_as(postnet_mel_pred_c)] = -8888.0
        pitch_pred_c[src_masks] = 5555.0
        energy_pred_c[mel_masks] = -4444.0
        log_d_pred_c[src_masks] = 3333.0

        corrupted_preds = (
            mel_pred_c,
            postnet_mel_pred_c,
            pitch_pred_c,
            energy_pred_c,
            log_d_pred_c,
            preds[5],
            preds[6],
            preds[7],
            preds[8],
            preds[9],
        )

        # Clone and corrupt batch targets
        mel_targets_c = batch[6].clone()
        pitch_targets_c = batch[9].clone()
        energy_targets_c = batch[10].clone()
        duration_targets_c = batch[11].clone()

        mel_targets_c[mel_masks.unsqueeze(-1).expand_as(mel_targets_c)] = -7777.0
        pitch_targets_c[src_masks] = 2222.0
        energy_targets_c[mel_masks] = 1111.0
        duration_targets_c[src_masks] = 99  # Valid non-negative integer in padding

        corrupted_batch = list(batch)
        corrupted_batch[6] = mel_targets_c
        corrupted_batch[9] = pitch_targets_c
        corrupted_batch[10] = energy_targets_c
        corrupted_batch[11] = duration_targets_c

        losses_corrupted = loss_fn(corrupted_batch, corrupted_preds)

        for name, orig, corrupt in zip(
            ["total", "mel", "postnet", "pitch", "energy", "duration"],
            losses_orig,
            losses_corrupted,
        ):
            self.assertEqual(
                orig.item(),
                corrupt.item(),
                f"Loss term {name} changed after corrupting padded positions! "
                f"Orig: {orig.item()}, Corrupted: {corrupt.item()}",
            )

    def test_06_gradient_propagation_cpu(self):
        """Verify backpropagation produces valid finite gradients on predictions."""
        p_cfg, m_cfg = _make_dummy_configs("phoneme_level", "phoneme_level")
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg)

        batch, preds = _build_synthetic_batch_and_predictions(
            batch_size=2,
            src_lens=[5, 3],
            mel_lens=[10, 7],
            pitch_level="phoneme_level",
            energy_level="phoneme_level",
        )

        total_loss = loss_fn(batch, preds)[0]
        total_loss.backward()

        for idx, name in enumerate(["mel", "postnet", "pitch", "energy", "log_d"]):
            tensor = preds[idx]
            self.assertIsNotNone(tensor.grad, f"grad is None for {name}")
            self.assertTrue(
                torch.isfinite(tensor.grad).all(),
                f"grad contains non-finite values for {name}",
            )
            # Padded regions must have zero gradient
            mask = preds[6] if idx in (2, 3, 4) else preds[7]
            if tensor.dim() == 3:
                mask_3d = mask.unsqueeze(-1).expand_as(tensor)
                self.assertTrue(
                    (tensor.grad[mask_3d] == 0.0).all(),
                    f"Padded gradient non-zero for {name}",
                )
            else:
                self.assertTrue(
                    (tensor.grad[mask] == 0.0).all(),
                    f"Padded gradient non-zero for {name}",
                )

    def test_07_mps_device_compatibility(self):
        """Verify forward and backward pass on Apple Silicon MPS if available."""
        if not torch.backends.mps.is_available():
            self.skipTest("Apple Silicon MPS is not available on this machine.")

        device = "mps"
        p_cfg, m_cfg = _make_dummy_configs("phoneme_level", "frame_level")
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg).to(device)

        batch, preds = _build_synthetic_batch_and_predictions(
            batch_size=2,
            src_lens=[5, 4],
            mel_lens=[12, 9],
            pitch_level="phoneme_level",
            energy_level="frame_level",
            device=device,
        )

        losses = loss_fn(batch, preds)
        self.assertEqual(len(losses), 6)
        for l in losses:
            self.assertEqual(l.device.type, "mps")
            self.assertTrue(torch.isfinite(l).item())

        total_loss = losses[0]
        total_loss.backward()

        for idx, name in enumerate(["mel", "postnet", "pitch", "energy", "log_d"]):
            t = preds[idx]
            self.assertIsNotNone(t.grad, f"grad is None for {name} on MPS")
            self.assertTrue(
                torch.isfinite(t.grad).all().item(),
                f"grad is non-finite for {name} on MPS",
            )

    def test_08_input_validation_and_empty_selections(self):
        """Verify proper ValueErrors for invalid inputs, negative durations, and empty masks."""
        p_cfg, m_cfg = _make_dummy_configs()
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg)

        batch, preds = _build_synthetic_batch_and_predictions()

        # Negative durations
        bad_batch = list(batch)
        bad_dur = batch[11].clone()
        bad_dur[0, 0] = -1
        bad_batch[11] = bad_dur
        with self.assertRaises(ValueError):
            loss_fn(bad_batch, preds)

        # Mismatched mel channels
        bad_preds = list(preds)
        bad_preds[0] = torch.randn(2, preds[0].shape[1], 40)
        with self.assertRaises(ValueError):
            loss_fn(batch, bad_preds)

        # Empty selection: all padded
        all_pad_preds = list(preds)
        all_pad_preds[6] = torch.ones_like(preds[6], dtype=torch.bool)
        with self.assertRaises(ValueError):
            loss_fn(batch, all_pad_preds)

    def test_09_mapping_input_interface(self):
        """Verify loss accepts dictionary inputs and predictions."""
        p_cfg, m_cfg = _make_dummy_configs("phoneme_level", "phoneme_level")
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg)

        batch, preds = _build_synthetic_batch_and_predictions()

        batch_dict = {
            "mel_targets": batch[6],
            "pitch_targets": batch[9],
            "energy_targets": batch[10],
            "duration_targets": batch[11],
        }
        pred_dict = {
            "mel_predictions": preds[0],
            "postnet_mel_predictions": preds[1],
            "pitch_predictions": preds[2],
            "energy_predictions": preds[3],
            "log_duration_predictions": preds[4],
            "src_masks": preds[6],
            "mel_masks": preds[7],
        }

        losses = loss_fn(batch_dict, pred_dict)
        self.assertEqual(len(losses), 6)
        for l in losses:
            self.assertTrue(torch.isfinite(l))

    def test_10_end_to_end_fastspeech2_training_step(self):
        """Full end-to-end forward, loss, and backward with FastSpeech2 model."""
        p_cfg_dict, m_cfg_dict = _make_dummy_configs()
        p_cfg = PreprocessConfig.from_dict(p_cfg_dict)
        from common.config.model import ModelConfig

        m_cfg = ModelConfig.from_dict(m_cfg_dict)

        stats = {"pitch": [10.0, 810.0, 0.0, 1.0], "energy": [1.0, 101.0, 0.0, 1.0]}
        model = FastSpeech2(p_cfg, m_cfg, stats=stats)
        loss_fn = FastSpeech2Loss(p_cfg, m_cfg)

        batch_size = 2
        src_lens = [6, 4]
        mel_lens = [12, 8]
        max_src = 6
        max_mel = 12

        src_lens_t = torch.tensor(src_lens, dtype=torch.long)
        mel_lens_t = torch.tensor(mel_lens, dtype=torch.long)
        texts = torch.randint(1, 30, (batch_size, max_src))
        mel_targets = torch.randn(batch_size, max_mel, 80)
        pitch_targets = torch.randn(batch_size, max_src)
        energy_targets = torch.randn(batch_size, max_src)
        duration_targets = torch.zeros(batch_size, max_src, dtype=torch.long)
        for b in range(batch_size):
            duration_targets[b, : src_lens[b]] = mel_lens[b] // src_lens[b]
            rem = mel_lens[b] % src_lens[b]
            duration_targets[b, :rem] += 1

        batch = (
            ["id1", "id2"],
            ["text1", "text2"],
            None,
            texts,
            src_lens_t,
            max_src,
            mel_targets,
            mel_lens_t,
            max_mel,
            pitch_targets,
            energy_targets,
            duration_targets,
        )

        output = model(*(batch[2:]))
        losses = loss_fn(batch, output)

        total_loss = losses[0]
        self.assertTrue(torch.isfinite(total_loss))

        total_loss.backward()

        trainable_grads = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"grad is None for {name}")
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"grad contains non-finite values for {name}",
                )
                trainable_grads += 1

        self.assertGreater(trainable_grads, 0)

    def test_11_numerical_parity_with_original_loss(self):
        """Verify bitwise/floating-point parity with original loss implementation on aligned batch."""
        import importlib.util
        from common.paths import ORIGINAL_FS2_DIR

        spec = importlib.util.spec_from_file_location(
            "orig_loss", str(ORIGINAL_FS2_DIR / "model" / "loss.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        OrigLoss = mod.FastSpeech2Loss

        p_cfg, m_cfg = _make_dummy_configs("phoneme_level", "phoneme_level")
        orig_fn = OrigLoss(p_cfg, m_cfg)
        new_fn = FastSpeech2Loss(p_cfg, m_cfg)

        batch, preds = _build_synthetic_batch_and_predictions(
            batch_size=2,
            src_lens=[5, 4],
            mel_lens=[10, 8],
            pitch_level="phoneme_level",
            energy_level="phoneme_level",
        )

        orig_losses = orig_fn(batch, preds)
        new_losses = new_fn(batch, preds)

        for idx, name in enumerate(["total", "mel", "postnet", "pitch", "energy", "duration"]):
            diff = abs(orig_losses[idx].item() - new_losses[idx].item())
            self.assertLess(
                diff,
                1e-6,
                f"Mismatch in {name} loss between original and new: "
                f"{orig_losses[idx].item()} vs {new_losses[idx].item()}",
            )

    def test_12_dec10_original_defect_reproduction(self):
        """Demonstrate that original loss crashes on frame-level truncation while new loss succeeds."""
        import importlib.util
        from common.paths import ORIGINAL_FS2_DIR

        spec = importlib.util.spec_from_file_location(
            "orig_loss", str(ORIGINAL_FS2_DIR / "model" / "loss.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        OrigLoss = mod.FastSpeech2Loss

        p_cfg, m_cfg = _make_dummy_configs("frame_level", "frame_level")
        orig_fn = OrigLoss(p_cfg, m_cfg)
        new_fn = FastSpeech2Loss(p_cfg, m_cfg)

        batch, preds = _build_synthetic_batch_and_predictions(
            batch_size=2,
            src_lens=[5, 4],
            mel_lens=[12, 10],
            pitch_level="frame_level",
            energy_level="frame_level",
            target_mel_extra_len=3,  # Target longer than decoder predictions
        )

        # New implementation succeeds cleanly
        new_losses = new_fn(batch, preds)
        self.assertEqual(len(new_losses), 6)
        for l in new_losses:
            self.assertTrue(torch.isfinite(l))

        # Original implementation fails with dimension mismatch RuntimeError
        with self.assertRaises(RuntimeError):
            orig_fn(batch, preds)


if __name__ == "__main__":
    unittest.main()
