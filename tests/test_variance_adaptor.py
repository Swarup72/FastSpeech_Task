"""Unit tests for `common.model.modules.VarianceAdaptor` (Stage 2 — Step 1.4F Part 2)."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
import warnings

import torch

from common.config import (
    EnergyConfig,
    ModelConfig,
    PitchConfig,
    PreprocessConfig,
    PreprocessingSettings,
    PreprocessPathConfig,
    TextConfig,
    TransformerConfig,
    VarianceEmbeddingConfig,
    VariancePredictorConfig,
)
from common.model.modules import LengthRegulator, VarianceAdaptor

H = 32
N_BINS = 16
STATS = {"pitch": [10.0, 810.0, 0.0, 1.0], "energy": [1.0, 101.0, 0.0, 1.0]}


def make_configs(
    pitch_level="phoneme_level",
    energy_level="phoneme_level",
    pitch_q="linear",
    energy_q="linear",
    stats=None,
    path="./prep",
):
    prep = PreprocessConfig(
        dataset="Test",
        path=PreprocessPathConfig(
            corpus_path="./c", raw_path="./r", preprocessed_path=path
        ),
        preprocessing=PreprocessingSettings(
            text=TextConfig(language="en"),
            pitch=PitchConfig(feature=pitch_level, normalization=False),
            energy=EnergyConfig(feature=energy_level, normalization=False),
        ),
    )
    model = ModelConfig(
        n_src_vocab=50,
        transformer=TransformerConfig(
            encoder_hidden=H, decoder_hidden=H, encoder_head=2, decoder_head=2
        ),
        variance_predictor=VariancePredictorConfig(
            filter_size=H, kernel_size=3, dropout=0.0
        ),
        variance_embedding=VarianceEmbeddingConfig(
            pitch_quantization=pitch_q,
            energy_quantization=energy_q,
            n_bins=N_BINS,
        ),
        stats=stats,
    )
    return prep, model


def build(**kw):
    stats = kw.pop("stats", STATS)
    prep, model = make_configs(**kw)
    return VarianceAdaptor(prep, model, stats=stats)


def batch(lengths=(5, 3), seed=0):
    g = torch.Generator().manual_seed(seed)
    b, l = len(lengths), max(lengths)
    x = torch.randn(b, l, H, generator=g)
    lens = torch.tensor(lengths)
    mask = torch.arange(l).unsqueeze(0) >= lens.unsqueeze(1)
    return x, mask


class TestVarianceAdaptor(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    # 1. Shapes / embeddings
    def test_01_pitch_energy_embedding_shapes(self):
        va = build().eval()
        x, mask = batch()
        pred, emb = va.get_pitch_embedding(x, None, mask, 1.0)
        self.assertEqual(tuple(pred.shape), (2, 5))
        self.assertEqual(tuple(emb.shape), (2, 5, H))
        pred, emb = va.get_energy_embedding(x, None, mask, 1.0)
        self.assertEqual(tuple(emb.shape), (2, 5, H))
        self.assertEqual(va.pitch_embedding.weight.shape, (N_BINS, H))
        self.assertEqual(va.pitch_bins.shape, (N_BINS - 1,))

    # 2. Quantization boundaries
    def test_02_quantization_boundaries(self):
        va = build()
        bins = va.pitch_bins
        self.assertAlmostEqual(bins[0].item(), 10.0, places=4)
        self.assertAlmostEqual(bins[-1].item(), 810.0, places=3)
        idx = torch.bucketize(torch.tensor([-5.0, 10.0, 810.0, 5000.0]), bins)
        self.assertEqual(idx.tolist(), [0, 0, N_BINS - 2, N_BINS - 1])
        # log mode: geometric bins, same endpoints
        va_log = build(pitch_q="log", energy_q="log")
        lb = va_log.pitch_bins
        self.assertAlmostEqual(lb[0].item(), 10.0, places=4)
        self.assertAlmostEqual(lb[-1].item(), 810.0, places=2)
        ratios = lb[1:] / lb[:-1]
        torch.testing.assert_close(ratios, ratios[0].expand_as(ratios), rtol=1e-4, atol=1e-4)
        # log requires positive min
        with self.assertRaises(ValueError):
            build(pitch_q="log", stats={"pitch": [0.0, 5.0], "energy": [1.0, 2.0]})
        # state_dict keys match original naming
        keys = set(va.state_dict().keys())
        self.assertIn("pitch_bins", keys)
        self.assertIn("energy_bins", keys)
        # bins are non-trainable Parameters (as in the original implementation)
        self.assertIsInstance(va.pitch_bins, torch.nn.Parameter)
        self.assertFalse(va.pitch_bins.requires_grad)
        self.assertFalse(va.energy_bins.requires_grad)

    # 3. Padding behaviour
    def test_03_padding_behavior(self):
        va = build().eval()
        x, mask = batch()
        pred, emb = va.get_pitch_embedding(x, torch.full((2, 5), 100.0), mask, 1.0)
        self.assertEqual(pred[1, 3:].abs().max().item(), 0.0)
        self.assertEqual(emb[1, 3:].abs().max().item(), 0.0)
        out = va(x, mask, duration_target=torch.ones(2, 5, dtype=torch.long) * 2)
        # padded tokens contribute no frames even with nonzero target durations
        self.assertEqual(out[5].tolist(), [10, 6])
        self.assertEqual(out[4][1, 3:].tolist(), [0, 0])
        self.assertTrue(out[6][1, 6:].all().item())
        self.assertFalse(out[6][1, :6].any().item())

    # 4. Duration conversion
    def test_04_duration_conversion_and_rounding(self):
        f = VarianceAdaptor.log_duration_to_frames
        logd = torch.log(torch.tensor([[1.0, 2.0, 3.0, 5.4, 1.5]]) )  # d+1
        out = f(logd)
        self.assertEqual(out.dtype, torch.long)
        self.assertEqual(out.tolist(), [[0, 1, 2, 4, 0]])  # round(0.5)=0 (banker's)
        # d_control applied before rounding; 1-frame phoneme survives 0.8
        self.assertEqual(f(torch.log(torch.tensor([[2.0]])), 0.8).tolist(), [[1]])
        self.assertEqual(f(torch.log(torch.tensor([[2.0]])), 0.4).tolist(), [[0]])
        # negative predictions clamp to 0
        self.assertEqual(f(torch.tensor([[-3.0]])).tolist(), [[0]])
        # LengthRegulator itself must not round
        with self.assertRaises(ValueError):
            LengthRegulator()(torch.randn(1, 2, 4), torch.tensor([[1.5, 2.0]]))
        out, lens = LengthRegulator()(torch.randn(1, 2, 4), torch.tensor([[1.0, 2.0]]))
        self.assertEqual(lens.tolist(), [3])

    # 5. Training mode with ground truth
    def test_05_training_mode_ground_truth(self):
        for level in ("phoneme_level", "frame_level"):
            va = build(pitch_level=level, energy_level=level).train()
            x, src_mask = batch()
            dur = torch.tensor([[1, 2, 0, 1, 2], [2, 1, 1, 0, 0]])
            mel_lens = [6, 4]
            max_len = 6
            mel_mask = torch.arange(max_len).unsqueeze(0) >= torch.tensor(mel_lens).unsqueeze(1)
            n = 5 if level == "phoneme_level" else max_len
            pt = torch.rand(2, n) * 800
            et = torch.rand(2, n) * 100
            x_out, p, e, logd, d_r, mel_len, mm = va(
                x, src_mask, mel_mask, max_len, pt, et, dur
            )
            self.assertEqual(tuple(x_out.shape), (2, max_len, H))
            self.assertEqual(tuple(p.shape), (2, n))
            self.assertEqual(tuple(e.shape), (2, n))
            self.assertEqual(tuple(logd.shape), (2, 5))
            self.assertEqual(d_r.dtype, torch.long)
            self.assertEqual(mel_len.tolist(), mel_lens)
            self.assertTrue(torch.equal(mm, mel_mask))
            # predictions returned (not targets), so they carry the predictor graph
            self.assertTrue(p.requires_grad)

    # 6. Inference with predictions
    def test_06_inference_predicted_values(self):
        va = build().eval()
        x, mask = batch()
        with torch.no_grad():
            # force known duration predictions
            va.duration_predictor.linear_layer.weight.zero_()
            va.duration_predictor.linear_layer.bias.fill_(math.log(3.0))  # d = 2
            x_out, p, e, logd, d_r, mel_len, mm = va(x, mask)
        self.assertEqual(d_r.tolist(), [[2] * 5, [2, 2, 2, 0, 0]])
        self.assertEqual(mel_len.tolist(), [10, 6])
        self.assertEqual(tuple(x_out.shape), (2, 10, H))
        self.assertEqual(mm.sum().item(), 4)
        # all-zero predicted durations -> 1-frame fallback on first valid token
        with torch.no_grad():
            va.duration_predictor.linear_layer.bias.fill_(-10.0)
            with self.assertWarns(RuntimeWarning):
                _, _, _, _, d_r, mel_len, _ = va(x, mask)
        self.assertEqual(d_r.tolist(), [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        self.assertEqual(mel_len.tolist(), [1, 1])

    # 7. Controls
    def test_07_pitch_energy_duration_controls(self):
        for level in ("phoneme_level", "frame_level"):
            va = build(pitch_level=level, energy_level=level).eval()
            x, mask = batch()
            with torch.no_grad():
                va.duration_predictor.linear_layer.weight.zero_()
                va.duration_predictor.linear_layer.bias.fill_(math.log(3.0))
                base = va(x, mask)
                pc = va(x, mask, p_control=1.5)
                ec = va(x, mask, e_control=1.5)
                dc = va(x, mask, d_control=2.0)
            torch.testing.assert_close(pc[1], base[1] * 1.5)  # pitch scaled
            torch.testing.assert_close(pc[2], base[2])  # energy untouched
            torch.testing.assert_close(ec[2], base[2] * 1.5)  # DEC-02
            torch.testing.assert_close(ec[1], base[1])
            self.assertEqual(dc[5].tolist(), [20, 12])  # d: 2*2=4 per token
            self.assertEqual(base[5].tolist(), [10, 6])
        with self.assertRaises(ValueError):
            va(x, mask, p_control=-1.0)
        with self.assertRaises(ValueError):
            va(x, mask, d_control=float("nan"))

    # 8. Length regulator integration order
    def test_08_length_regulator_integration(self):
        # phoneme-level: embeddings added before expansion (same token -> identical frames)
        va = build().eval()
        x, mask = batch((3,))
        dur = torch.tensor([[2, 1, 3]])
        out = va(x, mask, duration_target=dur)[0]
        torch.testing.assert_close(out[0, 0], out[0, 1])
        self.assertFalse(torch.allclose(out[0, 0], out[0, 2]))
        # frame-level: predictors see expanded frames -> shapes are L_mel
        vf = build(pitch_level="frame_level", energy_level="frame_level").eval()
        res = vf(x, mask, duration_target=dur)
        self.assertEqual(tuple(res[1].shape), (1, 6))
        self.assertEqual(tuple(res[3].shape), (1, 3))
        # max_len padding
        res = va(x, mask, max_len=9, duration_target=dur)
        self.assertEqual(tuple(res[0].shape), (1, 9, H))
        self.assertEqual(tuple(res[6].shape), (1, 9))
        self.assertEqual(res[6].sum().item(), 3)

    # 9. Devices
    def test_09_cpu_and_mps(self):
        x, mask = batch()
        dur = torch.tensor([[1, 2, 0, 1, 2], [2, 1, 1, 0, 0]])
        va = build().eval()
        with torch.no_grad():
            cpu = va(x, mask, duration_target=dur)
        self.assertEqual(cpu[0].device.type, "cpu")
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            self.skipTest("MPS unavailable")
        dev = torch.device("mps")
        vm = build().eval()
        vm.load_state_dict(va.state_dict())
        vm = vm.to(dev)
        with torch.no_grad():
            m = vm(x.to(dev), mask.to(dev), duration_target=dur.to(dev))
            import warnings

            with warnings.catch_warnings():
                # an untrained predictor may trigger the (tested) zero-duration fallback
                warnings.simplefilter("ignore", RuntimeWarning)
                inf = vm(x.to(dev), mask.to(dev))
        self.assertEqual(m[0].device.type, "mps")
        self.assertEqual(m[5].device.type, "mps")
        self.assertEqual(m[6].device.type, "mps")
        self.assertEqual(inf[4].dtype, torch.long)
        torch.testing.assert_close(m[0].cpu(), cpu[0], atol=1e-4, rtol=1e-4)

    # 10. Gradients
    def test_10_gradient_propagation(self):
        va = build().train()
        x, mask = batch()
        x.requires_grad_(True)
        dur = torch.tensor([[1, 2, 0, 1, 2], [2, 1, 1, 0, 0]])
        pt = torch.rand(2, 5) * 800
        et = torch.rand(2, 5) * 100
        out, p, e, logd, *_ = va(x, mask, None, None, pt, et, dur)
        loss = out.pow(2).mean() + p.pow(2).mean() + e.pow(2).mean() + logd.pow(2).mean()
        loss.backward()
        self.assertGreater(x.grad.abs().max().item(), 0)
        for name, prm in va.named_parameters():
            if "bins" in name:  # non-trainable quantization boundaries
                self.assertIsNone(prm.grad, name)
                continue
            self.assertIsNotNone(prm.grad, name)
            self.assertFalse(torch.isnan(prm.grad).any().item(), name)
        self.assertGreater(va.pitch_embedding.weight.grad.abs().max().item(), 0)
        self.assertGreater(va.energy_embedding.weight.grad.abs().max().item(), 0)
        self.assertIsNone(va.pitch_bins.grad if hasattr(va.pitch_bins, "grad") else None)

    # 11. Invalid / missing inputs
    def test_11_invalid_and_missing_inputs(self):
        va = build().eval()
        x, mask = batch()
        with self.assertRaises(ValueError):
            va(x[0], mask)
        with self.assertRaises(ValueError):
            va(x, None)
        with self.assertRaises(ValueError):
            va(x, mask[:, :3])
        with self.assertRaises(ValueError):
            va(torch.randn(2, 5, H + 1), mask)
        with self.assertRaises(ValueError):  # negative duration
            va(x, mask, duration_target=torch.tensor([[1, -1, 1, 1, 1]] * 2))
        with self.assertRaises(ValueError):  # float duration
            va(x, mask, duration_target=torch.ones(2, 5))
        with self.assertRaises(ValueError):  # wrong shape
            va(x, mask, duration_target=torch.ones(2, 4, dtype=torch.long))
        with self.assertRaises(ValueError):  # bad pitch target shape
            va(x, mask, pitch_target=torch.ones(2, 4),
               duration_target=torch.ones(2, 5, dtype=torch.long))
        with self.assertRaises(ValueError):  # mel_mask mismatch
            va(x, mask, mel_mask=torch.zeros(2, 3, dtype=torch.bool),
               duration_target=torch.ones(2, 5, dtype=torch.long))
        # missing stats
        prep, model = make_configs(path="/nonexistent_dir_xyz")
        with self.assertRaises(FileNotFoundError):
            VarianceAdaptor(prep, model)
        with self.assertRaises(ValueError):
            VarianceAdaptor(prep, model, stats={"pitch": [1.0]})
        with self.assertRaises(ValueError):
            VarianceAdaptor(prep, model, stats={"pitch": [5.0, 1.0], "energy": [0.0, 1.0]})

    # 13. Duration conversion regression (original vs current formula)
    def test_13_duration_control_regression(self):
        f = VarianceAdaptor.log_duration_to_frames
        d = torch.tensor([0.0, 1.0, 1.4, 1.5, 2.0, 2.4, 3.0, 5.0, 10.0])
        logd = torch.log(d + 1)
        expected = {
            0.8: [0, 1, 1, 1, 2, 2, 2, 4, 8],
            1.0: [0, 1, 1, 2, 2, 2, 3, 5, 10],
            1.5: [0, 2, 2, 2, 3, 4, 4, 8, 15],
        }
        for c, exp in expected.items():
            out = f(logd, c)
            self.assertEqual(out.dtype, torch.long)
            self.assertEqual(out.tolist(), exp, f"d_control={c}")
        # control == 1.0 is identical to the original formula (round(exp-1), clamp, int)
        orig = torch.clamp(torch.round(torch.exp(logd) - 1) * 1.0, min=0).int()
        self.assertEqual(f(logd, 1.0).tolist(), orig.tolist())
        # control != 1.0 documents the known difference: d=1, c=0.8 -> original 0, ours 1
        orig08 = torch.clamp(torch.round(torch.exp(logd) - 1) * 0.8, min=0).int()
        self.assertEqual(orig08[1].item(), 0)
        self.assertEqual(f(logd, 0.8)[1].item(), 1)
        self.assertIn("d_control", f.__doc__)
        self.assertIn("original", f.__doc__)
        # LengthRegulator still refuses to round
        lr = LengthRegulator()
        for bad in (torch.tensor([[1.5, 2.0]]), torch.tensor([[float("nan"), 1.0]])):
            with self.assertRaises(ValueError):
                lr(torch.randn(1, 2, 4), bad)

    # 14. Source-mask validation
    def test_14_fully_padded_source_rows_rejected(self):
        va = build().eval()
        x, mask = batch()
        bad = mask.clone()
        bad[1, :] = True
        with self.assertRaises(ValueError) as ctx:
            va(x, bad)
        self.assertIn("[1]", str(ctx.exception))
        self.assertIn("entirely padded", str(ctx.exception))
        with self.assertRaises(ValueError):  # also in training branch
            va(x, bad, duration_target=torch.ones(2, 5, dtype=torch.long))
        # valid individual zero-duration phonemes are still allowed
        dur = torch.tensor([[0, 2, 0, 1, 1], [1, 0, 2, 0, 0]])
        out = va(x, mask, duration_target=dur)
        self.assertEqual(out[5].tolist(), [4, 3])

    # 15. Fallback: never assigns to padded token, observable, no flooding
    def test_15_inference_fallback_warning_and_validity(self):
        import warnings

        va = build().eval()
        with torch.no_grad():
            va.duration_predictor.linear_layer.weight.zero_()
            va.duration_predictor.linear_layer.bias.fill_(-10.0)
        x, _ = batch((5, 5, 5))
        mask = torch.zeros(3, 5, dtype=torch.bool)
        mask[0, :2] = True  # leading padding: first valid token is index 2
        mask[1, 3:] = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with torch.no_grad():
                out = va(x, mask)
        fb = [i for i in w if issubclass(i.category, RuntimeWarning)]
        self.assertEqual(len(fb), 1)  # one warning per call, not per utterance
        self.assertIn("[0, 1, 2]", str(fb[0].message))
        d_r = out[4]
        self.assertEqual(d_r.tolist(), [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        self.assertEqual((d_r.masked_fill(~mask, 0)).sum().item(), 0)  # none on padding
        self.assertEqual(out[5].tolist(), [1, 1, 1])
        # no warning when the fallback is not needed
        with torch.no_grad():
            va.duration_predictor.linear_layer.bias.fill_(math.log(3.0))
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            with torch.no_grad():
                va(x, mask)
        self.assertEqual([i for i in w2 if issubclass(i.category, RuntimeWarning)], [])
        # index list is capped
        va2 = build().eval()
        with torch.no_grad():
            va2.duration_predictor.linear_layer.weight.zero_()
            va2.duration_predictor.linear_layer.bias.fill_(-10.0)
        xb, mb = batch(tuple([3] * 14))
        with warnings.catch_warnings(record=True) as w3:
            warnings.simplefilter("always")
            with torch.no_grad():
                va2(xb, mb)
        self.assertIn("+4 more", str(w3[0].message))

    def test_16_ground_truth_zero_length_error_and_no_fallback(self):
        import warnings

        va = build().train()
        x, mask = batch()
        gt = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 0, 0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError) as ctx:
                va(x, mask, duration_target=gt)
        self.assertIn("duration_target sums to 0 frames", str(ctx.exception))
        self.assertIn("[0]", str(ctx.exception))
        self.assertEqual([i for i in w if issubclass(i.category, RuntimeWarning)], [])
        # valid ground truth is returned unmodified
        gt_ok = torch.tensor([[1, 0, 2, 0, 1], [2, 1, 0, 0, 0]])
        out = va(x, mask, duration_target=gt_ok)
        self.assertTrue(torch.equal(out[4], gt_ok))

    # 17. Bins are non-trainable Parameters
    def test_17_bins_are_non_trainable_parameters(self):
        va = build()
        names = [n for n, _ in va.named_parameters()]
        self.assertIn("pitch_bins", names)
        self.assertIn("energy_bins", names)
        self.assertEqual(len(list(va.buffers())), 0)
        for b in (va.pitch_bins, va.energy_bins):
            self.assertIsInstance(b, torch.nn.Parameter)
            self.assertFalse(b.requires_grad)
            self.assertEqual(b.dtype, torch.float32)
            self.assertEqual(tuple(b.shape), (N_BINS - 1,))
        sd = va.state_dict()
        self.assertIn("pitch_bins", sd)
        self.assertIn("energy_bins", sd)
        # trainable parameter count excludes exactly the two bins
        trainable = [n for n, p in va.named_parameters() if p.requires_grad]
        self.assertEqual(len(trainable), len(names) - 2)
        # state_dict round trip
        other = build(stats={"pitch": [1.0, 2.0], "energy": [3.0, 4.0]})
        other.load_state_dict(sd, strict=True)
        torch.testing.assert_close(other.pitch_bins, va.pitch_bins)
        # bins never receive gradients, and stay non-trainable after backward
        x, mask = batch()
        va.train()
        dur = torch.tensor([[1, 2, 0, 1, 2], [2, 1, 1, 0, 0]])
        out = va(x, mask, None, None, torch.rand(2, 5) * 800, torch.rand(2, 5) * 100, dur)
        (out[0].pow(2).mean() + out[1].pow(2).mean()).backward()
        self.assertIsNone(va.pitch_bins.grad)
        self.assertIsNone(va.energy_bins.grad)
        self.assertFalse(va.pitch_bins.requires_grad)
        # an optimizer step leaves bins unchanged
        before = va.pitch_bins.detach().clone()
        torch.optim.SGD([p for p in va.parameters()], lr=0.1).step()
        torch.testing.assert_close(va.pitch_bins, before)

    def test_18_bins_move_with_to_device(self):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            self.skipTest("MPS unavailable")
        va = build().to(torch.device("mps"))
        self.assertEqual(va.pitch_bins.device.type, "mps")
        self.assertEqual(va.energy_bins.device.type, "mps")
        self.assertFalse(va.pitch_bins.requires_grad)
        self.assertIsInstance(va.pitch_bins, torch.nn.Parameter)

    # 19. Native MPS backward regression
    def test_19_mps_backward_phoneme_and_frame_level(self):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            self.skipTest("MPS unavailable")
        dev = torch.device("mps")
        for level in ("phoneme_level", "frame_level"):
            with self.subTest(level=level):
                va = build(pitch_level=level, energy_level=level).to(dev).train()
                x, mask = batch()
                x = x.to(dev).requires_grad_(True)
                mask = mask.to(dev)
                dur = torch.tensor([[1, 2, 0, 1, 2], [2, 1, 1, 0, 0]], device=dev)
                n = 5 if level == "phoneme_level" else 6
                pt = (torch.rand(2, n) * 800).to(dev)
                et = (torch.rand(2, n) * 100).to(dev)
                max_len = 6 if level == "frame_level" else None
                out = va(x, mask, None, max_len, pt, et, dur)
                for t in (out[0], out[1], out[2], out[3], out[5], out[6]):
                    self.assertEqual(t.device.type, "mps")
                loss = (
                    out[0].pow(2).mean() + out[1].pow(2).mean()
                    + out[2].pow(2).mean() + out[3].pow(2).mean()
                )
                loss.backward()
                self.assertEqual(x.grad.device.type, "mps")
                self.assertTrue(torch.isfinite(x.grad).all().item())
                for name, prm in va.named_parameters():
                    if "bins" in name:
                        self.assertIsNone(prm.grad, name)
                        continue
                    self.assertIsNotNone(prm.grad, name)
                    self.assertTrue(torch.isfinite(prm.grad).all().item(), name)
                self.assertGreater(va.pitch_embedding.weight.grad.abs().max().item(), 0)
                self.assertGreater(va.energy_embedding.weight.grad.abs().max().item(), 0)
                for pred in (va.duration_predictor, va.pitch_predictor, va.energy_predictor):
                    self.assertGreater(pred.linear_layer.weight.grad.abs().max().item(), 0)

    # 12. Stats sources
    def test_12_stats_sources(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "stats.json"), "w") as f:
                json.dump(STATS, f)
            prep, model = make_configs(path=d)
            va_file = VarianceAdaptor(prep, model)
        prep, model = make_configs(stats=STATS)
        va_cfg = VarianceAdaptor(prep, model)
        torch.testing.assert_close(va_file.pitch_bins, va_cfg.pitch_bins)
        torch.testing.assert_close(va_file.energy_bins, build().energy_bins)

    # 13. Phase 3 modular import identity and functional equivalence
    def test_13_phase3_variance_adaptor_import_identity_and_facade_equivalence(self):
        """13. Verify VarianceAdaptor import identity and bitwise execution equivalence."""
        from common.model.variance_adaptor import VarianceAdaptor as CanonicalVA
        from common.model.modules import VarianceAdaptor as FacadeVA
        from common.model import VarianceAdaptor as ModelVA
        from common.modules import VarianceAdaptor as ModulesVA
        from common import VarianceAdaptor as CommonVA

        # Import identity checks
        self.assertIs(CanonicalVA, FacadeVA)
        self.assertIs(CanonicalVA, ModelVA)
        self.assertIs(CanonicalVA, ModulesVA)
        self.assertIs(CanonicalVA, CommonVA)

        # State dict invariants and non-trainable bins
        torch.manual_seed(42)
        va_canonical = CanonicalVA(*make_configs(stats=STATS))
        torch.manual_seed(99)
        va_facade = FacadeVA(*make_configs(stats=STATS))
        va_facade.load_state_dict(va_canonical.state_dict())

        self.assertEqual(
            list(va_canonical.state_dict().keys()),
            list(va_facade.state_dict().keys()),
        )
        for k, v in va_canonical.state_dict().items():
            self.assertEqual(v.shape, va_facade.state_dict()[k].shape)
            torch.testing.assert_close(v, va_facade.state_dict()[k])

        self.assertFalse(va_canonical.pitch_bins.requires_grad)
        self.assertFalse(va_canonical.energy_bins.requires_grad)

        # Forward output equivalence: training mode
        va_canonical.eval()
        va_facade.eval()
        x, mask = batch()
        dur = torch.tensor([[2, 1, 0, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.long)
        pt = torch.rand(2, 5) * 800
        et = torch.rand(2, 5) * 100

        out_c = va_canonical(
            x, mask, duration_target=dur, pitch_target=pt, energy_target=et
        )
        out_f = va_facade(
            x, mask, duration_target=dur, pitch_target=pt, energy_target=et
        )
        self.assertEqual(len(out_c), 7)
        self.assertEqual(len(out_f), 7)
        for idx in range(7):
            torch.testing.assert_close(out_c[idx], out_f[idx])

        # Forward output equivalence: inference mode with controls
        torch.manual_seed(123)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out_c_inf = va_canonical(
                x, mask, p_control=1.2, e_control=0.9, d_control=1.1
            )
            out_f_inf = va_facade(
                x, mask, p_control=1.2, e_control=0.9, d_control=1.1
            )
        for idx in range(7):
            torch.testing.assert_close(out_c_inf[idx], out_f_inf[idx])

    # 14. Phase 3 decomposed forward helper methods
    def test_14_phase3_decomposed_helper_methods(self):
        """14. Verify decomposed private helper methods of VarianceAdaptor."""
        va = build(pitch_level="phoneme_level", energy_level="phoneme_level")
        x, mask = batch()

        # _validate_inputs: valid inputs
        clean_mask = va._validate_inputs(x, mask, 1.0, 1.0, 1.0)
        self.assertEqual(clean_mask.dtype, torch.bool)
        self.assertEqual(clean_mask.shape, mask.shape)

        # _validate_inputs: reject fully-padded source rows
        bad_mask = torch.ones(2, 5, dtype=torch.bool)
        with self.assertRaises(ValueError):
            va._validate_inputs(x, bad_mask, 1.0, 1.0, 1.0)

        # _validate_inputs: reject negative controls
        with self.assertRaises(ValueError):
            va._validate_inputs(x, mask, -1.0, 1.0, 1.0)

        # _process_pitch helper
        x_p, p_pred = va._process_pitch(x, None, clean_mask, 1.0)
        self.assertEqual(x_p.shape, x.shape)
        self.assertEqual(p_pred.shape, clean_mask.shape)

        # _process_energy helper
        x_e, e_pred = va._process_energy(x_p, None, clean_mask, 1.0)
        self.assertEqual(x_e.shape, x.shape)
        self.assertEqual(e_pred.shape, clean_mask.shape)

        # _process_durations helper (inference)
        log_d = va.duration_predictor(x, clean_mask)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_exp, d_round, mel_len, mel_mask = va._process_durations(
                x=x_e,
                src_mask=clean_mask,
                log_duration_prediction=log_d,
                duration_target=None,
                d_control=1.0,
                max_len=None,
                mel_mask=None,
            )
        self.assertEqual(x_exp.dim(), 3)
        self.assertEqual(x_exp.shape[-1], H)
        self.assertEqual(d_round.shape, clean_mask.shape)
        self.assertEqual(mel_len.shape, (2,))
        self.assertEqual(mel_mask.shape, (2, x_exp.shape[1]))


if __name__ == "__main__":
    unittest.main()

