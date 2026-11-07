"""Unit tests for the complete FastSpeech 2 model (`common.model.fastspeech2.FastSpeech2`).

Synthetic tensors only; no datasets, preprocessing or training.
"""

from __future__ import annotations

import unittest
import warnings

import torch

from common.config import (
    EnergyConfig,
    MelConfig,
    ModelConfig,
    PitchConfig,
    PostNetConfig,
    PreprocessConfig,
    PreprocessingSettings,
    PreprocessPathConfig,
    TextConfig,
    TransformerConfig,
    VarianceEmbeddingConfig,
    VariancePredictorConfig,
)
from common.model import FASTSPEECH2_OUTPUT_NAMES, FastSpeech2

H = 32
N_BINS = 16
STATS = {"pitch": [10.0, 810.0, 0.0, 1.0], "energy": [1.0, 101.0, 0.0, 1.0]}


def make_model(
    n_src_vocab=50,
    n_mel=80,
    level="phoneme_level",
    multi_speaker=False,
    n_speaker=None,
    max_seq_len=1000,
    language="en",
):
    prep = PreprocessConfig(
        dataset=f"Synthetic-{language}",
        path=PreprocessPathConfig(
            corpus_path="./c", raw_path="./r", preprocessed_path="./does_not_exist"
        ),
        preprocessing=PreprocessingSettings(
            text=TextConfig(language=language),
            mel=MelConfig(n_mel_channels=n_mel),
            pitch=PitchConfig(feature=level, normalization=False),
            energy=EnergyConfig(feature=level, normalization=False),
        ),
    )
    model = ModelConfig(
        n_src_vocab=n_src_vocab,
        transformer=TransformerConfig(
            encoder_layer=2,
            encoder_head=2,
            encoder_hidden=H,
            decoder_layer=2,
            decoder_head=2,
            decoder_hidden=H,
            conv_filter_size=64,
            conv_kernel_size=[9, 1],
            encoder_dropout=0.1,
            decoder_dropout=0.1,
        ),
        variance_predictor=VariancePredictorConfig(
            filter_size=H, kernel_size=3, dropout=0.1
        ),
        variance_embedding=VarianceEmbeddingConfig(n_bins=N_BINS),
        postnet=PostNetConfig(
            postnet_embedding_dim=32,
            postnet_kernel_size=5,
            postnet_n_convolutions=3,
            postnet_dropout=0.1,
        ),
        multi_speaker=multi_speaker,
        n_speaker=n_speaker,
        max_seq_len=max_seq_len,
        stats=STATS,
    )
    return FastSpeech2(prep, model)


def train_batch(vocab=50, n_mel=80, level="phoneme_level", multi=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    src_lens = torch.tensor([5, 3])
    texts = torch.randint(1, vocab, (2, 5), generator=g)
    texts[1, 3:] = 0
    d = torch.tensor([[1, 2, 1, 1, 2], [2, 1, 2, 0, 0]])
    mel_lens = d.sum(1)  # [7, 5]
    max_mel = int(mel_lens.max())
    n = 5 if level == "phoneme_level" else max_mel
    return dict(
        speakers=torch.tensor([0, 1]) if multi else None,
        texts=texts,
        src_lens=src_lens,
        max_src_len=5,
        mels=torch.randn(2, max_mel, n_mel, generator=g),
        mel_lens=mel_lens,
        max_mel_len=max_mel,
        p_targets=torch.rand(2, n, generator=g) * 800,
        e_targets=torch.rand(2, n, generator=g) * 100,
        d_targets=d,
    )


def infer_batch(vocab=50, multi=False, seed=0):
    b = train_batch(vocab=vocab, multi=multi, seed=seed)
    return dict(speakers=b["speakers"], texts=b["texts"], src_lens=b["src_lens"])


def make_durations_fixed(model, frames=2):
    """Force the duration predictor to output a constant, so inference is deterministic."""
    import math

    dp = model.variance_adaptor.duration_predictor.linear_layer
    with torch.no_grad():
        dp.weight.zero_()
        dp.bias.fill_(math.log(frames + 1.0))


class TestFastSpeech2(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    # 1. English / Telugu configurations
    def test_01_english_and_telugu_initialization(self):
        en = make_model(n_src_vocab=150, n_mel=80, language="en")
        te = make_model(n_src_vocab=95, n_mel=64, language="te")
        self.assertEqual(tuple(en.encoder.src_word_emb.weight.shape), (150, H))
        self.assertEqual(tuple(te.encoder.src_word_emb.weight.shape), (95, H))
        self.assertEqual(en.mel_linear.out_features, 80)
        self.assertEqual(te.mel_linear.out_features, 64)
        self.assertEqual(en.postnet.n_mel_channels, 80)
        self.assertEqual(te.postnet.n_mel_channels, 64)
        # both coexist and run in one process with their own vocabularies / mel sizes
        for m, v, n in ((en, 150, 80), (te, 95, 64)):
            m.eval()
            make_durations_fixed(m)
            with torch.no_grad():
                out = m(**infer_batch(vocab=v))
            self.assertEqual(out[0].shape[-1], n)
            self.assertEqual(out[1].shape[-1], n)
        # token id V-1 valid, V invalid
        en.eval()
        b = infer_batch(vocab=150)
        b["texts"][0, 0] = 149
        with torch.no_grad():
            en(**b)
        b["texts"][0, 0] = 150
        with self.assertRaises(ValueError):
            en(**b)

    # 2. Parameter dimensions / hierarchy
    def test_02_parameter_dimensions_and_hierarchy(self):
        m = make_model(multi_speaker=True, n_speaker=4)
        prefixes = {k.split(".")[0] for k in m.state_dict()}
        self.assertEqual(
            prefixes,
            {"encoder", "variance_adaptor", "decoder", "mel_linear", "postnet", "speaker_emb"},
        )
        self.assertEqual(tuple(m.speaker_emb.weight.shape), (4, H))
        self.assertEqual(tuple(m.mel_linear.weight.shape), (80, H))
        va = m.variance_adaptor
        self.assertEqual(tuple(va.pitch_embedding.weight.shape), (N_BINS, H))
        self.assertEqual(tuple(va.energy_embedding.weight.shape), (N_BINS, H))
        self.assertEqual(tuple(va.pitch_bins.shape), (N_BINS - 1,))
        self.assertFalse(va.pitch_bins.requires_grad)
        self.assertEqual(
            m.postnet.convolutions[0][0].conv.weight.shape, (32, 80, 5)
        )
        single = make_model()
        self.assertIsNone(single.speaker_emb)
        self.assertNotIn("speaker_emb.weight", single.state_dict())
        self.assertEqual(len(FASTSPEECH2_OUTPUT_NAMES), 10)

    # 3. Training forward
    def test_03_training_forward_contract(self):
        for level in ("phoneme_level", "frame_level"):
            for multi in (False, True):
                with self.subTest(level=level, multi=multi):
                    m = make_model(level=level, multi_speaker=multi, n_speaker=3).train()
                    b = train_batch(level=level, multi=multi)
                    out = m(**b)
                    self.assertEqual(len(out), 10)
                    mel, post, p, e, logd, d_r, sm, mm, sl, ml = out
                    self.assertEqual(tuple(mel.shape), (2, 7, 80))
                    self.assertEqual(tuple(post.shape), (2, 7, 80))
                    n = 5 if level == "phoneme_level" else 7
                    self.assertEqual(tuple(p.shape), (2, n))
                    self.assertEqual(tuple(e.shape), (2, n))
                    self.assertEqual(tuple(logd.shape), (2, 5))
                    self.assertEqual(d_r.dtype, torch.long)
                    self.assertTrue(torch.equal(d_r, b["d_targets"]))  # GT untouched
                    self.assertEqual(sm.dtype, torch.bool)
                    self.assertEqual(sm.tolist(), [[False] * 5, [False] * 3 + [True] * 2])
                    self.assertEqual(
                        mm.tolist(), [[False] * 7, [False] * 5 + [True] * 2]
                    )
                    self.assertEqual(sl.tolist(), [5, 3])
                    self.assertEqual(ml.tolist(), [7, 5])
                    for t in (mel, post, p, e, logd):
                        self.assertTrue(torch.isfinite(t).all().item())

    # 4. Inference + controls
    def test_04_inference_and_controls(self):
        m = make_model(level="phoneme_level").eval()
        make_durations_fixed(m, frames=2)
        b = infer_batch()
        with torch.no_grad():
            base = m(**b)
            pc = m(**b, p_control=1.5)
            ec = m(**b, e_control=1.5)
            dc = m(**b, d_control=2.0)
        self.assertEqual(base[9].tolist(), [10, 6])
        self.assertEqual(tuple(base[0].shape), (2, 10, 80))
        self.assertEqual(base[5].tolist(), [[2] * 5, [2, 2, 2, 0, 0]])
        torch.testing.assert_close(pc[2], base[2] * 1.5)
        torch.testing.assert_close(pc[3], base[3])
        torch.testing.assert_close(ec[3], base[3] * 1.5)
        torch.testing.assert_close(ec[2], base[2])
        self.assertEqual(dc[9].tolist(), [20, 12])
        for t in base[:5]:
            self.assertTrue(torch.isfinite(t).all().item())
        # frame-level inference
        mf = make_model(level="frame_level").eval()
        make_durations_fixed(mf, frames=2)
        with torch.no_grad():
            o = mf(**infer_batch())
        self.assertEqual(tuple(o[2].shape), (2, 10))
        self.assertEqual(tuple(o[3].shape), (2, 10))

    # 5. Mel dims, expansion and masks
    def test_05_expansion_and_masks(self):
        m = make_model().eval()
        b = train_batch()
        with torch.no_grad():
            mel, post, *_rest = m(**b)
            mm, ml = _rest[5], _rest[7]
        self.assertEqual(ml.tolist(), b["d_targets"].sum(1).tolist())
        self.assertEqual(mm.sum(1).tolist(), [0, 2])
        # inference lengths follow predicted durations
        make_durations_fixed(m, frames=3)
        with torch.no_grad():
            out = m(**infer_batch())
        self.assertEqual(out[9].tolist(), [15, 9])
        self.assertEqual(tuple(out[0].shape), (2, 15, 80))
        self.assertEqual(out[7].sum(1).tolist(), [0, 6])

    # 6. Padding behaviour
    def test_06_padding_behavior(self):
        m = make_model(multi_speaker=True, n_speaker=2).eval()
        b = train_batch(multi=True)
        with torch.no_grad():
            ref = m(**b)
        b2 = dict(b)
        texts2 = b["texts"].clone()
        texts2[1, 3:] = 7  # change padded tokens only
        b2["texts"] = texts2
        with torch.no_grad():
            alt = m(**b2)
        mm = ref[7]
        valid = ~mm
        torch.testing.assert_close(alt[0][valid], ref[0][valid], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alt[1][valid], ref[1][valid], atol=1e-5, rtol=1e-5)
        # PostNet residual is exactly zero on padded frames
        resid = ref[1] - ref[0]
        self.assertEqual(resid[mm].abs().max().item(), 0.0)
        # padded source positions contribute no frames and no predictions
        self.assertEqual(ref[4][1, 3:].abs().max().item(), 0.0)
        self.assertEqual(ref[2][1, 3:].abs().max().item(), 0.0)
        # utterance alone == utterance in padded batch (eval, valid frames)
        single = dict(
            speakers=b["speakers"][1:],
            texts=b["texts"][1:, :3],
            src_lens=torch.tensor([3]),
            max_src_len=3,
            mel_lens=b["mel_lens"][1:],
            d_targets=b["d_targets"][1:, :3],
            p_targets=b["p_targets"][1:, :3],
            e_targets=b["e_targets"][1:, :3],
        )
        with torch.no_grad():
            s = m(**single)
        torch.testing.assert_close(s[0][0], ref[0][1, :5], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(s[1][0], ref[1][1, :5], atol=1e-4, rtol=1e-4)

    # 7. CPU forward/backward + gradient propagation through all components
    def test_07_cpu_backward_and_component_gradients(self):
        for level in ("phoneme_level", "frame_level"):
            with self.subTest(level=level):
                m = make_model(level=level, multi_speaker=True, n_speaker=3).train()
                b = train_batch(level=level, multi=True)
                out = m(**b)
                valid = ~out[7]
                loss = (
                    (out[0][valid] - b["mels"][:, : out[0].shape[1]][valid]).abs().mean()
                    + (out[1][valid] - b["mels"][:, : out[1].shape[1]][valid]).abs().mean()
                    + out[2].pow(2).mean()
                    + out[3].pow(2).mean()
                    + out[4].pow(2).mean()
                )
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                groups = {
                    "encoder": 0, "variance_adaptor": 0, "decoder": 0,
                    "mel_linear": 0, "postnet": 0, "speaker_emb": 0,
                }
                for name, prm in m.named_parameters():
                    top = name.split(".")[0]
                    if not prm.requires_grad:  # frozen: *_bins, sinusoidal position_enc
                        self.assertIsNone(prm.grad, name)
                        continue
                    self.assertIsNotNone(prm.grad, f"no grad for {name}")
                    self.assertTrue(torch.isfinite(prm.grad).all().item(), name)
                    groups[top] += float(prm.grad.abs().sum())
                for top, total in groups.items():
                    self.assertGreater(total, 0.0, f"zero gradient in {top}")
                va = m.variance_adaptor
                for sub in (
                    va.duration_predictor, va.pitch_predictor, va.energy_predictor,
                ):
                    self.assertGreater(sub.linear_layer.weight.grad.abs().sum().item(), 0)
                self.assertGreater(va.pitch_embedding.weight.grad.abs().sum().item(), 0)

    # 8. MPS
    def test_08_mps_forward_and_backward(self):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            self.skipTest("MPS unavailable")
        dev = torch.device("mps")
        for level in ("phoneme_level", "frame_level"):
            with self.subTest(level=level):
                m = make_model(level=level, multi_speaker=True, n_speaker=3).to(dev).train()
                b = train_batch(level=level, multi=True)
                b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
                out = m(**b)
                for t in out:
                    self.assertEqual(t.device.type, "mps")
                valid = ~out[7]
                loss = (
                    out[0][valid].pow(2).mean() + out[1][valid].pow(2).mean()
                    + out[2].pow(2).mean() + out[3].pow(2).mean() + out[4].pow(2).mean()
                )
                loss.backward()
                for name, prm in m.named_parameters():
                    if not prm.requires_grad:
                        self.assertIsNone(prm.grad)
                        continue
                    self.assertIsNotNone(prm.grad, name)
                    self.assertTrue(torch.isfinite(prm.grad).all().item(), name)
        # inference natively on MPS matches CPU
        m = make_model().eval()
        make_durations_fixed(m)
        mm = make_model()
        mm.load_state_dict(m.state_dict())
        mm = mm.to(dev).eval()
        ib = infer_batch()
        with torch.no_grad():
            c = m(**ib)
            g = mm(**{k: (v.to(dev) if v is not None else None) for k, v in ib.items()})
        self.assertEqual(g[9].tolist(), c[9].tolist())
        torch.testing.assert_close(g[1].cpu(), c[1], atol=1e-3, rtol=1e-3)

    # 9. Invalid inputs
    def test_09_invalid_inputs(self):
        m = make_model(multi_speaker=True, n_speaker=2).eval()
        good = train_batch(multi=True)

        def call(**over):
            kw = dict(good)
            kw.update(over)
            return m(**kw)

        with self.assertRaises(ValueError):
            call(texts=good["texts"][0])  # 1D texts
        with self.assertRaises(ValueError):
            call(max_src_len=7)
        with self.assertRaises(ValueError):
            call(src_lens=torch.tensor([5, 9]))
        with self.assertRaises(ValueError):
            call(src_lens=torch.tensor([5, 0]))
        with self.assertRaises(ValueError):
            call(src_lens=torch.tensor([5.0, 3.0]))
        with self.assertRaises(ValueError):
            call(src_lens=torch.tensor([5, 3, 1]))
        with self.assertRaises(ValueError):
            call(speakers=None)
        with self.assertRaises(ValueError):
            call(speakers=torch.tensor([0, 2]))  # out of range
        with self.assertRaises(ValueError):
            call(mels=torch.randn(2, 7, 64))  # wrong mel channels
        with self.assertRaises(ValueError):
            call(mel_lens=torch.tensor([7, 6]))  # != sum(d_targets)
        with self.assertRaises(ValueError):
            call(d_targets=None)  # mel_lens without d_targets
        with self.assertRaises(ValueError):
            call(d_targets=torch.tensor([[1, -1, 1, 1, 2], [2, 1, 2, 0, 0]]))
        with self.assertRaises(ValueError):
            call(p_control=-1.0)
        with self.assertRaises(ValueError):
            call(texts=torch.full((2, 5), 99))  # out of vocabulary
        # fully padded source row
        with self.assertRaises(ValueError):
            call(src_lens=torch.tensor([5, 0]))
        # multi_speaker config needs n_speaker or speakers.json
        with self.assertRaises(FileNotFoundError):
            make_model(multi_speaker=True, n_speaker=None)
        # single-speaker model ignores speakers
        s = make_model().eval()
        kw = dict(good)
        kw["speakers"] = None
        s(**kw)

    # 10. Finite outputs and max_seq_len truncation
    def test_10_finite_and_max_seq_len(self):
        m = make_model(max_seq_len=6).train()
        b = train_batch()  # max mel length 7 > 6
        out = m(**b)
        self.assertEqual(tuple(out[0].shape), (2, 6, 80))  # decoder truncation (training)
        self.assertEqual(tuple(out[7].shape), (2, 6))
        self.assertEqual(out[9].tolist(), [7, 5])  # lengths are not truncated
        for t in out[:5]:
            self.assertTrue(torch.isfinite(t).all().item())
        # inference expands beyond max_seq_len without truncation
        m.eval()
        make_durations_fixed(m, frames=3)
        with torch.no_grad():
            o = m(**infer_batch())
        self.assertEqual(tuple(o[0].shape), (2, 15, 80))

    # 11. All-zero predicted duration fallback surfaces a warning at model level
    def test_11_inference_fallback_warning(self):
        m = make_model().eval()
        import math

        lin = m.variance_adaptor.duration_predictor.linear_layer
        with torch.no_grad():
            lin.weight.zero_()
            lin.bias.fill_(-10.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with torch.no_grad():
                out = m(**infer_batch())
        self.assertTrue(any(issubclass(i.category, RuntimeWarning) for i in w))
        self.assertEqual(out[9].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
