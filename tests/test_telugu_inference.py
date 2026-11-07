"""Unit and integration tests for Telugu FastSpeech 2 inference engine (Telugu/inference.py).

Covers:
1. Telugu text normalization (Unicode NFC, Latin-o -> Anusvara, punctuation/whitespace).
2. Word-boundary and clause-punctuation mapping to @sp (ID 4), avoiding untrained IDs 7..18.
3. Vocabulary ID validity, digit handling ('telugu_digits' and 'words'), and UNK reporting.
4. Speaker mapping (spk_000 -> ID 0 female, spk_001 -> ID 1 male, invalid speaker rejection).
5. Checkpoint loading and compatibility validation (step 13000, vocab=101, n_speaker=2, mel=80).
6. Model inference output shapes and HiFi-GAN vocoder integration.
7. Pitch and energy inverse normalization against preprocessed_data/IndicTTS/stats.json.
8. Duration, pitch, and energy plot generation with Telugu-capable fonts.
9. WAV output validation and end-to-end run_inference pipeline verification.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import yaml

from common.config import FastSpeech2Config
from common.model.fastspeech2 import FastSpeech2
from Telugu.inference import (
    ASSIGNMENT_SAMPLES,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_STATS_PATH,
    DEFAULT_TRAIN_CONFIG,
    EXPECTED_BEST_STEP,
    EXPECTED_HOP_LENGTH,
    EXPECTED_N_MEL_CHANNELS,
    EXPECTED_NUM_SPEAKERS,
    EXPECTED_SAMPLING_RATE,
    EXPECTED_VOCAB_SIZE,
    TELUGU_ENERGY_MEAN,
    TELUGU_ENERGY_STD,
    TELUGU_PITCH_MEAN_HZ,
    TELUGU_PITCH_STD_HZ,
    UNTRAINED_PUNCT_AND_SPACE_IDS,
    audit_assignment_samples,
    generate_prosody_plots,
    inverse_normalize_prosody,
    load_telugu_fastspeech2,
    load_telugu_stats,
    parse_args,
    preprocess_telugu_text,
    resolve_checkpoint_path,
    resolve_telugu_speaker,
    run_inference,
    synthesize_sentence,
    verify_wav_file,
)
from Telugu.text.normalizer import normalize_text, sequence_to_text, text_to_sequence
from Telugu.text.symbols import PAD_ID, UNK_ID, VOCAB_SIZE, symbol_to_id


class TestTeluguInferenceEngine(unittest.TestCase):
    """Test suite for Telugu FastSpeech 2 inference and prosody visualization."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(DEFAULT_PREPROCESS_CONFIG, "r", encoding="utf-8") as f:
            cls.prep_dict = yaml.safe_load(f)
        with open(DEFAULT_MODEL_CONFIG, "r", encoding="utf-8") as f:
            cls.model_dict = yaml.safe_load(f)
        with open(DEFAULT_TRAIN_CONFIG, "r", encoding="utf-8") as f:
            cls.train_dict = yaml.safe_load(f)
        cls.stats = load_telugu_stats(DEFAULT_STATS_PATH)

    def test_01_telugu_text_normalization(self) -> None:
        """Verify Unicode NFC normalization, Latin-o -> Anusvara correction, and whitespace cleanup."""
        raw = "  అందరూ   మౌనoగా ఉన్న,\nఆ తరుణంలో\u200c కాళిదాసు లేచి।  "
        norm = normalize_text(raw)
        self.assertEqual(norm, "అందరూ మౌనంగా ఉన్న, ఆ తరుణంలో కాళిదాసు లేచి.")
        self.assertNotIn("\u200c", norm)
        self.assertNotIn("o", norm)
        self.assertIn("ం", norm)

    def test_02_word_boundary_and_clause_mapping_to_sp(self) -> None:
        """Verify spaces and clause punctuation map to @sp (ID 4) and never emit IDs 7..18."""
        text = "అందరూ మౌనంగా ఉన్న, ఆ తరుణంలో కాళిదాసు లేచి."
        seq_np, tokens, curly_str, norm_text, unk_report = preprocess_telugu_text(text)

        self.assertEqual(len(unk_report), 0)
        self.assertTrue(curly_str.startswith("{") and curly_str.endswith("}"))
        self.assertIn("@sp", tokens)
        self.assertNotEqual(tokens[0], "@sp")
        self.assertNotEqual(tokens[-1], "@sp")

        # Consecutive ', ' between 'ఉన్న,' and 'ఆ' must collapse to a single '@sp'
        for i in range(len(tokens) - 1):
            self.assertFalse(
                tokens[i] == "@sp" and tokens[i + 1] == "@sp",
                f"Consecutive @sp found at index {i}: {tokens}",
            )

        # Verify @sp is mapped to token ID 4 and no untrained punctuation/space IDs (7..18) exist
        sp_id = symbol_to_id["@sp"]
        self.assertEqual(sp_id, 4)
        self.assertIn(4, seq_np.tolist())
        for tok_id in seq_np.tolist():
            self.assertNotIn(tok_id, UNTRAINED_PUNCT_AND_SPACE_IDS)

        # Verify all meaningful Telugu characters are preserved
        reconstructed = sequence_to_text(seq_np.tolist())
        expected_graphemes = "".join(
            ch for ch in norm_text if ch not in {" ", ",", "."}
        )
        self.assertEqual(reconstructed, expected_graphemes)

    def test_03_vocabulary_id_validity_digits_and_unk_reporting(self) -> None:
        """Verify vocabulary ID bounds, safe digit handling, and explicit UNK reporting."""
        # 1. Digits in 'telugu_digits' mode map ASCII 0-9 -> Telugu ౦-౯ (IDs 91..100)
        seq_digits, toks_digits, _, _, unk_digits = preprocess_telugu_text(
            "పది 10 గంటలు", digit_mode="telugu_digits"
        )
        self.assertEqual(len(unk_digits), 0)
        self.assertIn("౧", toks_digits)
        self.assertIn("౦", toks_digits)
        self.assertTrue((seq_digits > PAD_ID).all() and (seq_digits < VOCAB_SIZE).all())

        # 2. Digits in 'words' mode expand to spoken Telugu words
        seq_words, toks_words, _, _, unk_words = preprocess_telugu_text(
            "1 2", digit_mode="words"
        )
        self.assertEqual(len(unk_words), 0)
        self.assertEqual(sequence_to_text(seq_words.tolist()), "ఒకటిరెండు")

        # 3. Unsupported character is mapped to UNK_ID (1) and reported in unk_report
        seq_unk, toks_unk, _, _, unk_rep = preprocess_telugu_text(
            "కాళిదాసు @ లేచి", strict_no_unk=False
        )
        self.assertEqual(len(unk_rep), 1)
        self.assertEqual(unk_rep[0]["char"], "@")
        self.assertIn(UNK_ID, seq_unk.tolist())

        # With strict_no_unk=True, unsupported character raises ValueError
        with self.assertRaises(ValueError):
            preprocess_telugu_text("కాళిదాసు @ లేచి", strict_no_unk=True)

    def test_04_speaker_mapping(self) -> None:
        """Verify exact speaker mapping for spk_000 (0, female) and spk_001 (1, male)."""
        spk0_int = resolve_telugu_speaker(0)
        spk0_str = resolve_telugu_speaker("spk_000")
        spk0_fem = resolve_telugu_speaker("female")
        self.assertEqual(spk0_int, {"speaker_id": 0, "speaker_code": "spk_000", "gender": "female"})
        self.assertEqual(spk0_str, spk0_int)
        self.assertEqual(spk0_fem, spk0_int)

        spk1_int = resolve_telugu_speaker(1)
        spk1_str = resolve_telugu_speaker("spk_001")
        spk1_male = resolve_telugu_speaker("male")
        self.assertEqual(spk1_int, {"speaker_id": 1, "speaker_code": "spk_001", "gender": "male"})
        self.assertEqual(spk1_str, spk1_int)
        self.assertEqual(spk1_male, spk1_int)

        for bad_spk in (-1, 2, 88, "spk_002", "unknown", True):
            with self.assertRaises(ValueError):
                resolve_telugu_speaker(bad_spk)

    def test_05_checkpoint_loading_and_compatibility_validation(self) -> None:
        """Verify load_telugu_fastspeech2 loads a step-13000 checkpoint and validates shapes."""
        configs = (self.prep_dict, self.model_dict, self.train_dict)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoint_best.pth.tar"
            fs2_cfg = FastSpeech2Config.from_yaml_paths(
                DEFAULT_PREPROCESS_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG
            )
            ref_model = FastSpeech2(fs2_cfg.preprocess, fs2_cfg.model)
            torch.save(
                {
                    "model": ref_model.state_dict(),
                    "global_step": EXPECTED_BEST_STEP,
                    "epoch": 92,
                    "best_val_loss": 2.0869,
                },
                ckpt_path,
            )

            loaded_model, meta = load_telugu_fastspeech2(
                ckpt_path, configs, device="cpu", expected_step=EXPECTED_BEST_STEP, strict_step=True
            )
            self.assertEqual(meta["global_step"], EXPECTED_BEST_STEP)
            self.assertEqual(meta["epoch"], 92)
            self.assertAlmostEqual(meta["best_val_loss"], 2.0869, places=4)
            self.assertEqual(loaded_model.n_speaker, EXPECTED_NUM_SPEAKERS)
            self.assertEqual(loaded_model.n_mel_channels, EXPECTED_N_MEL_CHANNELS)
            self.assertEqual(
                loaded_model.encoder.src_word_emb.weight.shape[0], EXPECTED_VOCAB_SIZE
            )

            # Verify missing checkpoint raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                load_telugu_fastspeech2(Path(tmp_dir) / "missing.pth.tar", configs)

    def test_06_model_inference_output_shapes(self) -> None:
        """Verify real Telugu FastSpeech2 forward pass and mocked HiFi-GAN vocoder shapes."""
        fs2_cfg = FastSpeech2Config.from_yaml_paths(
            DEFAULT_PREPROCESS_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG
        )
        model = FastSpeech2(fs2_cfg.preprocess, fs2_cfg.model)
        model.eval()

        def _fake_vocoder(mel_in: torch.Tensor) -> torch.Tensor:
            # mel_in must have shape [1, 80, T_mel]
            self.assertEqual(mel_in.dim(), 3)
            self.assertEqual(mel_in.shape[0], 1)
            self.assertEqual(mel_in.shape[1], EXPECTED_N_MEL_CHANNELS)
            t_mel = int(mel_in.shape[2])
            t_wav = max(t_mel * EXPECTED_HOP_LENGTH, EXPECTED_HOP_LENGTH)
            return torch.sin(torch.linspace(0, 20 * np.pi, t_wav)).view(1, 1, t_wav)

        mock_vocoder = MagicMock(side_effect=_fake_vocoder)
        configs = (self.prep_dict, self.model_dict, self.train_dict)

        wav_arr, sr, meta = synthesize_sentence(
            model=model,
            vocoder=mock_vocoder,
            text="సత్యసంధుడు.",
            speaker="spk_000",
            configs=configs,
            device="cpu",
        )
        self.assertEqual(sr, EXPECTED_SAMPLING_RATE)
        self.assertEqual(wav_arr.dtype, np.int16)
        self.assertGreater(len(wav_arr), 0)
        self.assertEqual(meta["speaker_id"], 0)
        self.assertEqual(meta["speaker_code"], "spk_000")
        self.assertEqual(len(meta["variance_predictions"]["duration"]), meta["phoneme_tokens"])
        self.assertEqual(len(meta["variance_predictions"]["pitch_hz"]), meta["mel_frames"])
        self.assertEqual(len(meta["variance_predictions"]["energy_raw"]), meta["mel_frames"])

    def test_07_pitch_and_energy_inverse_normalization(self) -> None:
        """Verify exact inverse normalization formulas against preprocessed_data/IndicTTS/stats.json."""
        self.assertAlmostEqual(float(self.stats["pitch"][2]), TELUGU_PITCH_MEAN_HZ, places=9)
        self.assertAlmostEqual(float(self.stats["pitch"][3]), TELUGU_PITCH_STD_HZ, places=9)
        self.assertAlmostEqual(float(self.stats["energy"][2]), TELUGU_ENERGY_MEAN, places=9)
        self.assertAlmostEqual(float(self.stats["energy"][3]), TELUGU_ENERGY_STD, places=9)

        p_norm = np.array([0.0, 1.0, -1.0], dtype=np.float64)
        e_norm = np.array([0.0, 1.0, -0.5], dtype=np.float64)
        inv = inverse_normalize_prosody(p_norm, e_norm, stats=self.stats)

        expected_p_hz = p_norm * 46.419600809453875 + 195.619853412152
        expected_e_raw = e_norm * 26.661124763887056 + 28.10710166547053
        expected_t_sec = np.array([0.0, 256.0 / 22050.0, 512.0 / 22050.0])

        np.testing.assert_allclose(inv["pitch_hz"], expected_p_hz, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(inv["energy_raw"], expected_e_raw, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(inv["time_seconds"], expected_t_sec, rtol=1e-12, atol=1e-12)

    def test_08_plot_generation(self) -> None:
        """Verify duration, pitch, and energy PNG plots are generated and non-empty."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokens = ["స", "త", "్", "య", "@sp", "స", "ం", "ధ", "ు", "డ", "ు"]
            durations = [6, 5, 3, 7, 4, 6, 4, 6, 5, 5, 8]
            t_mel = sum(durations)
            time_sec = np.arange(t_mel) * 256.0 / 22050.0
            pitch_hz = 195.6 + 20.0 * np.sin(np.linspace(0, 2 * np.pi, t_mel))
            energy_raw = 28.1 + 10.0 * np.cos(np.linspace(0, 2 * np.pi, t_mel))

            res = generate_prosody_plots(
                sample_id="T_test",
                tokens=tokens,
                durations=durations,
                pitch_hz=pitch_hz,
                energy_raw=energy_raw,
                time_seconds=time_sec,
                plots_dir=tmp_dir,
            )
            for key in ("duration_plot", "pitch_plot", "energy_plot"):
                p = Path(res[key])
                self.assertTrue(p.is_file(), f"Missing plot: {p}")
                self.assertGreater(p.stat().st_size, 1000, f"Plot file too small: {p}")

    def test_09_wav_validation_and_end_to_end_pipeline(self) -> None:
        """Verify end-to-end run_inference generates WAV, 3 plots, and JSON metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            ckpt_path = tmp_root / "checkpoint_best.pth.tar"
            audio_dir = tmp_root / "audio"
            plots_dir = tmp_root / "plots"
            meta_dir = tmp_root / "metadata"

            fs2_cfg = FastSpeech2Config.from_yaml_paths(
                DEFAULT_PREPROCESS_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG
            )
            model = FastSpeech2(fs2_cfg.preprocess, fs2_cfg.model)
            # Bias duration predictor slightly positive so untrained model predicts non-zero durations
            with torch.no_grad():
                model.variance_adaptor.duration_predictor.linear_layer.bias.fill_(1.5)
            torch.save(
                {
                    "model": model.state_dict(),
                    "global_step": EXPECTED_BEST_STEP,
                    "epoch": 92,
                    "best_val_loss": 2.0869,
                },
                ckpt_path,
            )

            args = parse_args(
                [
                    "--mode",
                    "single",
                    "--text",
                    "సత్యసంధుడు.",
                    "--speaker",
                    "spk_000",
                    "--checkpoint",
                    str(ckpt_path),
                    "--output_dir",
                    str(audio_dir),
                    "--plots_dir",
                    str(plots_dir),
                    "--metadata_dir",
                    str(meta_dir),
                    "--output_filename",
                    "T_e2e_test.wav",
                    "--device",
                    "cpu",
                ]
            )
            results = run_inference(args)
            self.assertEqual(len(results), 1)

            wav_path = audio_dir / "T_e2e_test.wav"
            wav_info = verify_wav_file(wav_path, expected_sr=EXPECTED_SAMPLING_RATE)
            self.assertTrue(wav_info["is_finite"])
            self.assertGreater(wav_info["num_samples"], 0)

            self.assertTrue((plots_dir / "T_e2e_test_duration.png").is_file())
            self.assertTrue((plots_dir / "T_e2e_test_pitch.png").is_file())
            self.assertTrue((plots_dir / "T_e2e_test_energy.png").is_file())
            self.assertTrue((meta_dir / "T_e2e_test_metadata.json").is_file())
            self.assertTrue((meta_dir / "telugu_prosody_metadata.json").is_file())
            self.assertTrue((meta_dir / "telugu_samples_metadata.json").is_file())

    def test_10_canonical_assignment_samples_audit(self) -> None:
        """Verify all 5 canonical Telugu assignment samples pass tokenizer audit with 0 UNKs."""
        self.assertEqual(len(ASSIGNMENT_SAMPLES), 5)
        self.assertEqual(list(ASSIGNMENT_SAMPLES.keys()), ["T1", "T2_a", "T2_b", "T3", "T4"])
        self.assertIn("2026", ASSIGNMENT_SAMPLES["T4"])
        self.assertIn("10:45", ASSIGNMENT_SAMPLES["T4"])

        audit = audit_assignment_samples(digit_mode="telugu_digits")
        self.assertEqual(len(audit), 5)
        for sid, rec in audit.items():
            self.assertEqual(rec["unk_count"], 0, f"Unexpected UNK in {sid}")
            self.assertEqual(rec["untrained_ids_7_to_18"], [], f"Untrained IDs in {sid}")
            self.assertTrue(rec["character_preservation_verified"], f"Char loss in {sid}")
            self.assertEqual(rec["original_text"], ASSIGNMENT_SAMPLES[sid])

        # Verify default CLI speaker_id is 1 (spk_001, male)
        default_args = parse_args([])
        self.assertEqual(default_args.speaker_id, 1)
        self.assertEqual(default_args.digit_mode, "telugu_digits")


if __name__ == "__main__":
    unittest.main()

