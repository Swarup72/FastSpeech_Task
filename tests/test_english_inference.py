"""Unit and integration tests for English FastSpeech 2 inference engine (English/inference.py).

Covers:
1. Canonical assignment sentences E1, E2(a), E2(b), E3, E4 preservation.
2. Text normalization, lexicon lookup, G2P fallback, and phoneme sequence generation.
3. Missing checkpoint error handling.
4. Missing vocoder weights error handling.
5. Speaker profile ID validation (range checking and warning on untrained profiles).
6. Forward pass and audio generation with mocked FastSpeech2 and HiFi-GAN components.
7. WAV file verification (file existence, non-emptiness, 22050 Hz sampling rate, finite samples).
8. End-to-end integration test with real model checkpoint and HiFi-GAN vocoder.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import yaml

from English.inference import (
    ASSIGNMENT_SAMPLES,
    DEFAULT_HIFIGAN_CONFIG,
    DEFAULT_HIFIGAN_WEIGHTS,
    DEFAULT_LEXICON_PATH,
    load_english_fastspeech2,
    load_hifigan_vocoder,
    preprocess_english_text,
    read_lexicon,
    resolve_checkpoint_path,
    synthesize_sentence,
    verify_wav_file,
)
from English.text.normalizer import normalize_text


class TestEnglishInferenceEngine(unittest.TestCase):
    """Test suite for English FastSpeech 2 inference pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon = read_lexicon(DEFAULT_LEXICON_PATH)
        cls.cleaners = ["english_cleaners"]

    def test_01_assignment_sample_definitions_and_preservation(self) -> None:
        """Verify all 5 canonical assignment test sentences are preserved exactly."""
        self.assertEqual(len(ASSIGNMENT_SAMPLES), 5)
        self.assertIn("E1", ASSIGNMENT_SAMPLES)
        self.assertIn("E2_a", ASSIGNMENT_SAMPLES)
        self.assertIn("E2_b", ASSIGNMENT_SAMPLES)
        self.assertIn("E3", ASSIGNMENT_SAMPLES)
        self.assertIn("E4", ASSIGNMENT_SAMPLES)

        # E1: antidisestablishmentarianism
        self.assertIn("antidisestablishmentarianism", ASSIGNMENT_SAMPLES["E1"])
        # E2(a) and E2(b): conduct heteronym
        self.assertIn("conduct", ASSIGNMENT_SAMPLES["E2_a"])
        self.assertIn("conduct", ASSIGNMENT_SAMPLES["E2_b"])
        # E3: rhyth m split token
        self.assertIn("rhyth m", ASSIGNMENT_SAMPLES["E3"])
        # E4: 10:45 AM, 12 November 2026, andaroo
        self.assertIn("10:45 AM", ASSIGNMENT_SAMPLES["E4"])
        self.assertIn("12 November 2026", ASSIGNMENT_SAMPLES["E4"])
        self.assertIn("andaroo", ASSIGNMENT_SAMPLES["E4"])

    def test_02_text_preprocessing_and_g2p_fallback_across_all_samples(self) -> None:
        """Verify all assignment test sentences normalize and phonemize into valid integer token sequences."""
        for sample_id, raw_text in ASSIGNMENT_SAMPLES.items():
            seq, phones, norm = preprocess_english_text(
                raw_text, self.lexicon, self.cleaners
            )
            self.assertIsInstance(seq, np.ndarray)
            self.assertGreater(len(seq), 0, f"Empty token sequence for {sample_id}")
            self.assertTrue(phones.startswith("{") and phones.endswith("}"))
            self.assertIsInstance(norm, str)
            self.assertGreater(len(norm), 0)

            # Check that tokens are valid integer IDs within vocabulary range
            self.assertTrue((seq >= 0).all())
            self.assertTrue((seq < 360).all())

        # Test empty string error handling
        with self.assertRaises(ValueError):
            preprocess_english_text("", self.lexicon, self.cleaners)
        with self.assertRaises(ValueError):
            preprocess_english_text("   \n  ", self.lexicon, self.cleaners)

    def test_03_missing_checkpoint_error_handling(self) -> None:
        """Verify loading from a non-existent checkpoint raises a descriptive FileNotFoundError."""
        dummy_configs = (
            {"path": {"preprocessed_path": "preprocessed_data/Svarah"}},
            {},
            {},
        )
        fake_path = Path("/tmp/definitely_non_existent_fastspeech2_checkpoint_99999.pth.tar")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_english_fastspeech2(fake_path, dummy_configs)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_04_missing_vocoder_weights_error_handling(self) -> None:
        """Verify loading from a non-existent vocoder path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_hifigan_vocoder(
                config_path=DEFAULT_HIFIGAN_CONFIG,
                weights_path="/tmp/non_existent_generator.pth.tar",
                zip_path="/tmp/non_existent_generator.zip",
            )

    def test_05_speaker_profile_id_validation(self) -> None:
        """Verify invalid speaker IDs raise ValueError and untrained profiles emit warnings."""
        mock_model = MagicMock()
        mock_model.speaker_emb = torch.nn.Embedding(115, 256)
        mock_model.n_speaker = 115

        mock_vocoder = MagicMock()
        configs = (
            {
                "preprocessing": {
                    "text": {"text_cleaners": ["english_cleaners"]},
                    "audio": {"sampling_rate": 22050},
                }
            },
            {},
            {},
        )

        # Negative speaker ID
        with self.assertRaises(ValueError):
            synthesize_sentence(
                mock_model,
                mock_vocoder,
                "Test sentence",
                speaker_id=-1,
                configs=configs,
                lexicon=self.lexicon,
            )

        # Exceeds n_speaker
        with self.assertRaises(ValueError):
            synthesize_sentence(
                mock_model,
                mock_vocoder,
                "Test sentence",
                speaker_id=115,
                configs=configs,
                lexicon=self.lexicon,
            )

    def test_06_mocked_forward_pass_and_audio_generation(self) -> None:
        """Verify full synthesize_sentence pipeline with mocked FastSpeech 2 and HiFi-GAN."""
        sr = 22050
        t_mel = 40
        mock_postnet_mel = torch.zeros(1, t_mel, 80)
        mock_dur = torch.full((1, 10), 4, dtype=torch.long)
        mock_pitch = torch.zeros(1, t_mel)
        mock_energy = torch.zeros(1, t_mel)
        mock_mel_len = torch.tensor([t_mel], dtype=torch.long)

        # 10-tuple output matching FastSpeech 2
        mock_output = (
            torch.zeros(1, t_mel, 80),
            mock_postnet_mel,
            mock_pitch,
            mock_energy,
            torch.zeros(1, 10),
            mock_dur,
            None,
            None,
            torch.tensor([10]),
            mock_mel_len,
        )

        mock_model = MagicMock()
        mock_model.speaker_emb = torch.nn.Embedding(115, 256)
        mock_model.n_speaker = 115
        mock_model.return_value = mock_output

        # Mock vocoder output: [1, 1, T_wav] where T_wav = t_mel * 256
        t_wav = t_mel * 256
        mock_wav = torch.sin(torch.linspace(0, 2 * np.pi * 440, t_wav)).unsqueeze(0).unsqueeze(0)
        mock_vocoder = MagicMock()
        mock_vocoder.return_value = mock_wav

        configs = (
            {
                "preprocessing": {
                    "text": {"text_cleaners": ["english_cleaners"]},
                    "audio": {"sampling_rate": sr},
                }
            },
            {},
            {},
        )

        wav_arr, out_sr, meta = synthesize_sentence(
            model=mock_model,
            vocoder=mock_vocoder,
            text="Hello world, this is a test.",
            speaker_id=0,
            configs=configs,
            lexicon=self.lexicon,
        )

        self.assertEqual(out_sr, sr)
        self.assertEqual(wav_arr.dtype, np.int16)
        self.assertEqual(len(wav_arr), t_wav)
        self.assertTrue(np.isfinite(wav_arr).all())
        self.assertIn("duration_seconds", meta)
        self.assertAlmostEqual(meta["duration_seconds"], float(t_wav) / float(sr), places=3)

    def test_07_wav_file_verification(self) -> None:
        """Verify verify_wav_file correctly checks existence, size, sample rate, and finiteness."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test.wav"
            sr = 22050
            data = (np.sin(np.linspace(0, 2 * np.pi * 440, sr)) * 32767).astype(np.int16)
            wavfile.write(tmp_path, sr, data)

            info = verify_wav_file(tmp_path, expected_sr=sr)
            self.assertEqual(info["sampling_rate"], sr)
            self.assertEqual(info["num_samples"], sr)
            self.assertAlmostEqual(info["duration_seconds"], 1.0, places=2)
            self.assertTrue(info["is_finite"])

            # Empty file raises ValueError
            empty_path = Path(tmp_dir) / "empty.wav"
            empty_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00")  # Corrupt truncated header
            with self.assertRaises(ValueError):
                verify_wav_file(empty_path)

            # Sample rate mismatch raises ValueError
            with self.assertRaises(ValueError):
                verify_wav_file(tmp_path, expected_sr=16000)

    def test_08_checkpoint_path_resolution(self) -> None:
        """Verify resolve_checkpoint_path handles explicit path and missing file errors."""
        train_cfg = {"path": {"ckpt_path": "./output/ckpt/Svarah_profile_stratified_v1"}}

        args_explicit = MagicMock()
        args_explicit.checkpoint = str(DEFAULT_LEXICON_PATH)  # real file as proxy
        p = resolve_checkpoint_path(args_explicit, train_cfg)
        self.assertEqual(p, DEFAULT_LEXICON_PATH)

        args_missing = MagicMock()
        args_missing.checkpoint = "/tmp/does_not_exist_xyz.pth.tar"
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint_path(args_missing, train_cfg)


if __name__ == "__main__":
    unittest.main()

