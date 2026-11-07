"""Unit tests for the shared FastSpeech 2 configuration foundation (`common.config`)."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from common.config import (
    ConfigValidationError,
    FastSpeech2Config,
    ModelConfig,
    PreprocessConfig,
    TrainConfig,
    load_fastspeech2_config,
    load_model_config,
    load_preprocess_config,
    load_train_config,
)


def _sample_english_dicts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Sample configuration dictionaries for the English (Svarah) pipeline."""
    preprocess = {
        "dataset": "Svarah_English",
        "path": {
            "corpus_path": "./data/English/Svarah_corpus",
            "lexicon_path": "./English/lexicon/svarah_lexicon.txt",
            "raw_path": "./English/raw_data",
            "preprocessed_path": "./English/preprocessed_data",
        },
        "preprocessing": {
            "val_size": 256,
            "text": {
                "text_cleaners": ["english_cleaners"],
                "language": "en",
            },
            "audio": {
                "sampling_rate": 22050,
                "max_wav_value": 32768.0,
            },
            "stft": {
                "filter_length": 1024,
                "hop_length": 256,
                "win_length": 1024,
            },
            "mel": {
                "n_mel_channels": 80,
                "mel_fmin": 0,
                "mel_fmax": 8000,
            },
            "pitch": {
                "feature": "phoneme_level",
                "normalization": True,
            },
            "energy": {
                "feature": "phoneme_level",
                "normalization": True,
            },
        },
    }
    model = {
        "n_src_vocab": 88,  # Distinct test vocabulary size for English ARPAbet + punctuation
        "transformer": {
            "encoder_layer": 4,
            "encoder_head": 2,
            "encoder_hidden": 256,
            "decoder_layer": 6,
            "decoder_head": 2,
            "decoder_hidden": 256,
            "conv_filter_size": 1024,
            "conv_kernel_size": [9, 1],
            "encoder_dropout": 0.2,
            "decoder_dropout": 0.2,
        },
        "variance_predictor": {
            "filter_size": 256,
            "kernel_size": 3,
            "dropout": 0.5,
        },
        "variance_embedding": {
            "pitch_quantization": "linear",
            "energy_quantization": "linear",
            "n_bins": 256,
        },
        "postnet": {
            "postnet_embedding_dim": 512,
            "postnet_kernel_size": 5,
            "postnet_n_convolutions": 5,
            "postnet_dropout": 0.5,
        },
        "multi_speaker": True,
        "n_speaker": 12,
        "max_seq_len": 1000,
        "vocoder": {
            "model": "HiFi-GAN",
            "speaker": "universal",
        },
    }
    train = {
        "path": {
            "ckpt_path": "./English/output/ckpt",
            "log_path": "./English/output/log",
            "result_path": "./English/output/result",
        },
        "optimizer": {
            "batch_size": 16,
            "betas": [0.9, 0.98],
            "eps": 1e-9,
            "weight_decay": 0.0,
            "grad_clip_thresh": 1.0,
            "grad_acc_step": 1,
            "warm_up_step": 4000,
            "anneal_steps": [300000, 400000, 500000],
            "anneal_rate": 0.3,
        },
        "step": {
            "total_step": 300000,
            "log_step": 100,
            "synth_step": 1000,
            "val_step": 1000,
            "save_step": 50000,
        },
    }
    return preprocess, model, train


def _sample_telugu_dicts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Sample configuration dictionaries for the Telugu (IndicTTS) pipeline."""
    preprocess = {
        "dataset": "IndicTTS_Telugu",
        "path": {
            "corpus_path": "./data/Telugu/IndicTTS_corpus",
            "lexicon_path": "./Telugu/lexicon/indictts_telugu_lexicon.txt",
            "raw_path": "./Telugu/raw_data",
            "preprocessed_path": "./Telugu/preprocessed_data",
        },
        "preprocessing": {
            "val_size": 128,
            "text": {
                "text_cleaners": [],  # Telugu script must NOT use ASCII unidecode cleaners
                "language": "te",
            },
            "audio": {
                "sampling_rate": 22050,
                "max_wav_value": 32768.0,
            },
            "stft": {
                "filter_length": 1024,
                "hop_length": 256,
                "win_length": 1024,
            },
            "mel": {
                "n_mel_channels": 80,
                "mel_fmin": 0,
                "mel_fmax": 8000,
            },
            "pitch": {
                "feature": "phoneme_level",
                "normalization": True,
            },
            "energy": {
                "feature": "phoneme_level",
                "normalization": True,
            },
        },
    }
    model = {
        "n_src_vocab": 64,  # Distinct test vocabulary size for Telugu phoneme set
        "transformer": {
            "encoder_layer": 4,
            "encoder_head": 2,
            "encoder_hidden": 256,
            "decoder_layer": 6,
            "decoder_head": 2,
            "decoder_hidden": 256,
            "conv_filter_size": 1024,
            "conv_kernel_size": [9, 1],
            "encoder_dropout": 0.2,
            "decoder_dropout": 0.2,
        },
        "variance_predictor": {
            "filter_size": 256,
            "kernel_size": 3,
            "dropout": 0.5,
        },
        "variance_embedding": {
            "pitch_quantization": "linear",
            "energy_quantization": "linear",
            "n_bins": 256,
        },
        "multi_speaker": False,
        "max_seq_len": 1200,
        "vocoder": {
            "model": "HiFi-GAN",
            "speaker": "universal",
        },
    }
    train = {
        "path": {
            "ckpt_path": "./Telugu/output/ckpt",
            "log_path": "./Telugu/output/log",
            "result_path": "./Telugu/output/result",
        },
        "optimizer": {
            "batch_size": 16,
            "betas": [0.9, 0.98],
            "eps": 1e-9,
            "weight_decay": 0.0,
            "grad_clip_thresh": 1.0,
            "grad_acc_step": 1,
            "warm_up_step": 4000,
            "anneal_steps": [200000, 300000],
            "anneal_rate": 0.3,
        },
        "step": {
            "total_step": 250000,
            "log_step": 100,
            "synth_step": 1000,
            "val_step": 1000,
            "save_step": 25000,
        },
    }
    return preprocess, model, train


class TestFastSpeech2Configuration(unittest.TestCase):
    """Tests for English/Telugu configurations, validation rules, and YAML loading."""

    def test_valid_english_and_telugu_configs_are_independent(self) -> None:
        """Verify valid English and Telugu configs coexist with distinct vocab sizes and settings."""
        en_prep, en_model, en_train = _sample_english_dicts()
        te_prep, te_model, te_train = _sample_telugu_dicts()

        en_cfg = FastSpeech2Config.from_dicts(en_prep, en_model, en_train)
        te_cfg = FastSpeech2Config.from_dicts(te_prep, te_model, te_train)

        # Distinct vocabulary sizes
        self.assertEqual(en_cfg.model.n_src_vocab, 88)
        self.assertEqual(te_cfg.model.n_src_vocab, 64)
        self.assertNotEqual(en_cfg.model.n_src_vocab, te_cfg.model.n_src_vocab)

        # Distinct language and text cleaners
        self.assertEqual(en_cfg.preprocess.preprocessing.text.language, "en")
        self.assertEqual(
            en_cfg.preprocess.preprocessing.text.text_cleaners,
            ["english_cleaners"],
        )
        self.assertEqual(te_cfg.preprocess.preprocessing.text.language, "te")
        self.assertEqual(te_cfg.preprocess.preprocessing.text.text_cleaners, [])

        # Distinct speaker settings and sequence lengths
        self.assertTrue(en_cfg.model.multi_speaker)
        self.assertEqual(en_cfg.model.n_speaker, 12)
        self.assertFalse(te_cfg.model.multi_speaker)
        self.assertIsNone(te_cfg.model.n_speaker)
        self.assertEqual(en_cfg.model.max_seq_len, 1000)
        self.assertEqual(te_cfg.model.max_seq_len, 1200)

        # Both attribute and dict-style indexing must return identical values
        self.assertEqual(
            en_cfg.model["transformer"]["encoder_hidden"],
            en_cfg.model.transformer.encoder_hidden,
        )
        self.assertEqual(
            en_cfg.preprocess["preprocessing"]["mel"]["n_mel_channels"], 80
        )
        self.assertEqual(
            te_cfg.model.get("postnet")["postnet_embedding_dim"], 512
        )

        # Round-trip to_dict()
        en_roundtrip = FastSpeech2Config.from_dicts(
            en_cfg.preprocess.to_dict(),
            en_cfg.model.to_dict(),
            en_cfg.train.to_dict() if en_cfg.train else None,
        )
        self.assertEqual(en_roundtrip.model.n_src_vocab, 88)

    def test_invalid_hidden_dim_and_attention_head_combinations(self) -> None:
        """Verify encoder_hidden/decoder_hidden divisibility by heads and equality checks."""
        _, base_model, _ = _sample_english_dicts()

        # Case 1: encoder_hidden (256) not divisible by encoder_head (3)
        bad_enc_head = copy.deepcopy(base_model)
        bad_enc_head["transformer"]["encoder_head"] = 3
        with self.assertRaises(ConfigValidationError) as ctx:
            ModelConfig.from_dict(bad_enc_head)
        self.assertIn("transformer.encoder_hidden", str(ctx.exception))
        self.assertIn("transformer.encoder_head", str(ctx.exception))

        # Case 2: decoder_hidden (256) not divisible by decoder_head (6)
        bad_dec_head = copy.deepcopy(base_model)
        bad_dec_head["transformer"]["decoder_head"] = 6
        with self.assertRaises(ConfigValidationError) as ctx:
            ModelConfig.from_dict(bad_dec_head)
        self.assertIn("transformer.decoder_hidden", str(ctx.exception))
        self.assertIn("transformer.decoder_head", str(ctx.exception))

        # Case 3: encoder_hidden (256) != decoder_hidden (512)
        mismatched_hidden = copy.deepcopy(base_model)
        mismatched_hidden["transformer"]["decoder_hidden"] = 512
        with self.assertRaises(ConfigValidationError) as ctx:
            ModelConfig.from_dict(mismatched_hidden)
        self.assertIn("must equal", str(ctx.exception))

        # Case 4: Non-positive hidden dimension or layer count
        zero_hidden = copy.deepcopy(base_model)
        zero_hidden["transformer"]["encoder_hidden"] = 0
        with self.assertRaises(ConfigValidationError) as ctx:
            ModelConfig.from_dict(zero_hidden)
        self.assertIn("transformer.encoder_hidden", str(ctx.exception))

        # Case 5: Even kernel size in PositionwiseFeedForward or VariancePredictor
        even_kernel = copy.deepcopy(base_model)
        even_kernel["transformer"]["conv_kernel_size"] = [8, 1]
        with self.assertRaises(ConfigValidationError) as ctx:
            ModelConfig.from_dict(even_kernel)
        self.assertIn("odd positive integer", str(ctx.exception))

        even_vp_kernel = copy.deepcopy(base_model)
        even_vp_kernel["variance_predictor"]["kernel_size"] = 4
        with self.assertRaises(ConfigValidationError) as ctx:
            ModelConfig.from_dict(even_vp_kernel)
        self.assertIn("variance_predictor.kernel_size", str(ctx.exception))

    def test_invalid_dropout_values(self) -> None:
        """Verify dropout values outside [0.0, 1.0) are rejected with clear errors."""
        _, base_model, _ = _sample_english_dicts()

        for section, key, bad_val in [
            ("transformer", "encoder_dropout", -0.1),
            ("transformer", "decoder_dropout", 1.0),
            ("transformer", "encoder_dropout", 1.5),
            ("variance_predictor", "dropout", -0.01),
            ("variance_predictor", "dropout", 1.0),
            ("postnet", "postnet_dropout", 1.2),
            ("postnet", "postnet_dropout", True),
        ]:
            mutated = copy.deepcopy(base_model)
            mutated[section][key] = bad_val
            with self.assertRaises(ConfigValidationError) as ctx:
                ModelConfig.from_dict(mutated)
            self.assertIn(f"{section}.{key}", str(ctx.exception))

    def test_missing_required_fields_raise_clear_errors(self) -> None:
        """Verify missing required fields in preprocess, model, and train configs raise ConfigValidationError."""
        en_prep, en_model, en_train = _sample_english_dicts()

        # 1. Missing n_src_vocab in ModelConfig
        no_vocab = copy.deepcopy(en_model)
        del no_vocab["n_src_vocab"]
        with self.assertRaises(ConfigValidationError) as ctx:
            ModelConfig.from_dict(no_vocab)
        self.assertIn("n_src_vocab", str(ctx.exception))

        # 2. Missing transformer section in ModelConfig
        no_transformer = copy.deepcopy(en_model)
        del no_transformer["transformer"]
        with self.assertRaises(ConfigValidationError) as ctx:
            ModelConfig.from_dict(no_transformer)
        self.assertIn("model_config.transformer", str(ctx.exception))

        # 3. Missing dataset in PreprocessConfig
        no_dataset = copy.deepcopy(en_prep)
        del no_dataset["dataset"]
        with self.assertRaises(ConfigValidationError) as ctx:
            PreprocessConfig.from_dict(no_dataset)
        self.assertIn("preprocess_config.dataset", str(ctx.exception))

        # 4. Missing preprocessing.text.language in PreprocessConfig
        no_lang = copy.deepcopy(en_prep)
        del no_lang["preprocessing"]["text"]["language"]
        with self.assertRaises(ConfigValidationError) as ctx:
            PreprocessConfig.from_dict(no_lang)
        self.assertIn("preprocessing.text.language", str(ctx.exception))

        # 5. Missing path.ckpt_path in TrainConfig
        no_ckpt = copy.deepcopy(en_train)
        del no_ckpt["path"]["ckpt_path"]
        with self.assertRaises(ConfigValidationError) as ctx:
            TrainConfig.from_dict(no_ckpt)
        self.assertIn("train_config.path.ckpt_path", str(ctx.exception))

    def test_cross_config_and_stats_log_quantization_validation(self) -> None:
        """Verify DEC-04 / B-03: log quantization is rejected when normalized or when min <= 0."""
        en_prep, en_model, _ = _sample_english_dicts()

        # Cross-config check: pitch.normalization=True + pitch_quantization='log'
        bad_log_model = copy.deepcopy(en_model)
        bad_log_model["variance_embedding"]["pitch_quantization"] = "log"
        with self.assertRaises(ConfigValidationError) as ctx:
            FastSpeech2Config.from_dicts(en_prep, bad_log_model)
        self.assertIn("pitch_quantization", str(ctx.exception))
        self.assertIn("normalization", str(ctx.exception))

        # In-memory stats check: pitch_min <= 0.0 with pitch_quantization='log'
        bad_stats_model = copy.deepcopy(en_model)
        bad_stats_model["variance_embedding"]["pitch_quantization"] = "log"
        bad_stats_model["stats"] = {
            "pitch": [0.0, 350.0, 180.0, 45.0],
            "energy": [1.0, 90.0, 25.0, 10.0],
        }
        with self.assertRaises(ConfigValidationError) as ctx:
            ModelConfig.from_dict(bad_stats_model)
        self.assertIn("stats.pitch", str(ctx.exception))

    def test_yaml_loading_and_original_repo_yaml_compatibility(self) -> None:
        """Verify YAML loading for English/Telugu configs and compatibility with original repo YAMLs."""
        en_prep, en_model, en_train = _sample_english_dicts()
        te_prep, te_model, te_train = _sample_telugu_dicts()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            en_prep_yaml = tmp_path / "en_preprocess.yaml"
            en_model_yaml = tmp_path / "en_model.yaml"
            en_train_yaml = tmp_path / "en_train.yaml"
            te_prep_yaml = tmp_path / "te_preprocess.yaml"
            te_model_yaml = tmp_path / "te_model.yaml"

            en_prep_yaml.write_text(yaml.safe_dump(en_prep), encoding="utf-8")
            en_model_yaml.write_text(yaml.safe_dump(en_model), encoding="utf-8")
            en_train_yaml.write_text(yaml.safe_dump(en_train), encoding="utf-8")
            te_prep_yaml.write_text(yaml.safe_dump(te_prep), encoding="utf-8")
            te_model_yaml.write_text(yaml.safe_dump(te_model), encoding="utf-8")

            loaded_en = load_fastspeech2_config(
                en_prep_yaml, en_model_yaml, en_train_yaml
            )
            loaded_te = load_fastspeech2_config(te_prep_yaml, te_model_yaml)

            self.assertEqual(loaded_en.model.n_src_vocab, 88)
            self.assertIsNotNone(loaded_en.train)
            self.assertEqual(loaded_te.model.n_src_vocab, 64)
            self.assertIsNone(loaded_te.train)

            # Non-existent YAML file raises ConfigValidationError
            with self.assertRaises(ConfigValidationError):
                load_preprocess_config(tmp_path / "nonexistent.yaml")

            # Malformed YAML file raises ConfigValidationError
            bad_yaml = tmp_path / "malformed.yaml"
            bad_yaml.write_text("transformer: [unclosed", encoding="utf-8")
            with self.assertRaises(ConfigValidationError):
                load_model_config(bad_yaml, n_src_vocab=50)

        # Also verify read-only loading of original_fastspeech2 LJSpeech YAMLs when n_src_vocab is supplied
        orig_root = (
            Path(__file__).resolve().parent.parent
            / "original_fastspeech2"
            / "config"
            / "LJSpeech"
        )
        orig_cfg = load_fastspeech2_config(
            orig_root / "preprocess.yaml",
            orig_root / "model.yaml",
            orig_root / "train.yaml",
            n_src_vocab=361,
        )
        self.assertEqual(orig_cfg.preprocess.dataset, "LJSpeech")
        self.assertEqual(orig_cfg.model.n_src_vocab, 361)
        self.assertEqual(orig_cfg.model.transformer.encoder_hidden, 256)
        self.assertIsNotNone(orig_cfg.train)
        assert orig_cfg.train is not None
        self.assertEqual(orig_cfg.train.optimizer.batch_size, 16)

    def test_modular_and_backward_compatible_imports(self) -> None:
        """Verify modular submodule imports, backward-compatible package re-exports, and no circular imports."""
        import common
        import common.config as pkg_cfg
        from common.config import base as base_mod
        from common.config import loader as loader_mod
        from common.config import model as model_mod
        from common.config import preprocess as preprocess_mod
        from common.config import training as training_mod

        # Verify base submodule exports and identity with common.config and common
        self.assertIs(base_mod.ConfigNode, pkg_cfg.ConfigNode)
        self.assertIs(base_mod.ConfigNode, common.ConfigNode)
        self.assertIs(
            base_mod.ConfigValidationError, pkg_cfg.ConfigValidationError
        )
        self.assertIs(base_mod.ConfigValidationError, common.ConfigValidationError)
        self.assertEqual(
            base_mod.VALID_FEATURE_LEVELS, pkg_cfg.VALID_FEATURE_LEVELS
        )
        self.assertEqual(
            base_mod.VALID_QUANTIZATION_MODES, pkg_cfg.VALID_QUANTIZATION_MODES
        )

        # Verify preprocess submodule exports and identity
        for sym in (
            "PreprocessPathConfig",
            "TextConfig",
            "AudioConfig",
            "STFTConfig",
            "MelConfig",
            "PitchConfig",
            "EnergyConfig",
            "PreprocessingSettings",
            "PreprocessConfig",
        ):
            self.assertIs(getattr(preprocess_mod, sym), getattr(pkg_cfg, sym))
            self.assertIs(getattr(preprocess_mod, sym), getattr(common, sym))

        # Verify model submodule exports and identity
        for sym in (
            "TransformerConfig",
            "VariancePredictorConfig",
            "VarianceEmbeddingConfig",
            "PostNetConfig",
            "VocoderConfig",
            "ModelConfig",
        ):
            self.assertIs(getattr(model_mod, sym), getattr(pkg_cfg, sym))
            self.assertIs(getattr(model_mod, sym), getattr(common, sym))

        # Verify training submodule exports and identity
        for sym in (
            "TrainPathConfig",
            "OptimizerConfig",
            "StepConfig",
            "TrainConfig",
        ):
            self.assertIs(getattr(training_mod, sym), getattr(pkg_cfg, sym))
            self.assertIs(getattr(training_mod, sym), getattr(common, sym))

        # Verify loader submodule exports and identity
        for sym in (
            "FastSpeech2Config",
            "load_preprocess_config",
            "load_model_config",
            "load_train_config",
            "load_fastspeech2_config",
        ):
            self.assertIs(getattr(loader_mod, sym), getattr(pkg_cfg, sym))
            self.assertIs(getattr(loader_mod, sym), getattr(common, sym))


if __name__ == "__main__":
    unittest.main()

