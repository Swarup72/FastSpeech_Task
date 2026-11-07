"""Shared Transformer building blocks for FastSpeech 2 (`common.transformer`)."""

from __future__ import annotations

from common.transformer import constants
from common.transformer.constants import (
    BOS,
    BOS_WORD,
    EOS,
    EOS_WORD,
    PAD,
    PAD_WORD,
    UNK,
    UNK_WORD,
)
from common.transformer.layers import ConvNorm, FFTBlock, PostNet
from common.transformer.models import Decoder, Encoder
from common.transformer.modules import (
    PositionalEncoding,
    ScaledDotProductAttention,
    get_sinusoid_encoding_table,
)
from common.transformer.sublayers import (
    MultiHeadAttention,
    PositionwiseFeedForward,
)

__all__ = [
    "BOS",
    "BOS_WORD",
    "ConvNorm",
    "Decoder",
    "EOS",
    "EOS_WORD",
    "Encoder",
    "FFTBlock",
    "MultiHeadAttention",
    "PAD",
    "PAD_WORD",
    "PositionalEncoding",
    "PositionwiseFeedForward",
    "PostNet",
    "ScaledDotProductAttention",
    "UNK",
    "UNK_WORD",
    "constants",
    "get_sinusoid_encoding_table",
]

