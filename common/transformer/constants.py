"""Special token constants for the FastSpeech 2 Transformer."""

from __future__ import annotations

PAD: int = 0
UNK: int = 1
BOS: int = 2
EOS: int = 3

PAD_WORD: str = "<blank>"
UNK_WORD: str = "<unk>"
BOS_WORD: str = "<s>"
EOS_WORD: str = "</s>"

__all__ = [
    "PAD",
    "UNK",
    "BOS",
    "EOS",
    "PAD_WORD",
    "UNK_WORD",
    "BOS_WORD",
    "EOS_WORD",
]

