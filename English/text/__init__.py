"""English text normalization, phonemization, and token conversion (`English.text`)."""
from __future__ import annotations

import re
from typing import Sequence

from . import cleaners
from .normalizer import normalize_text
from .symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text: str, cleaner_names: Sequence[str] | None = None) -> list[int]:
    """Converts a string of text to a sequence of IDs corresponding to symbols in the text.

    Supports embedded ARPAbet in curly braces (e.g. "{HH AW1 S S T AH0 N}").
    """
    if cleaner_names is None:
        cleaner_names = ["english_cleaners"]
    sequence: list[int] = []

    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence


def sequence_to_text(sequence: Sequence[int]) -> str:
    """Converts a sequence of integer IDs back to a string."""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text: str, cleaner_names: Sequence[str]) -> str:
    for name in cleaner_names:
        cleaner = getattr(cleaners, name, None)
        if not cleaner:
            raise ValueError(f"Unknown cleaner: {name}")
        text = cleaner(text)
    return text


def _symbols_to_sequence(syms: Sequence[str]) -> list[int]:
    return [_symbol_to_id[s] for s in syms if _should_keep_symbol(s)]


def _arpabet_to_sequence(text: str) -> list[int]:
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s: str) -> bool:
    return s in _symbol_to_id and s != "_" and s != "~"


__all__ = [
    "symbols",
    "text_to_sequence",
    "sequence_to_text",
    "normalize_text",
    "cleaners",
]
