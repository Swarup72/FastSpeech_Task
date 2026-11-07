"""Telugu text normalization and tokenization for FastSpeech 2 TTS.

Performs:
1. Canonical Unicode NFC composition (essential for composite Indic matras).
2. Normalization of non-printing zero-width joiners (ZWJ) and non-joiners (ZWNJ).
3. Punctuation standardization (dandas, curly quotes, dashes to canonical forms).
4. Whitespace cleanup (collapsing multiple spaces, stripping newlines).
5. Grapheme tokenization and symbol sequence mapping.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence

from Telugu.text.symbols import (
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
    id_to_symbol,
    symbol_to_id,
    symbols,
)

# Punctuation and formatting replacements
_PUNCTUATION_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\u2018\u2019\u201A\u201B]"), "'"),  # Curly single quotes
    (re.compile(r"[\u201C\u201D\u201E\u201F]"), '"'),  # Curly double quotes
    (re.compile(r"[\u2013\u2014]"), "-"),  # En/em dashes
    (re.compile(r"\u2026"), "..."),  # Ellipsis
    (re.compile(r"[\u0964\u0965]"), "."),  # Devanagari/Bengali dandas to full stop
    (re.compile(r"[\r\n\t\f\v]"), " "),  # Control whitespaces
    (re.compile(r"[\u200B\u200C\u200D\uFEFF]"), ""),  # Zero-width spaces/joiners
]

_MULTI_SPACE_REGEX = re.compile(r" {2,}")

# Contextual replacement for Latin 'o' typo in place of Telugu Anusvara (ం)
_LATIN_O_ANUSVARA_REGEX = re.compile(
    r"(?<=[\u0C00-\u0C7F])o(?=[\u0C00-\u0C7F]|\b|[,.!?;:\"'])"
)


def normalize_text(text: str) -> str:
    """Normalize a raw Telugu text string into canonical FastSpeech 2 input format.

    Args:
        text: Raw Telugu text.

    Returns:
        Cleaned, NFC-normalized string.
    """
    if not text:
        return ""

    # 1. Unicode NFC normalization (composes vowels and matras canonically)
    text = unicodedata.normalize("NFC", text)

    # 2. Contextual correction of Latin 'o' typo for Telugu Anusvara (ం)
    text = _LATIN_O_ANUSVARA_REGEX.sub("ం", text)

    # 3. Apply punctuation replacements
    for pattern, repl in _PUNCTUATION_REPLACEMENTS:
        text = pattern.sub(repl, text)

    # 3. Collapse multiple whitespaces
    text = _MULTI_SPACE_REGEX.sub(" ", text)

    # 4. Strip leading/trailing whitespaces
    text = text.strip()

    return text


_CURLY_TOKEN_RE = re.compile(r"^\{(.+)\}$")


def text_to_sequence(
    text: str,
    add_bos: bool | Sequence[str] = False,
    add_eos: bool = False,
) -> list[int]:
    """Convert normalized Telugu text or curly-brace aligned token sequence into symbol IDs.

    Supports two input modes:
    1. Preprocessed aligned token sequence enclosed in curly braces, e.g.
       `"{వ ై క ు ం ఠ @sp ర ా వ ు}"`: splits space-separated tokens and maps each
       directly via `symbol_to_id` (matching the exact length of `duration.npy`).
    2. Raw or normalized Telugu text string, e.g. `"నరకవాసి."`: normalizes text
       and tokenizes character-by-character into Telugu grapheme IDs.

    Args:
        text: Telugu text string or `{tok1 tok2 ...}` aligned token sequence.
        add_bos: Whether to prepend BOS token (or `cleaner_names` list/tuple when
            invoked by `Dataset.__getitem__`, in which case BOS is not prepended).
        add_eos: Whether to append EOS token.

    Returns:
        List of integer symbol IDs.
    """
    use_bos = False if isinstance(add_bos, (list, tuple)) else bool(add_bos)
    stripped = text.strip() if text else ""
    curly_match = _CURLY_TOKEN_RE.match(stripped)

    sequence: list[int] = []
    if use_bos:
        sequence.append(BOS_ID)

    if curly_match:
        for tok in curly_match.group(1).split():
            if tok in symbol_to_id:
                sequence.append(symbol_to_id[tok])
            elif f"@{tok}" in symbol_to_id:
                sequence.append(symbol_to_id[f"@{tok}"])
            else:
                sequence.append(UNK_ID)
    else:
        norm_text = normalize_text(text)
        for char in norm_text:
            token_id = symbol_to_id.get(char, UNK_ID)
            sequence.append(token_id)

    if add_eos:
        sequence.append(EOS_ID)

    return sequence


def sequence_to_text(sequence: Sequence[int]) -> str:
    """Decode a sequence of integer symbol IDs back into a Telugu string.

    Args:
        sequence: Sequence of token IDs.

    Returns:
        Reconstructed string (ignoring special tokens).
    """
    chars: list[str] = []
    for token_id in sequence:
        sym = id_to_symbol.get(token_id, "")
        if sym in {"_", "<unk>", "<bos>", "<eos>", "@sp", "@spn", "@sil"}:
            continue
        chars.append(sym)
    return "".join(chars)

