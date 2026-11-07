"""Defines the vocabulary of symbols for Telugu FastSpeech 2 TTS.

FastSpeech 2 accepts discrete integer token sequences. For Telugu, the
phonetically transparent alphasyllabic (abugida) orthography is modeled via
canonical Unicode graphemes (elementary vowels, consonants, vowel matras,
virama, modifiers, and punctuation). This corresponds 1-to-1 with
AI4Bharat's IndicMFA Grapheme-to-Grapheme (G2G) alignment target tokens.

Special tokens:
- Index 0: `_` (PAD) — zero padding token conforming to FastSpeech 2 masking
- Index 1: `<unk>` (UNK) — out-of-vocabulary fallback
- Index 2: `<bos>` (BOS) — beginning of sequence
- Index 3: `<eos>` (EOS) — end of sequence
"""

from __future__ import annotations

# Special tokens
_pad = "_"
_unk = "<unk>"
_bos = "<bos>"
_eos = "<eos>"
_special = [_pad, _unk, _bos, _eos]

# Alignment silences (compatible with Montreal Forced Aligner / Kaldi)
_silences = ["@sp", "@spn", "@sil"]

# Standard punctuation
_punctuation = list(" !'(),-.:;?\"")

# Telugu Independent Vowels (అచ్చులు)
_telugu_vowels = [
    "అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ౠ",
    "ఌ", "ౡ", "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ"
]

# Telugu Consonants (హల్లులు)
_telugu_consonants = [
    "క", "ఖ", "గ", "ఘ", "ఙ",
    "చ", "ఛ", "జ", "ఝ", "ఞ",
    "ట", "ఠ", "డ", "ఢ", "ణ",
    "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ",
    "య", "ర", "ఱ", "ల", "ళ", "ఴ",
    "వ", "శ", "ష", "స", "హ"
]

# Telugu Dependent Vowel Signs / Matras (గుణింతాలు)
_telugu_matras = [
    "ా", "ి", "ీ", "ు", "ూ", "ృ", "ౄ",
    "ె", "ే", "ై", "ొ", "ో", "ౌ", "ౢ", "ౣ"
]

# Telugu Virama / Pollu (హలంతము - suppresses inherent vowel / creates conjuncts)
_telugu_virama = ["్"]

# Telugu Modifiers (అనుస్వారము, విసర్గ, అర్ధానుస్వారము, అవగ్రహము)
_telugu_modifiers = [
    "ం",  # Anusvara (sunna)
    "ః",  # Visarga
    "ఁ",  # Candrabindu (arasunna)
    "ఽ",  # Avagraha
]

# Telugu Numerals (సంఖ్యలు)
_telugu_digits = ["౦", "౧", "౨", "౩", "౪", "౫", "౬", "౭", "౮", "౯"]

# Combined full symbol inventory (ordered, deterministic)
symbols: list[str] = (
    _special
    + _silences
    + _punctuation
    + _telugu_vowels
    + _telugu_consonants
    + _telugu_matras
    + _telugu_virama
    + _telugu_modifiers
    + _telugu_digits
)

# Mappings
symbol_to_id: dict[str, int] = {s: i for i, s in enumerate(symbols)}
id_to_symbol: dict[int, str] = {i: s for i, s in enumerate(symbols)}

# Constants
PAD_ID = symbol_to_id[_pad]
UNK_ID = symbol_to_id[_unk]
BOS_ID = symbol_to_id[_bos]
EOS_ID = symbol_to_id[_eos]
VOCAB_SIZE = len(symbols)

