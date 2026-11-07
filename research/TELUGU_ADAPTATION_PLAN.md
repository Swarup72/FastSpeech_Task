# Telugu & English Adaptation Plan (`TELUGU_ADAPTATION_PLAN.md`)

## 1. Existing Text Representation in `original_fastspeech2`

Based on source code inspection of `text/__init__.py`, `text/symbols.py`, `text/cleaners.py`, `text/cmudict.py`, `text/pinyin.py`, and `synthesize.py`:

1. **Symbol Table (`text/symbols.py`)**:
   - The vocabulary is a flat Python list `symbols` of 360 entries:
     - `_pad = "_"` (ID 0)
     - `_special = "-"` (ID 1)
     - `_punctuation = "!'(),.:;? "` (IDs 2–11)
     - `_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"` (IDs 12–63)
     - `_arpabet`: 84 English CMU ARPAbet phonemes prefixed with `"@"` (`@AA`, `@AA0`, ..., `@ZH`, IDs 64–147)
     - `_pinyin`: 209 Mandarin Pinyin initials/finals prefixed with `"@"` (IDs 148–356)
     - `_silences = ["@sp", "@spn", "@sil"]` (IDs 357–359)
2. **Sequence Conversion (`text/__init__.py`)**:
   - `text_to_sequence(text, cleaner_names)` uses `_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")` to detect phoneme sequences enclosed in curly braces `{...}`.
   - Inside `{...}`: text cleaners are **bypassed**, and `_arpabet_to_sequence()` splits the string on whitespace, prepends `"@"` to each token, and looks up `"@" + token` in `_symbol_to_id`.
   - Outside `{...}`: `_clean_text(text, cleaner_names)` runs first, and `_symbols_to_sequence()` maps each character to `_symbol_to_id`.
   - **Silent Filtering**: `_should_keep_symbol(s)` silently discards any token not found in `_symbol_to_id`.
3. **Text Cleaners (`text/cleaners.py`)**:
   - `english_cleaners(text)` runs `convert_to_ascii(text)` (`unidecode`), `lowercase(text)`, `expand_numbers(text)` (English words via `inflect`), `expand_abbreviations(text)`, and `collapse_whitespace(text)`.

---

## 2. Why the Existing Frontend Cannot Process Telugu Directly

1. **`unidecode` Destroys Telugu Script**:
   - Passing Telugu text (`తెలుగు`) through `english_cleaners` or `transliteration_cleaners` calls `unidecode()`, which strips Telugu Unicode codepoints (`U+0C00–U+0C7F`) into lossy ASCII approximations that lose phonemic distinctions (retroflex vs. dental stops, vowel length, aspiration) or fail to match the lexicon.
2. **Missing Telugu Symbols in `text/symbols.py`**:
   - Neither Telugu Unicode graphemes nor Indic phoneme tokens (e.g., IndicTTS Common Label Set / IPA / ARPAbet-like Telugu phones) exist in `text/symbols.py`. Any unmapped symbol is silently dropped by `_should_keep_symbol()`, producing empty input tensors (`len == 0`).
3. **Hardcoded Language Dispatch in `synthesize.py`**:
   - Lines 205–208 of `synthesize.py` only check `if language == "en": preprocess_english(...)` and `elif language == "zh": preprocess_mandarin(...)`. Any other language code (such as `"te"`) leaves `texts` undefined and raises an `UnboundLocalError`.
4. **Hardcoded Dataset Dispatch in `prepare_align.py`**:
   - `prepare_align.py` only checks for `"LJSpeech"`, `"AISHELL3"`, and `"LibriTTS"` in `config["dataset"]`.

---

## 3. Telugu Unicode & Linguistic Requirements

Telugu is an **abugida (akshara-based)** Dravidian language encoded in the **Telugu Unicode block (`U+0C00` – `U+0C7F`)**:

- **Independent Vowels (Achchulu)**: `అ` (`U+0C05`) through `ఔ` (`U+0C14`), covering short/long pairs (`a/aa`, `i/ii`, `u/uu`, `e/ee`, `o/oo`) and diphthongs (`ai`, `au`), plus vocalic `ఋ` (`U+0C0B`).
- **Consonants (Hallulu)**: `క` (`U+0C15`) through `హ` (`U+0C39`), including unaspirated/aspirated, voiceless/voiced stops across velar, palatal, retroflex, dental, and labial places of articulation, nasals (`ఙ, ఞ, ణ, న, మ`), approximants (`య, ర, ల, ళ, వ`), and fricatives (`శ, ష, స, హ`).
- **Dependent Vowel Signs (Matras / Gunintalu)**: `ా` (`U+0C3E`) through `ౌ` (`U+0C4C`).
- **Virama / Halant (Pollu)**: `్` (`U+0C4D`) — suppresses the inherent schwa/vowel `a` of a consonant and forms consonant clusters (**ottulu / conjuncts** such as `క్క`, `స్త`, `ప్ర`).
- **Modifiers (Yogavahalu)**: Candrabindu `ఁ` (`U+0C01`), Anusvara (Sunna) `ం` (`U+0C02`), Visarga `ః` (`U+0C03`).
- **Telugu Digits**: `౦`–`౯` (`U+0C66`–`U+0C6F`) plus ASCII digits `0`–`9`.

### Required Telugu Text Normalization (`telugu_cleaners`)
1. **Unicode Canonical Normalization (`unicodedata.normalize("NFC", text)`)**: Ensures composed sequences (such as `ఈ` or matra combinations) have a single canonical codepoint sequence across dataset transcripts and user inference queries.
2. **Zero-Width Character Removal**: Strip Zero-Width Non-Joiner (`\u200c`) and Zero-Width Joiner (`\u200d`) frequently found in IndicTTS transcripts.
3. **Number Expansion**: Convert ASCII (`0–9`) and Telugu (`౦–౯`) digits into spoken Telugu number words (or handle via Telugu text normalizer).
4. **Punctuation & Whitespace Normalization**: Map sentence-final punctuation/danda (`।`) to standard pause markers and collapse repeated whitespace, **without** calling `unidecode()` or ASCII `.lower()`.

---

## 4. Grapheme vs. Phoneme Strategy for Telugu

We compare the two candidate frontend representations for FastSpeech 2 on Telugu:

| Criterion | Option A: Phoneme / Common Label Set (CLS / IPA) *(Recommended)* | Option B: Direct Unicode Codepoint / Grapheme |
|---|---|---|
| **Inherent Vowel (`a`) & Virama (`్`) Handling** | Explicitly resolves inherent vowel `a` on bare consonants (`క` -> `k a`) and deletes it when followed by a matra (`కా` -> `k aa`) or virama (`క్` -> `k`). | A bare consonant `క` represents two acoustic segments (`[k]` + `[a]`), whereas `క్` (`క` + `్`) represents one segment (`[k]`), making per-token duration alignment mismatched with acoustic phone boundaries. |
| **Anusvara (`ం`) Assimilation** | Contextually maps `ం` (`U+0C02`) to homorganic nasal (`ng`, `nj`, `n`, `m`) before stops or `[m]` word-finally. | Single symbol `ం` has different acoustic realizations depending on the following stop. |
| **Compatibility with MFA & `Preprocessor`** | **100% compatible**: MFA aligns phoneme sequences from a lexicon (`word -> p1 p2 p3`), producing a `"phones"` tier in `.TextGrid` that `Preprocessor.get_alignment()` converts directly into `{p1 p2 p3}` and integer frame durations. | Requires character-level lexicon tricks in MFA and produces poor phone boundary alignments on consonant + virama clusters. |
| **OOV Generalization** | Because Telugu orthography is nearly phonemic (transparent grapheme-to-phoneme rules), a deterministic rule-based G2P (or `indic-nlp-library` / IndicTTS Unified Parser / `espeak-ng`) achieves >99% G2P accuracy on arbitrary unseen Telugu words. | Cannot disambiguate inherent vowel suppression without the model implicitly learning virama lookahead in the Encoder. |

### Selected Strategy: Deterministic Rule-Based Telugu G2P (Phoneme Representation)
- Use a **deterministic Telugu G2P function** (`telugu_g2p(word)`) that decomposes NFC-normalized Telugu Unicode strings into a standardized Telugu phoneme set (~52 phonemes: short/long vowels, diphthongs, velar/palatal/retroflex/dental/labial stops, nasals, liquids, glides, sibilants, plus `@sp`, `@spn`, `@sil`).
- During `prepare_align.py`:
  - Clean the Telugu transcript with `telugu_cleaners`.
  - Generate the MFA pronunciation dictionary (`lexicon/telugu-lexicon.txt`) automatically by running `telugu_g2p(word)` over all unique Telugu words in the IndicTTS Telugu corpus.
- During `preprocess.py`:
  - MFA outputs `.TextGrid` files with the `"phones"` tier containing Telugu phoneme labels (e.g., `TEL_k`, `TEL_a`, `TEL_l`, `TEL_u`, ...).
  - `Preprocessor` writes `{TEL_k TEL_a ...}` into `train.txt` and `val.txt`.
- During `synthesize.py`:
  - `preprocess_telugu(text, preprocess_config)` normalizes input Telugu text, looks up words in `telugu-lexicon.txt` (falling back to `telugu_g2p(w)` for any OOV word), formats `{TEL_...}`, and calls `text_to_sequence()`.

---

## 5. Dataset Integration Plan (Svarah English & IndicTTS Telugu)

### 5.1 English Model — Svarah Dataset
- **Dataset Characteristics**: Indian-accented English speech dataset (`Svarah`), typically distributed with audio files (`.wav`, often 16 kHz or 22.05 kHz, multiple speakers or subset) and manifest/metadata files.
- **Required Integration**:
  1. Create `preprocessor/svarah.py` to read Svarah metadata, normalize text using `english_cleaners`, resample audio explicitly via `librosa.load(wav_path, sr=sampling_rate)` to `22050` Hz (to match HiFi-GAN universal vocoder), and output `raw_data/Svarah/<speaker>/<basename>.{wav,lab}`.
  2. Set `multi_speaker: True` (if training across Svarah speakers) and `vocoder.speaker: "universal"` in `config/Svarah/model.yaml`.
  3. Generate MFA alignments using `librispeech-lexicon.txt` (extended with `g2p_en` for Indian-English proper nouns in Svarah).

### 5.2 Telugu Model — IndicTTS Telugu Dataset
- **Dataset Characteristics**: IndicTTS Telugu corpus (typically single-speaker male and/or female studio recordings with `txt.done.data` or `metadata.csv` in Festival/UTF-8 format and `wav/*.wav`).
- **Required Integration**:
  1. Create `preprocessor/indictts_telugu.py` to parse IndicTTS metadata (`txt.done.data` or CSV/TSV), apply `telugu_cleaners`, resample `.wav` to `22050` Hz, write `raw_data/IndicTTS_Telugu/<speaker>/<basename>.{wav,lab}`, and build `lexicon/telugu-lexicon.txt` using `telugu_g2p`.
  2. Run `mfa_train_and_align` (or `mfa train` in MFA 2.x/3.x on Colab) to train a Telugu acoustic alignment model from scratch on `raw_data/IndicTTS_Telugu/` and `lexicon/telugu-lexicon.txt` (no pre-trained Telugu MFA model is required when using `mfa train`!).
  3. Set `vocoder.speaker: "universal"` in `config/IndicTTS_Telugu/model.yaml` (universal HiFi-GAN is language-agnostic because it synthesizes waveforms purely from 80-bin Mel-spectrograms).

---

## 6. Repository Modification Plan (Phase G Table)

> **Important Rule**: `original_fastspeech2/` will remain **100% untouched**. All modifications below will be implemented in our modular project directories (`common/`, `English/`, `Telugu/`).

| File (Relative to Codebase) | Existing Purpose | English Changes (`Svarah`) | Telugu Changes (`IndicTTS Telugu`) | Priority |
|---|---|---|---|---|
| `text/symbols.py` | Defines global symbol list (`_pad`, `_letters`, `_arpabet`, `_pinyin`, `_silences`) | No change needed (uses 84 CMU `_arpabet` symbols) | Add `_telugu` phoneme symbol list (e.g., `@te_a`, `@te_aa`, ..., `@te_ha`) exported in `symbols` | **P0 (Critical)** |
| `text/cleaners.py` | Text normalization pipelines (`english_cleaners`, `basic_cleaners`, `transliteration_cleaners`) | Keep `english_cleaners` | Add `telugu_cleaners()` (NFC Unicode normalization, ZWNJ/ZWJ removal, Telugu digit expansion, whitespace cleanup; **no `unidecode`**) | **P0 (Critical)** |
| `text/telugu.py` *(New File)* | N/A | N/A | Define `valid_symbols` for Telugu phoneme inventory and `telugu_g2p(word)` rule-based grapheme-to-phoneme converter for `\u0C00–\u0C7F` | **P0 (Critical)** |
| `prepare_align.py` | Dispatches `prepare_align(config)` for `LJSpeech`, `AISHELL3`, `LibriTTS` | Add dispatch for `"Svarah"` -> `svarah.prepare_align(config)` | Add dispatch for `"IndicTTS_Telugu"` / `"Telugu"` -> `indictts_telugu.prepare_align(config)` | **P0 (Critical)** |
| `preprocessor/svarah.py` *(New File)* | N/A | Parse Svarah manifest, run `english_cleaners`, resample audio to 22,050 Hz, write `.wav`/`.lab`, and augment lexicon with `g2p_en` OOV words | N/A | **P0 (Critical)** |
| `preprocessor/indictts_telugu.py` *(New File)* | N/A | N/A | Parse IndicTTS Telugu transcripts (`txt.done.data`/CSV), run `telugu_cleaners`, resample `.wav` to 22,050 Hz, write `.wav`/`.lab`, and generate `lexicon/telugu-lexicon.txt` | **P0 (Critical)** |
| `preprocessor/preprocessor.py` | Extracts MFA durations, DIO+StoneMask F0, STFT energy, Mel-spectrograms, and writes `train.txt`, `val.txt`, `stats.json` | Fix `librosa.load(wav_path, sr=self.sampling_rate)` on line 172; handle utterances with leading/trailing silence edge cases | Same `sr=self.sampling_rate` fix; ensure `encoding="utf-8"` when reading `.lab` files on line 178 (`open(text_path, "r", encoding="utf-8")`) | **P0 (Critical)** |
| `audio/stft.py` | Computes STFT & 80-bin Mel-spectrograms + frame energy | Replace hardcoded `.cuda()` on lines 68–72 and 111 with device-agnostic tensor ops; update `librosa_mel_fn` & `pad_center` calls for modern `librosa` compatibility | Same device-agnostic & `librosa` keyword argument fixes | **P0 (Critical)** |
| `model/modules.py` | `VarianceAdaptor`, `LengthRegulator`, `VariancePredictor` | Fix line 124 bug where `p_control` is passed instead of `e_control` in `get_energy_embedding()`; make `device` dynamic (`x.device`) instead of module-global | Same `e_control` bug fix and dynamic `x.device` handling | **P1 (High)** |
| `synthesize.py` | CLI for single-sentence and batch TTS inference | Support `language == "en"` with Svarah lexicon; add option to export predicted duration/pitch/energy arrays | Add `elif language == "te": preprocess_telugu(args.text, preprocess_config)` using `telugu-lexicon.txt` + `telugu_g2p` fallback | **P0 (Critical)** |
| `utils/tools.py` | `synth_samples()`, `synth_one_sample()`, `plot_mel()`, device helpers | Extend `synth_samples()` to save predicted `duration` (`output[5]`), de-normalized `pitch` F0 (`output[2]`), and de-normalized `energy` (`output[3]`) as `.npy`/`.json` and dedicated plots in `Results/English/` | Same extraction & plotting in `Results/Telugu/` (avoid rendering raw Telugu glyphs in matplotlib titles unless a Telugu-capable font is configured) | **P1 (High)** |
| `utils/model.py` | Loads FastSpeech2 checkpoints and HiFi-GAN/MelGAN vocoder | Add `map_location=device` to all `torch.load()` calls (lines 20, 63, 65) so Colab GPU checkpoints load on CPU/MPS | Same `map_location=device` fix; use `generator_universal.pth.tar` | **P1 (High)** |
| `train.py` | Main training loop | Make `nn.DataParallel` conditional or unwrap nicely; lower `save_step` (e.g., `5000`–`10000`) for Colab session safety | Same checkpoint frequency and device fixes | **P1 (High)** |
| `config/{Svarah,Telugu}/*.yaml` *(New Configs)* | Dataset/model/training hyperparameters | Create `English/config/{preprocess,model,train}.yaml` (`language: "en"`, `vocoder.speaker: "universal"`) | Create `Telugu/config/{preprocess,model,train}.yaml` (`language: "te"`, `text_cleaners: ["telugu_cleaners"]`, `vocoder.speaker: "universal"`) | **P0 (Critical)** |

---

## 7. Technical Risks & Proposed Solutions

| Risk | Root Cause | Proposed Solution |
|---|---|---|
| **Silent Token Dropping in Telugu** | `_should_keep_symbol(s)` in `text/__init__.py` silently drops any phoneme not in `text/symbols.py` | Add an explicit assertion/warning in `text_to_sequence()` during preprocessing & inference if any token in `{...}` is not in `_symbol_to_id`, and unit-test `telugu_g2p` coverage on 100% of the IndicTTS Telugu vocabulary before running MFA. |
| **UTF-8 File Encoding on Non-ASCII Text** | `preprocessor/preprocessor.py` line 178 (`with open(text_path, "r") as f:`) and `synthesize.py` line 22 (`with open(lex_path) as f:`) omit `encoding="utf-8"` | Explicitly specify `encoding="utf-8"` on all file reads/writes in `preprocessor.py`, `indictts_telugu.py`, and `synthesize.py`. |
| **MFA Alignment on Telugu Without Pretrained Acoustic Model** | Official MFA pretrained models focus on major European/Mandarin languages | Use MFA's acoustic training mode (`mfa train` / `mfa_train_and_align`) directly on the IndicTTS Telugu corpus + our generated `telugu-lexicon.txt` on Google Colab. Because IndicTTS Telugu has clean studio audio and phonemic orthography, flat-start GMM-HMM alignment converges reliably. |
| **Colab Session Timeouts During Training** | Default `train.yaml` sets `save_step: 100000` and `total_step: 900000` | Set `save_step: 5000` (or `10000`) and mount Google Drive for `ckpt_path` and `log_path` so training can resume at any step via `--restore_step`. |
| **Local 8 GB Unified Memory Limit** | Running MFA or large-batch training locally on an 8 GB M2 MacBook causes swap thrashing | Perform only code development, unit tests, 5-utterance smoke tests, and final checkpoint inference locally; run full MFA alignment, full preprocessing, and GPU training on Google Colab. |

