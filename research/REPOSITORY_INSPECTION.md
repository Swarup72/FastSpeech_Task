# FastSpeech 2 Repository Technical Inspection (`ming024/FastSpeech2`)

## 1. Repository Overview

- **Source Repository**: `https://github.com/ming024/FastSpeech2.git`
- **Local Clone Path**: `/Users/swaruppal/FastSpeech2_Project/original_fastspeech2`
- **Branch**: `master`
- **HEAD Commit**: `d4e79eb52e8b01d24703b2dfc0385544092958f3` (`d4e79eb` — *"update README and demo page"*, Thu Jul 8 11:40:51 2021 +0800)
- **Working Tree Status**: Clean (`nothing to commit, working tree clean`)
- **Paper Reference**: *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (Ren et al., 2020, v1 — direct F0 prediction rather than CWT pitch spectrogram), extended with a Tacotron-2-style 5-layer convolutional `PostNet` and phoneme-level variance prediction (`Chien et al., ICASSP 2021`).
- **Out-of-the-Box Datasets Supported**:
  - `LJSpeech` (English, single-speaker, 22,050 Hz)
  - `LibriTTS` (English, multi-speaker, 22,050 Hz)
  - `AISHELL3` (Mandarin Chinese, multi-speaker, 22,050 Hz)

### Repository Directory Structure

```text
original_fastspeech2/
├── audio/
│   ├── __init__.py
│   ├── audio_processing.py     # Window sum-square, Griffin-Lim, log dynamic range compression
│   ├── stft.py                 # STFT and TacotronSTFT (Mel-spectrogram + L2 frame energy)
│   └── tools.py                # get_mel_from_wav(), inv_mel_spec()
├── config/
│   ├── AISHELL3/               # preprocess.yaml, model.yaml, train.yaml
│   ├── LJSpeech/               # preprocess.yaml, model.yaml, train.yaml
│   ├── LJSpeech_paper/         # Frame-level unnormalized pitch/energy config
│   ├── LibriTTS/               # Multi-speaker English config
│   └── README.md
├── hifigan/
│   ├── __init__.py
│   ├── config.json             # Universal/LJSpeech HiFi-GAN V1 architecture config
│   ├── env.py
│   ├── generator_LJSpeech.pth.tar.zip
│   ├── generator_universal.pth.tar.zip
│   └── models.py               # HiFi-GAN Generator & ResBlock
├── lexicon/
│   ├── librispeech-lexicon.txt # English word -> ARPAbet phoneme lexicon for MFA & inference
│   └── pinyin-lexicon-r.txt    # Mandarin Pinyin -> initial/final lexicon
├── model/
│   ├── __init__.py
│   ├── fastspeech2.py          # Top-level FastSpeech2 nn.Module
│   ├── loss.py                 # FastSpeech2Loss (MAE Mel/PostNet + MSE Pitch/Energy/Log-Duration)
│   ├── modules.py              # VarianceAdaptor, LengthRegulator, VariancePredictor, Conv
│   └── optimizer.py            # ScheduledOptim (Transformer Noam warmup + step annealing)
├── preprocessed_data/          # Pre-included speakers.json, stats.json, train.txt, val.txt
├── preprocessor/
│   ├── aishell3.py             # prepare_align() for AISHELL-3
│   ├── libritts.py             # prepare_align() for LibriTTS
│   ├── ljspeech.py             # prepare_align() for LJSpeech
│   └── preprocessor.py         # Preprocessor class: MFA TextGrid parsing, DIO+StoneMask F0, Energy, Mel
├── text/
│   ├── __init__.py             # text_to_sequence(), sequence_to_text(), ARPAbet curly-brace parser
│   ├── cleaners.py             # english_cleaners, transliteration_cleaners, basic_cleaners
│   ├── cmudict.py              # 84 valid ARPAbet symbols + CMUDict parser
│   ├── numbers.py              # English number-to-words normalization via inflect
│   ├── pinyin.py               # Mandarin initials/finals symbol list
│   └── symbols.py              # Global symbol table (360 symbols total)
├── transformer/
│   ├── Constants.py            # PAD = 0, UNK = 1, BOS = 2, EOS = 3
│   ├── Layers.py               # FFTBlock and PostNet
│   ├── Models.py               # Encoder, Decoder, get_sinusoid_encoding_table()
│   ├── Modules.py              # ScaledDotProductAttention
│   └── SubLayers.py            # MultiHeadAttention, PositionwiseFeedForward (1D Conv)
├── utils/
│   ├── model.py                # get_model(), get_vocoder(), vocoder_infer(), get_param_num()
│   └── tools.py                # to_device(), log(), synth_one_sample(), synth_samples(), plot_mel(), pad_*
├── dataset.py                  # PyTorch Dataset (training/eval) and TextDataset (batch inference)
├── evaluate.py                 # Validation loop over val.txt
├── prepare_align.py            # Entry point for resampling audio & writing .lab files for MFA
├── preprocess.py               # Entry point for feature extraction from MFA TextGrids
├── requirements.txt            # Original 2020 pinned dependencies
├── synthesize.py               # Entry point for single-sentence or batch TTS inference
└── train.py                    # Entry point for model training
```

---

## 2. Model Architecture Inspection

The FastSpeech 2 architecture is implemented across `model/fastspeech2.py`, `model/modules.py`, `transformer/Models.py`, `transformer/Layers.py`, and `transformer/SubLayers.py`.

### 2.1 Top-Level Module: `FastSpeech2` (`model/fastspeech2.py`)
- **Constructor (`__init__`)**:
  1. `self.encoder = Encoder(model_config)`
  2. `self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)`
  3. `self.decoder = Decoder(model_config)`
  4. `self.mel_linear = nn.Linear(decoder_hidden, n_mel_channels)` (projects 256 -> 80)
  5. `self.postnet = PostNet()` (5-layer 1D Conv residual refinement network)
  6. `self.speaker_emb`: If `model_config["multi_speaker"]` is `True`, loads `speakers.json` and creates `nn.Embedding(n_speaker, encoder_hidden)` added to the encoder output (`output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)`).

### 2.2 Component-by-Component Input/Output Analysis

| # | Component | Source File & Class | Input Tensors & Shapes | Internal Architecture | Output Tensors & Shapes |
|---|---|---|---|---|---|
| 1 | **Text Encoder** | `transformer/Models.py` (`Encoder`), `transformer/Layers.py` (`FFTBlock`) | `src_seq`: `[B, L_src]` (LongTensor token IDs)<br>`mask`: `[B, L_src]` (BoolTensor, `True` at padded positions) | Token embedding `nn.Embedding(len(symbols)+1, 256, padding_idx=0)` + fixed sinusoidal positional encoding (`max_seq_len+1 = 1001`), followed by **4 `FFTBlock` layers**. Each `FFTBlock` uses 2-head `MultiHeadAttention` (`d_k=d_v=128`) + residual `LayerNorm` + `PositionwiseFeedForward` (two 1D convolutions: kernel 9 `256->1024` with ReLU, then kernel 1 `1024->256`) + residual `LayerNorm`, zero-masking padded positions after each sublayer. | `enc_output`: `[B, L_src, 256]` |
| 2 | **Variance Adaptor** | `model/modules.py` (`VarianceAdaptor`) | `x`: `[B, L_src, 256]` (Encoder output + optional speaker embedding)<br>`src_mask`: `[B, L_src]`<br>`mel_mask`: `[B, L_mel]` (or `None` in inference)<br>`pitch_target`, `energy_target`, `duration_target` (training) or `p_control`, `e_control`, `d_control` (inference) | Orchestrates duration prediction, phoneme-level (default) or frame-level pitch & energy prediction, quantization via `torch.bucketize` into 256 bins (`n_bins=256`, linear or log spaced between `stats.json` min/max), adds `pitch_embedding` (`[B, L, 256]`) and `energy_embedding` (`[B, L, 256]`) to `x`, and expands phoneme hidden states to frame level via `LengthRegulator`. | `x`: `[B, L_mel, 256]`<br>`pitch_prediction`: `[B, L_src]` (or `[B, L_mel]`)<br>`energy_prediction`: `[B, L_src]` (or `[B, L_mel]`)<br>`log_duration_prediction`: `[B, L_src]`<br>`duration_rounded`: `[B, L_src]`<br>`mel_len`: `[B]`<br>`mel_mask`: `[B, L_mel]` |
| 3 | **Duration Predictor** | `model/modules.py` (`VariancePredictor` as `self.duration_predictor`) | `x`: `[B, L_src, 256]`<br>`src_mask`: `[B, L_src]` | 2-layer 1D Conv stack (`256 -> 256`, kernel=3, ReLU, LayerNorm, Dropout=0.5) + `nn.Linear(256, 1)` + `.squeeze(-1)` + `masked_fill(mask, 0.0)`. Predicts durations in the **log domain**: $\hat{d}_{\text{log}} = \log(d + 1)$. | `log_duration_prediction`: `[B, L_src]` (FloatTensor) |
| 4 | **Pitch Predictor** | `model/modules.py` (`VariancePredictor` as `self.pitch_predictor` + `get_pitch_embedding`) | `x`: `[B, L_src, 256]` (if `phoneme_level`) or `[B, L_mel, 256]` (if `frame_level`)<br>`mask`: `src_mask` or `mel_mask` | Identical `VariancePredictor` architecture (`Conv1d(k=3) -> ReLU -> LayerNorm -> Dropout(0.5)` x2 -> `Linear(256, 1)`). Output scalar F0 (normalized by default) is bucketized against `self.pitch_bins` (`n_bins - 1 = 255` boundaries) and embedded via `self.pitch_embedding = nn.Embedding(256, 256)`. During training, ground-truth `pitch_target` is bucketized (teacher forcing); during inference, `prediction * p_control` is bucketized. | `pitch_prediction`: `[B, L_src]` (or `[B, L_mel]`)<br>`pitch_embedding`: `[B, L_src, 256]` (added to `x`) |
| 5 | **Energy Predictor** | `model/modules.py` (`VariancePredictor` as `self.energy_predictor` + `get_energy_embedding`) | `x`: `[B, L_src, 256]` (if `phoneme_level`) or `[B, L_mel, 256]` (if `frame_level`)<br>`mask`: `src_mask` or `mel_mask` | Identical `VariancePredictor` architecture. Output energy is bucketized against `self.energy_bins` and embedded via `self.energy_embedding = nn.Embedding(256, 256)`. Note: in `VarianceAdaptor.forward` line 124, `self.get_energy_embedding(x, energy_target, src_mask, p_control)` contains an upstream bug where `p_control` was passed instead of `e_control` for `phoneme_level` energy! | `energy_prediction`: `[B, L_src]` (or `[B, L_mel]`)<br>`energy_embedding`: `[B, L_src, 256]` (added to `x`) |
| 6 | **Length Regulator** | `model/modules.py` (`LengthRegulator`) | `x`: `[B, L_src, 256]`<br>`duration`: `[B, L_src]` (ground-truth `duration_target` in training; `clamp(round((exp(log_d) - 1) * d_control), min=0)` in inference)<br>`max_len`: scalar int or `None` | For each item in batch and each token $i$, repeats hidden vector `vec` `max(int(duration[i]), 0)` times along the time axis (`vec.expand(...)`), concatenates along time, and pads batch sequences to `max_len` with `0.0`. | `output`: `[B, L_mel, 256]`<br>`mel_len`: `[B]` (LongTensor) |
| 7 | **Mel-Spectrogram Decoder** | `transformer/Models.py` (`Decoder`) + `model/fastspeech2.py` (`self.mel_linear`) | `enc_seq`: `[B, L_mel, 256]`<br>`mask`: `mel_mask` `[B, L_mel]` | Adds sinusoidal positional encoding (`position_enc`, dynamically extended if `L_mel > max_seq_len` at inference) and passes through **6 `FFTBlock` layers** (2 heads, hidden 256, Conv1D filter 1024, kernels `[9, 1]`, dropout 0.2), followed by `self.mel_linear = nn.Linear(256, 80)`. | `output` (pre-PostNet Mel): `[B, L_mel, 80]` |
| 8 | **PostNet** | `transformer/Layers.py` (`PostNet`) + `model/fastspeech2.py` (line 97) | `output`: `[B, L_mel, 80]` | 5-layer 1D Conv stack: layers 1–4 are `Conv1d(in, 512, k=5, pad=2) -> BatchNorm1d(512) -> tanh -> Dropout(0.5)`; layer 5 is `Conv1d(512, 80, k=5, pad=2) -> BatchNorm1d(80) -> Dropout(0.5)`. Added residually to pre-PostNet Mel: `postnet_output = self.postnet(output) + output`. | `postnet_output`: `[B, L_mel, 80]` |
| 9 | **Loss Function** | `model/loss.py` (`FastSpeech2Loss`) | `inputs`: 12-element batch tuple (`inputs[6:]` has `mel_targets`, `mel_lens`, `max_mel_len`, `pitch_targets`, `energy_targets`, `duration_targets`)<br>`predictions`: 10-element tuple from `FastSpeech2.forward` | Inverts masks (`src_masks = ~src_masks`, `mel_masks = ~mel_masks`) to select only non-padded elements via `masked_select`:\n- `mel_loss = nn.L1Loss()(mel_predictions, mel_targets)`\n- `postnet_mel_loss = nn.L1Loss()(postnet_mel_predictions, mel_targets)`\n- `pitch_loss = nn.MSELoss()(pitch_predictions, pitch_targets)`\n- `energy_loss = nn.MSELoss()(energy_predictions, energy_targets)`\n- `duration_loss = nn.MSELoss()(log_duration_predictions, torch.log(duration_targets.float() + 1))`\n- `total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss` | 6-tuple of scalar losses: `(total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss)` |

---

## 3. Text Processing Inspection (`text/`)

1. **Graphemes vs. Phonemes**:
   - The model is strictly **phoneme-based** during both training and inference for English and Mandarin.
   - During preprocessing (`preprocessor/preprocessor.py` line 167), phoneme sequences extracted from MFA TextGrids are formatted with curly braces, e.g., `"{DH IY0 S OW1 V IY0 AH0 T ...}"`.
   - In `text/__init__.py`, `text_to_sequence(text, cleaner_names)` matches `{...}` blocks using `_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")` and routes their contents to `_arpabet_to_sequence()`, which prepends `"@"` to each space-separated token (`["@" + s for s in text.split()]`) and maps each symbol to its integer ID in `_symbol_to_id`.
2. **English Representation**:
   - Raw English text is normalized via `english_cleaners` (`text/cleaners.py`): `convert_to_ascii` (`unidecode`) -> `lowercase` -> `expand_numbers` (`text/numbers.py` using `inflect`) -> `expand_abbreviations` -> `collapse_whitespace`.
   - Words are mapped to **ARPAbet** phonemes (84 symbols in `text/cmudict.py`, such as `AA0`, `AA1`, `AE1`, `B`, `CH`, etc., stored in `symbols.py` with `"@"` prefix: `@AA0`, `@B`, etc.).
   - During inference (`synthesize.py` `preprocess_english`), words are looked up in `lexicon/librispeech-lexicon.txt` first, falling back to `g2p_en.G2p()` for out-of-vocabulary words.
3. **Mandarin Representation**:
   - Handled via `pypinyin` (`Style.TONE3`, `neutral_tone_with_five=True`) in `synthesize.py` (`preprocess_mandarin`), which converts Chinese characters into tone-numbered Pinyin syllables and looks them up in `lexicon/pinyin-lexicon-r.txt` to split into Mandarin initials (`23` symbols) and finals (`185` symbols) + `"rr"`, all included in `text/pinyin.py` and prefixed with `"@"` in `text/symbols.py`.
4. **Vocabulary Generation (`text/symbols.py`)**:
   - Static list `symbols` combining:
     - `_pad = "_"` (1 symbol, ID `0`)
     - `_special = "-"` (1 symbol, ID `1`)
     - `_punctuation = "!'(),.:;? "` (10 symbols, IDs `2..11`)
     - `_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"` (52 symbols, IDs `12..63`)
     - `_arpabet = ["@" + s for s in cmudict.valid_symbols]` (84 symbols, IDs `64..147`)
     - `_pinyin = ["@" + s for s in pinyin.valid_symbols]` (209 symbols, IDs `148..356`)
     - `_silences = ["@sp", "@spn", "@sil"]` (3 symbols, IDs `357..359`)
   - Total vocabulary size: `len(symbols) = 360` (`n_src_vocab = len(symbols) + 1 = 361` in `Encoder.__init__`).
5. **Handling of Unknown Tokens**:
   - In `text/__init__.py` lines 66–75:
     ```python
     def _symbols_to_sequence(symbols):
         return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

     def _should_keep_symbol(s):
         return s in _symbol_to_id and s != "_" and s != "~"
     ```
   - **Critical observation**: Unknown symbols not present in `_symbol_to_id` are **silently dropped** (filtered out)! In MFA alignments, spoken noise / OOV words are labeled `spn` (mapped to `@spn` in `_silences`). If a new language's phonemes or graphemes are not added to `symbols.py`, `_symbols_to_sequence` will return an empty list `[]`!
6. **Difficulty of Introducing Telugu**:
   - **Moderate and well-scoped**, provided we carefully modify the text frontend (`text/symbols.py`, `text/cleaners.py`, `synthesize.py`) and alignment/preprocessing pipeline:
     - Calling `english_cleaners` or `transliteration_cleaners` on Telugu script (`\u0C00–\u0C7F`) would invoke `unidecode()`, destroying Telugu characters by lossily transliterating them to ASCII.
     - If using **phoneme-based** Telugu (e.g., IndicTTS/MFA phoneme set or IPA/ISO-15919 phones from MFA TextGrids), the Telugu phone inventory must be registered in `text/symbols.py` with `"@"` prefixes so `{phone1 phone2 ...}` sequences are retained by `_arpabet_to_sequence()`.
     - If using **grapheme/akshara-based** or hybrid Telugu without curly braces, Telugu Unicode codepoints (`\u0C00–\u0C7F`) must be added to `symbols.py` and a `telugu_cleaners` function (performing Unicode NFC normalization and whitespace cleanup without `unidecode` or ASCII lowercasing) must be added to `text/cleaners.py`.

---

## 4. Preprocessing Pipeline Inspection (`prepare_align.py`, `preprocess.py`, `preprocessor/`, `audio/`)

Preprocessing is a two-stage pipeline:

### Stage 1: `prepare_align.py` (`preprocessor/{ljspeech,libritts,aishell3}.py`)
- Reads raw dataset transcripts and audio files.
- Loads audio with `librosa.load(wav_path, sampling_rate)`, peak-normalizes to `max_wav_value` (`32768.0`), and writes 16-bit PCM `.wav` files alongside cleaned `.lab` transcript files into `raw_data/<Dataset>/<speaker>/`.
- External **Montreal Forced Aligner (MFA)** is then executed on `raw_data/<Dataset>` with a pronunciation lexicon to produce Praat `.TextGrid` files in `preprocessed_data/<Dataset>/TextGrid/<speaker>/<basename>.TextGrid`.

### Stage 2: `preprocess.py` (`preprocessor/preprocessor.py`)
1. **Audio Loading & Sample-Rate Handling**:
   - In `Preprocessor.process_utterance()` (line 172): `wav, _ = librosa.load(wav_path)` loads the resampled `.wav` from `raw_data/` and trims it to `[int(sampling_rate * start) : int(sampling_rate * end)]` using the start/end timestamps of non-silence phones from the MFA `TextGrid`.
   - **Subtle Bug**: `librosa.load(wav_path)` does not pass `sr=self.sampling_rate` (it relies on `librosa`'s default `22050` Hz matching the file written by `prepare_align.py`). If a dataset uses 16 kHz or 24 kHz, `librosa.load(wav_path, sr=self.sampling_rate)` must be explicitly specified.
2. **Mel-Spectrogram Configuration (`config/*/preprocess.yaml` & `audio/stft.py`)**:
   - `sampling_rate`: `22050` Hz
   - `filter_length` (`n_fft`): `1024`
   - `hop_length`: `256` (~11.6 ms frame shift)
   - `win_length`: `1024` (~46.4 ms Hann window)
   - `n_mel_channels`: `80`
   - `mel_fmin`: `0` Hz, `mel_fmax`: `8000` Hz (for HiFi-GAN; `null` for MelGAN)
   - Dynamic range compression: `torch.log(torch.clamp(x, min=1e-5))` (natural log Mel-spectrogram).
   - Saved as transposed array `mel_spectrogram.T` of shape `[T_frames, 80]` in `preprocessed_data/<Dataset>/mel/<speaker>-mel-<basename>.npy`.
3. **Pitch (F0) Extraction**:
   - Uses `pyworld` (`pw.dio` followed by `pw.stonemask` refinement) on `wav.astype(np.float64)` with `frame_period = hop_length / sampling_rate * 1000` (`256 / 22050 * 1000` ms).
   - Trimmed to `pitch[:sum(duration)]`. Utterances with `<= 1` voiced frames are skipped.
   - If `pitch.feature == "phoneme_level"` (default): unvoiced (`0`) frames are linearly interpolated across voiced anchors using `scipy.interpolate.interp1d`, then averaged per phoneme over its duration `d` (`np.mean(pitch[pos : pos + d])`).
   - Normalized across the corpus using `sklearn.preprocessing.StandardScaler` (after IQR outlier removal in `remove_outlier()`) so stored `.npy` files have zero mean and unit variance: `(values - pitch_mean) / pitch_std`.
4. **Energy Extraction**:
   - Computed in `TacotronSTFT.mel_spectrogram()` (`audio/stft.py` line 176) as the L2 norm of the linear STFT magnitude spectrum along the frequency axis: `energy = torch.norm(magnitudes, dim=1)`.
   - If `energy.feature == "phoneme_level"`, frame energies are averaged over each phoneme's duration `d`.
   - Normalized across the corpus via `StandardScaler` (`(values - energy_mean) / energy_std`).
5. **Duration Extraction & Forced Alignment Integration**:
   - Reads the `"phones"` tier from `<basename>.TextGrid` using `tgt.io.read_textgrid(tg_path)`.
   - `get_alignment()` trims leading and trailing silences (`["sil", "sp", "spn"]`), keeping internal pauses (`sp`, `spn`, `sil`), and computes integer frame duration for each phoneme as:
     ```python
     int(np.round(e * self.sampling_rate / self.hop_length) - np.round(s * self.sampling_rate / self.hop_length))
     ```
6. **Metadata Format (`train.txt`, `val.txt`, `stats.json`, `speakers.json`)**:
   - `train.txt` / `val.txt`: Pipe-delimited lines `basename|speaker|{phoneme sequence}|raw_text`
     - Example: `LJ042-0094|LJSpeech|{DH IY0 S OW1 V IY0 AH0 T ...}|the soviet authorities denied oswald permission`
   - `stats.json`: `{"pitch": [min, max, mean, std], "energy": [min, max, mean, std]}` where `min` and `max` are the post-normalization extrema and `mean`, `std` are the raw corpus statistics used for de-normalizing plots.
   - `speakers.json`: Dictionary mapping speaker name string to integer ID `{"LJSpeech": 0}`.

---

## 5. Training Pipeline Inspection (`train.py`, `dataset.py`, `model/optimizer.py`, `evaluate.py`)

1. **Training Entry Point (`train.py`)**:
   - Invoked via: `python3 train.py -p preprocess.yaml -m model.yaml -t train.yaml [--restore_step N]`
2. **Dataset Loading & Batch Construction (`dataset.py`)**:
   - `Dataset` loads `train.txt` (or `val.txt`) and `speakers.json`.
   - `__getitem__` converts `{phoneme sequence}` via `text_to_sequence()` and loads the 4 `.npy` arrays (`mel`, `pitch`, `energy`, `duration`).
   - **Two-level sorted batching**: `DataLoader` fetches `batch_size * group_size` (`16 * 4 = 64`) samples at a time. `Dataset.collate_fn` sorts those 64 samples by descending text length (`np.argsort(-len_arr)`), splits them into `group_size = 4` sub-batches of size `batch_size = 16`, and calls `reprocess()` on each sub-batch to pad 1D/2D arrays to the sub-batch's maximum length.
   - Consequently, `for batchs in loader:` yields a list of 4 sub-batches, and `for batch in batchs:` iterates over each 12-element sub-batch tuple.
3. **Optimizer & Learning-Rate Schedule (`model/optimizer.py`)**:
   - `ScheduledOptim` wraps `torch.optim.Adam(betas=[0.9, 0.98], eps=1e-9, weight_decay=0.0)`.
   - Base scale: $d_{\text{model}}^{-0.5} = 256^{-0.5} = 0.0625$.
   - Noam warmup (`warm_up_step = 4000`) + multiplicative step annealing (`anneal_steps = [300000, 400000, 500000]`, `anneal_rate = 0.3`):
     $$\text{lr}(\text{step}) = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup}^{-1.5}\right) \cdot \prod_{s \in \text{anneal\_steps},\, \text{step} > s} \text{anneal\_rate}$$
4. **Gradient Accumulation & Clipping**:
   - Divides `total_loss / grad_acc_step`, calls `.backward()`, and every `grad_acc_step` steps clips gradient norm via `nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh=1.0)` before `optimizer.step_and_update_lr()`.
5. **Checkpointing, Validation, Logging & Resume**:
   - Saves checkpoints every `save_step` (`100,000` by default — should be lowered to `5,000` or `10,000` for Colab!) to `output/ckpt/<Dataset>/<step>.pth.tar` containing `{"model": model.module.state_dict(), "optimizer": optimizer._optimizer.state_dict()}`.
   - Runs `evaluate()` (`evaluate.py`) every `val_step` (`1,000` steps) over `val.txt` and logs validation losses, spectrogram figure, and audio to TensorBoard (`output/log/<Dataset>/val`).
   - Resumes seamlessly when `--restore_step N` is passed (`utils/model.py` loads `<N>.pth.tar` into both `model` and `scheduled_optim`, and sets `step = args.restore_step + 1`).

---

## 6. Inference & Extraction of Predicted Duration, Pitch (F0), and Energy (`synthesize.py`, `utils/tools.py`, `utils/model.py`)

### 6.1 End-to-End Inference Flow
1. **Input Text Processing (`synthesize.py`)**:
   - **Single mode (`--mode single --text "..."`)**: Calls `preprocess_english()` (looks up words in `lexicon/librispeech-lexicon.txt` or runs `g2p_en.G2p()`, wraps phones in `{...}`, and calls `text_to_sequence`) or `preprocess_mandarin()`. Creates a 6-element batch tuple `(ids, raw_texts, speakers, texts, text_lens, max_src_len)`.
   - **Batch mode (`--mode batch --source val.txt`)**: Uses `TextDataset` to read pre-phonemized `{...}` strings from a metadata file.
2. **Model Forward Pass (`synthesize.py` lines 95–100)**:
   ```python
   output = model(
       *(batch[2:]),
       p_control=pitch_control,
       e_control=energy_control,
       d_control=duration_control
   )
   ```
3. **Vocoder Conversion & Audio Saving (`utils/tools.py` `synth_samples` & `utils/model.py` `vocoder_infer`)**:
   - Uses post-PostNet Mel prediction `predictions[1]` (transposed to `[B, 80, L_mel]`).
   - Passes `mel_predictions` to `HiFi-GAN` (`vocoder(mels).squeeze(1)`) or `MelGAN` (`vocoder.inverse(mels / np.log(10))`).
   - Scales waveform by `max_wav_value` (`32768.0`), casts to `int16`, trims each utterance $i$ to `lengths[i] = predictions[9][i] * hop_length`, and saves `<result_path>/<basename>.wav` via `scipy.io.wavfile.write` and spectrogram plot `<result_path>/<basename>.png` via `plot_mel()`.

### 6.2 Exact Tensor Locations for Extracting Predicted Duration, Pitch (F0), and Energy

During inference, `output = model(*(batch[2:]), ...)` returns a **10-element tuple** (`model/fastspeech2.py` lines 99–110):

| Tuple Index | Variable Name in `FastSpeech2.forward` | Shape | Description |
|---|---|---|---|
| `output[0]` | `output` | `[B, L_mel, 80]` | Pre-PostNet Mel-spectrogram prediction |
| `output[1]` | `postnet_output` | `[B, L_mel, 80]` | Post-PostNet Mel-spectrogram prediction (used for vocoder) |
| **`output[2]`** | **`p_predictions`** | `[B, L_src]` (if `phoneme_level`) or `[B, L_mel]` (if `frame_level`) | **Predicted Pitch (normalized Z-score F0 values)** |
| **`output[3]`** | **`e_predictions`** | `[B, L_src]` (if `phoneme_level`) or `[B, L_mel]` (if `frame_level`) | **Predicted Energy (normalized Z-score L2 magnitude norm)** |
| `output[4]` | `log_d_predictions` | `[B, L_src]` | Raw log-domain duration predictions $\log(\hat{d} + 1)$ |
| **`output[5]`** | **`d_rounded`** | `[B, L_src]` | **Predicted Integer Phoneme Durations (in Mel frames)** after `clamp(round((exp(log_d) - 1) * d_control), min=0)` |
| `output[6]` | `src_masks` | `[B, L_src]` | Encoder padding mask |
| `output[7]` | `mel_masks` | `[B, L_mel]` | Decoder Mel frame padding mask |
| `output[8]` | `src_lens` | `[B]` | Unpadded phoneme sequence lengths |
| `output[9]` | `mel_lens` | `[B]` | Unpadded synthesized Mel frame lengths |

#### How `utils/tools.py` (`synth_samples`, lines 164–198) Currently Uses Them (and What We Must Add):
In the original code, `synth_samples()` extracts for sample `i` (where `src_len = predictions[8][i].item()` and `mel_len = predictions[9][i].item()`):
- **Predicted Duration (per phoneme, in frames)**:
  ```python
  duration = predictions[5][i, :src_len].detach().cpu().numpy()  # 1D array of length src_len
  ```
  *(To convert frames to seconds: `duration_seconds = duration * hop_length / sampling_rate`)*
- **Predicted Pitch (F0)**:
  ```python
  pitch_norm = predictions[2][i, :src_len].detach().cpu().numpy()  # Phoneme-level normalized F0
  pitch_frame_norm = expand(pitch_norm, duration)                  # Expanded to frame-level [mel_len]
  # De-normalize to Hz using stats.json ("pitch": [min, max, mean, std]):
  pitch_hz = pitch_frame_norm * pitch_std + pitch_mean
  ```
- **Predicted Energy**:
  ```python
  energy_norm = predictions[3][i, :src_len].detach().cpu().numpy() # Phoneme-level normalized energy
  energy_frame_norm = expand(energy_norm, duration)                # Expanded to frame-level [mel_len]
  # De-normalize using stats.json ("energy": [min, max, mean, std]):
  energy_raw = energy_frame_norm * energy_std + energy_mean
  ```
- **What is currently saved vs. what is missing**:
  - Currently, `synth_samples()` only overlays `pitch_hz` and `energy_norm` on top of the Mel-spectrogram image (`<basename>.png`) via `plot_mel()` and **discards the arrays** without saving standalone `.npy`/`.json` files or dedicated F0/Energy/Duration plots.
  - For our assignment (`Results/English/` and `Results/Telugu/`), we will extend `synth_samples()` (in our project copy/wrapper, keeping `original_fastspeech2/` untouched) to save `.npy`/`.json` arrays and standalone plots for `duration`, `pitch_hz`, `energy_raw`, and `mel_prediction`.

---

## 7. Existing Capabilities vs. Limitations & Bugs Identified

### Existing Capabilities
- Complete two-stage MFA alignment + acoustic feature preprocessing (`prepare_align.py`, `preprocess.py`).
- Full FastSpeech 2 architecture with phoneme-level or frame-level Variance Adaptor (`model/fastspeech2.py`, `model/modules.py`).
- Multi-speaker embedding table support (`model_config["multi_speaker"]`).
- Built-in HiFi-GAN (`hifigan/`) and MelGAN (`torch.hub`) vocoder inference.
- Controllable synthesis via `--pitch_control`, `--energy_control`, and `--duration_control`.

### Critical Limitations & Code Bugs Found During Inspection
1. **Hardcoded `.cuda()` in `audio/stft.py` (Lines 68–72 & 111)**:
   - `STFT.transform()` explicitly calls `input_data.cuda()` and `self.forward_basis.cuda()`.
   - **Impact**: `preprocess.py` (which computes Mel-spectrograms and energy via `TacotronSTFT`) **crashes immediately on CPU or Apple Silicon MPS** unless `.cuda()` is replaced with device-agnostic tensor operations.
2. **Energy Control Bug in `model/modules.py` (Line 124)**:
   - Inside `VarianceAdaptor.forward()`:
     ```python
     if self.energy_feature_level == "phoneme_level":
         energy_prediction, energy_embedding = self.get_energy_embedding(
             x, energy_target, src_mask, p_control
         )
     ```
   - `p_control` is mistakenly passed instead of `e_control` when `energy_feature_level == "phoneme_level"` (whereas line 146 correctly passes `e_control` for `frame_level`).
3. **Hardcoded `torch.load(...)` Without `map_location` (`utils/model.py` Lines 20, 63, 65)**:
   - `ckpt = torch.load(ckpt_path)` and `torch.load("hifigan/generator_universal.pth.tar")` do not pass `map_location=device`, which fails when loading GPU-trained checkpoints on a CPU/MPS MacBook.
4. **Unconditional `nn.DataParallel(model)` in `train.py` (Line 42)**:
   - Wraps `model = nn.DataParallel(model)` even on single-GPU, CPU, or MPS setups, and saves `model.module.state_dict()`.
5. **`librosa` API Deprecations (`audio/stft.py` & `preprocessor/preprocessor.py`)**:
   - `audio/stft.py` line 42: `pad_center(fft_window, filter_length)` uses positional `size`, which was changed to keyword `size=` in `librosa >= 0.9.0`.
   - `audio/stft.py` line 145: `librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)` passes positional arguments, which raises a `TypeError` in `librosa >= 0.10.0` (requires keyword arguments `sr=..., n_fft=..., n_mels=..., fmin=..., fmax=...`).
   - `preprocessor/*.py`: `librosa.load(wav_path, sampling_rate)` passes `sampling_rate` as a positional argument (`sr=` keyword required in `librosa >= 0.10.0`), and `preprocessor/preprocessor.py` line 172 calls `librosa.load(wav_path)` without `sr=self.sampling_rate`, silently resampling any non-22.05kHz audio to 22,050 Hz.
6. **HiFi-GAN Checkpoints Are Zipped**:
   - `hifigan/generator_LJSpeech.pth.tar.zip` and `hifigan/generator_universal.pth.tar.zip` must be unzipped before running `synthesize.py` or `train.py`.

