# Implementation of FastSpeech 2-Based Text-to-Speech Synthesis
## A Comparative Multi-Lingual Study on English (Svarah) and Telugu (IndicTTS)

**Author / Candidate**: Swarup Pal (`swaruppal1511@gmail.com`)  
**Academic Affiliation**: International Institute of Information Technology, Hyderabad (IIIT Hyderabad)  
**Course / Specification**: Speech Signal Processing / Advanced Speech Synthesis (`FastSpeech2_Assignment.pdf`)  
**Submission Deadline**: 07-11-2026  
**Repository**: `FastSpeech2_Project`  
**Primary Deliverables**: Two Independently Trained FastSpeech 2 Acoustic Models (English & Telugu), 10 Synthesized Test Audio Samples (`22,050 Hz` 16-bit PCM), 30 Variance Adaptor Plots, 4 Training/Validation Loss Curves, Reproducible Codebase (`252/252` Regression Tests Passing), and Comprehensive Technical Report  

---

## 1. Verified Artifact & Checkpoint Signatures

| Language & Corpus | Checkpoint File | Global Step | Cumulative Epoch | Step Val Loss | Best Val Loss | Size (MiB) | SHA-256 Checksum (Prefix) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **English (`Svarah`)** | `checkpoint_best.pth.tar` | **19,000** | **95** *(Session Ep 27)* | **2.348900** | **2.348900** | 399.08 | `794b5123a25f7052...8cf54245e` |
| **English (`Svarah`)** | `checkpoint_latest.pth.tar` | 20,000 | 100 *(Session Ep 32)* | 2.368902 | 2.348900 | 399.08 | `6063dd284e32d7ec...fe0c31241` |
| **Telugu (`IndicTTS`)** | `checkpoint_best.pth.tar` | **13,000** | **91** | **2.086877** | **2.086877** | 397.99 | `0a6c443a7baea134...4a4cb55fd` |
| **Telugu (`IndicTTS`)** | `checkpoint_latest.pth.tar` | 20,000 | 140 | 2.118075 *(Step: 2.094272)* | 2.086877 | 397.99 | `8e46b77bedb23c80...6005f4556f` |
| **Vocoder (`HiFi-GAN`)** | `generator_universal.pth.tar` | Pretrained | Universal v1 | — | — | 53.20 | `771eaf48adb7bcbe...b3ec6aa2e6` |

---

## Table of Contents

1. [Verified Artifact & Checkpoint Signatures](#1-verified-artifact--checkpoint-signatures)
2. [Abstract](#2-abstract)
3. [Introduction](#3-introduction)
4. [Objectives and Problem Statement](#4-objectives-and-problem-statement)
5. [FastSpeech 2 Architecture](#5-fastspeech-2-architecture)
6. [Reference Implementation and Modifications](#6-reference-implementation-and-modifications)
7. [System Architecture & End-to-End Workflow](#7-system-architecture--end-to-end-workflow)
8. [Dataset Selection, Subsampling Strategy & Split Verification](#8-dataset-selection-subsampling-strategy--split-verification)
9. [Dataset Preparation and Preprocessing Pipeline](#9-dataset-preparation-and-preprocessing-pipeline)
10. [Model Configuration & Architectural Hyperparameters](#10-model-configuration--architectural-hyperparameters)
11. [Training Setup & Optimization Regime](#11-training-setup--optimization-regime)
12. [Training and Validation Loss Analysis](#12-training-and-validation-loss-analysis)
13. [Inference Methodology & Prosody Inverse-Normalization](#13-inference-methodology--prosody-inverse-normalization)
14. [Generated English Samples Analysis (E1, E2a, E2b, E3, E4)](#14-generated-english-samples-analysis-e1-e2a-e2b-e3-e4)
15. [Generated Telugu Samples Analysis (T1, T2a, T2b, T3, T4)](#15-generated-telugu-samples-analysis-t1-t2a-t2b-t3-t4)
16. [Variance Adaptor Plot Analysis (All 30 Contours)](#16-variance-adaptor-plot-analysis-all-30-contours)
17. [Comparative Analysis: English vs. Telugu](#17-comparative-analysis-english-vs-telugu)
18. [Limitations and Acoustic Failure Analysis](#18-limitations-and-acoustic-failure-analysis)
19. [Possible Improvements & Future Directions](#19-possible-improvements--future-directions)
20. [Reproducibility, Dependencies & Execution Guide](#20-reproducibility-dependencies--execution-guide)
21. [Assignment Compliance Matrix](#21-assignment-compliance-matrix)
22. [Conclusion](#22-conclusion)
23. [References](#23-references)

---

## 2. Abstract

Neural text-to-speech (TTS) synthesis has transitioned from sequential autoregressive architectures to parallel, non-autoregressive acoustic models capable of deterministic alignment, fast inference, and explicit prosody control. This technical report presents the complete design, implementation, independent training, and quantitative evaluation of **FastSpeech 2** across two typologically distinct languages: **English (`Svarah` Indian English corpus)** and **Telugu (`IndicTTS` Dravidian speech corpus)**. In strict adherence to the assignment specification, two separate acoustic models were trained from scratch for 20,000 optimizer steps without shared weights or cross-language data leakage.

The preprocessing pipeline integrates **Montreal Forced Aligner (MFA)** token-duration extraction, PyWorld (`DIO` + `StoneMask`) continuous fundamental frequency ($F_0$) estimation with linear interpolation over unvoiced segments, and Short-Time Fourier Transform (STFT) frame-level $L_2$-norm energy computation. For English, the `profile_stratified_v1` subsampling strategy curated **3,208 training utterances ($4.999\text{ h}$)** and **664 validation utterances ($1.000\text{ h}$)** across 88 speaker profiles using CMUdict ARPAbet phonemization with `g2p_en` fallback. For Telugu, the `option_1b` stratified two-speaker split curated **2,300 training utterances ($5.000\text{ h}$: 1,053 female `spk_000` and 1,247 male `spk_001`)** and **421 validation utterances ($1.000\text{ h}$: 192 female `spk_000` and 229 male `spk_001`)**, paired with a custom 101-token native alpha-syllabic grapheme tokenizer incorporating Unicode NFC normalization, Latin-$o$ to anusvara (`ం`) repair, and word/clause boundary pause mapping (`@sp`, token ID 4).

Validation evaluations conducted every 1,000 steps identified **Step 19,000** (validation loss: $2.34890$) as the best observed validation checkpoint for English and **Step 13,000** (validation loss: $2.08688$) as the best observed validation checkpoint for Telugu, preventing post-13k pitch loss drift ($0.7481 \to 0.7688$). All ten canonical assignment test sentences (`E1–E4` and `T1–T4`) were synthesized at $22,050\text{ Hz}$ 16-bit mono PCM via a universal HiFi-GAN vocoder, accompanied by all 30 variance adaptor plots (duration, pitch, and energy). Controlled homograph comparisons on English *"conduct"* (`E2(a)` vs. `E2(b)`) and Telugu *"పట్టు"* (`T2(a)` vs. `T2(b)`) demonstrate that while Transformer self-attention induces phrasal prosodic variation, these experiments do not demonstrate semantic disambiguation of contextual meanings.

---

## 3. Introduction

Text-to-Speech (TTS) systems synthesize human-like speech waveforms from raw orthographic text. Early neural TTS systems—most notably Tacotron, Tacotron 2, Deep Voice 3, and Transformer TTS—formulated acoustic generation as an autoregressive sequence-to-sequence problem, predicting each mel-spectrogram frame $\mathbf{y}_t$ conditioned on all preceding frames $\mathbf{y}_{<t}$. Despite high segmental fidelity, autoregressive TTS architectures exhibit four structural bottlenecks:

1. **Sequential Inference Latency**: Generating $T$ mel frames requires $O(T)$ serial forward passes through the autoregressive decoder, preventing GPU parallelism on long utterances.
2. **Exposure Bias & Error Accumulation**: Teacher forcing during training supplies ground-truth frames $\mathbf{y}_{t-1}$, whereas inference feeds back predicted frames $\hat{\mathbf{y}}_{t-1}$, causing small acoustic errors to compound over time.
3. **Attention Alignment Collapse**: Soft content-based or location-sensitive attention frequently fails on out-of-distribution lengths, polysyllabic compounds (e.g., *"antidisestablishmentarianism"*), or repeated syllables, resulting in word skipping, stuttering repetitions, or premature termination.
4. **Under-Determined One-to-Many Mapping**: Identical text sequences correspond to infinitely many valid acoustic realizations differing in speaking rate, pitch contour, vocal effort, and speaker timbre. Without explicit prosodic conditioning, an $L_1/L_2$ regression loss averages over modes, producing over-smoothed mel-spectrograms and flat intonation.

**FastSpeech** resolved sequential latency and attention collapse by replacing soft attention with an explicit **Duration Predictor** and **Length Regulator** inside a non-autoregressive Feed-Forward Transformer (FFT) stack, but still relied on knowledge distillation from an autoregressive teacher model. **FastSpeech 2** (Ren et al., 2020) eliminated teacher-student distillation entirely by training directly on ground-truth mel-spectrograms and introducing an explicit **Variance Adaptor** that conditions the decoder on phoneme duration, continuous fundamental frequency ($F_0$), and frame-level spectral energy.

In this study, we implement, adapt, and contrast FastSpeech 2 across two structurally distinct languages:
- **English (Indo-European, Stress-Timed)**: Characterized by deep, non-transparent orthography requiring lexicon + neural G2P conversion into stressed ARPAbet phonemes, variable consonant clusters, and lexical stress contrasts.
- **Telugu (Dravidian, Syllable/Mora-Timed)**: Characterized by a phonetically transparent alpha-syllabic *akshara* writing system (vowels, consonants, *gunintalu* vowel diacritics, *vattulu* conjuncts via *virama* `్`, and *anusvara* `ం`), rich agglutinative morphology, and phonemic consonant gemination (e.g., `ట్ట` in `పట్టు`, `శ్శ` in `నిశ్శబ్దంగా`).

---

## 4. Objectives and Problem Statement

In accordance with `FastSpeech2_Assignment.pdf`, the objectives of this work are:
1. **Architecture Mastery & Dual-Language Adaptation**: Understand the FastSpeech 2 paper and reference codebase (`ming024/FastSpeech2`), refactor shared components into a clean modular library (`common/`), and implement independent language pipelines for English (`English/`) and Telugu (`Telugu/`).
2. **Balanced ~5h Training / ~1h Validation Curation**: Construct strictly disjoint, non-overlapping training (~5 hours) and validation (~1 hour) splits for English (`Svarah`) and Telugu (`IndicTTS`) with documented subsampling strategies.
3. **Forced Alignment & Acoustic Feature Extraction**: Extract token-level durations via Montreal Forced Aligner (MFA) and frame-level mel-spectrograms, pitch ($F_0$), and energy arrays with corpus-level normalization statistics (`stats.json`).
4. **Independent Model Training & Convergence Analysis**: Train separate English and Telugu FastSpeech 2 models for 20,000 steps, analyze training and validation loss trajectories, and select optimal checkpoints based on validation evidence.
5. **Canonical Test Synthesis & Variance Adaptor Inspection**: Synthesize all ten exact assignment sentences (`E1, E2a, E2b, E3, E4` and `T1, T2a, T2b, T3, T4`), generate all 30 duration, pitch, and energy plots, and perform sample-by-sample quantitative and perceptual evaluations.

---

## 5. FastSpeech 2 Architecture

FastSpeech 2 maps a discrete token sequence $\mathbf{x} = (x_1, x_2, \dots, x_{L_{\text{src}}})$ of length $L_{\text{src}}$ to a continuous 80-channel log-mel spectrogram $\hat{\mathbf{Y}}_{\text{post}} \in \mathbb{R}^{L_{\text{mel}} \times 80}$ of frame length $L_{\text{mel}} = \sum_{i=1}^{L_{\text{src}}} d_i$. Figure 1 illustrates the exact architecture implemented in `common/model/fastspeech2.py`, `common/model/variance_adaptor.py`, and `common/model/postnet.py`.

![FastSpeech 2 and Universal HiFi-GAN Architecture](figures/fastspeech2_architecture.svg)

### 5.1 Token Embedding, Positional Encoding & Multi-Speaker Conditioning
Given token indices $\mathbf{x} \in \{0, \dots, V-1\}^{B \times L_{\text{src}}}$ ($V = 360$ for English; $V = 101$ for Telugu), the encoder input combines a learned token embedding table $\mathbf{E}_{\text{tok}} \in \mathbb{R}^{V \times d}$ ($d = 256$) with sinusoidal positional encodings $\mathbf{P} \in \mathbb{R}^{L_{\text{src}} \times d}$:
$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
After passing through the $N_e = 4$ Feed-Forward Transformer (FFT) encoder blocks, multi-speaker conditioning is applied prior to the Variance Adaptor (`common/model/fastspeech2.py`, lines 183–187) by broadcasting and adding the speaker embedding $\mathbf{e}_s = \mathbf{E}_{\text{spk}}[s] \in \mathbb{R}^{d}$ across all token positions:
$$\mathbf{H}_{\text{spk}} = \mathbf{H}_{\text{enc}} + \mathbf{e}_s \cdot \mathbf{1}_{L_{\text{src}}}^T$$

### 5.2 Feed-Forward Transformer (FFT) Encoder and Decoder Blocks
Our implementation uses $N_e = 4$ FFT blocks in the **Text Encoder** and $N_d = 6$ FFT blocks in the **Mel Decoder** (`config/Svarah/model.yaml`, `config/IndicTTS/model.yaml`). Each FFT block consists of:
1. **Multi-Head Scaled Dot-Product Self-Attention** ($h = 2$ heads, $d_k = d_v = d / h = 128$):
   $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right)\mathbf{V}$$
   $$\mathbf{H}' = \text{LayerNorm}\left(\mathbf{H} + \text{Dropout}(\text{MultiHead}(\mathbf{H}))\right)$$
2. **Position-Wise 2-Layer 1D Convolutional Feed-Forward Network** with asymmetric kernel sizes $k_1 = 9, k_2 = 1$ and inner filter dimension $d_{\text{filter}} = 1024$:
   $$\text{ConvFFN}(\mathbf{H}') = \text{Conv1D}_{k=1}\left(\text{ReLU}\left(\text{Conv1D}_{k=9}(\mathbf{H}')\right)\right)$$
   $$\mathbf{H}_{\text{out}} = \text{LayerNorm}\left(\mathbf{H}' + \text{Dropout}(\text{ConvFFN}(\mathbf{H}'))\right)$$
   Using a $k_1 = 9$ 1D convolution instead of a dense $k = 1$ MLP captures local co-articulatory context across adjacent phonemes/graphemes and adjacent mel frames.

### 5.3 Explicit Variance Adaptor (`common/model/variance_adaptor.py`)
The Variance Adaptor predicts and injects three prosodic factors—duration, pitch, and energy—using identical 2-layer 1D Convolutional Variance Predictor architectures (`filter_size = 256`, `kernel_size = 3`, ReLU, LayerNorm, `dropout = 0.5`, followed by a scalar linear projection $\mathbb{R}^{256} \to \mathbb{R}^1$):

1. **Duration Predictor & Length Regulator**:
   - The Duration Predictor operates at token resolution $[B, L_{\text{src}}, 256]$ to predict log-domain token durations $\hat{d}_{\log, i} = \ln(\hat{d}_i + 1)$.
   - During **training**, the Length Regulator expands token states $\mathbf{h}_i$ using ground-truth MFA integer durations $d_i$. During **inference**, predicted log-durations are inverted, scaled by the user duration control multiplier $\alpha_d$, rounded to the nearest non-negative integer, and clamped at $\ge 0$:
     $$\hat{d}_i = \text{clamp}\left(\text{round}\left(\left(\exp(\hat{d}_{\log, i}) - 1\right) \cdot \alpha_d\right),\, \min=0\right)$$
   - The Length Regulator repeats each token representation $\mathbf{h}_i$ exactly $d_i$ times to yield frame-level hidden states $\mathbf{H}_{\text{exp}} \in \mathbb{R}^{B \times L_{\text{mel}} \times 256}$.

2. **Frame-Level Pitch Predictor & Quantization**:
   - Configured with `pitch.feature = "frame_level"` (`preprocess.yaml`), the Pitch Predictor operates on the length-regulated frame states $\mathbf{H}_{\text{exp}}$ to predict normalized continuous pitch $z$-scores $\hat{z}_{p, t}$.
   - During inference, predictions are scaled by $\alpha_p$: $\tilde{z}_{p, t} = \hat{z}_{p, t} \cdot \alpha_p$.
   - Continuous normalized pitch is quantized via `torch.bucketize` over $255$ linear bin boundaries spanning $[p_{\min}, p_{\max}]$ from `stats.json` into a 256-entry embedding table $\mathbf{E}_{\text{pitch}} \in \mathbb{R}^{256 \times 256}$.

3. **Frame-Level Energy Predictor & Quantization**:
   - Configured with `energy.feature = "frame_level"`, the Energy Predictor operates on $\mathbf{H}_{\text{exp}}$ to predict normalized frame-level STFT $L_2$-norm energy $\hat{z}_{e, t}$, scaled by $\alpha_e$ at inference and quantized across $255$ linear boundaries $[e_{\min}, e_{\max}]$ into $\mathbf{E}_{\text{energy}} \in \mathbb{R}^{256 \times 256}$.
   - The adapted frame representation entering the 6-layer FFT Decoder is:
     $$\mathbf{H}_{\text{adapt}} = \mathbf{H}_{\text{exp}} + \mathbf{E}_{\text{pitch}}\left[\text{bucketize}(z_p)\right] + \mathbf{E}_{\text{energy}}\left[\text{bucketize}(z_e)\right]$$

### 5.4 Mel Linear Projection, Mask-Aware PostNet & Total Loss Formulation
The 6-layer FFT Decoder output $\mathbf{H}_{\text{dec}} \in \mathbb{R}^{B \times L_{\text{mel}} \times 256}$ is projected via `mel_linear` ($\mathbb{R}^{256 \to 80}$) to produce the pre-PostNet spectrogram $\hat{\mathbf{Y}}_{\text{mel}}$. A 5-layer 1D Convolutional **PostNet** (`512` channels, kernel size `5`, BatchNorm1D, Tanh activation on layers 1–4, dropout `0.5`) predicts a residual correction:
$$\hat{\mathbf{Y}}_{\text{post}} = \hat{\mathbf{Y}}_{\text{mel}} + \text{PostNet}(\hat{\mathbf{Y}}_{\text{mel}},\, \mathbf{M}_{\text{mel}})$$

The total training objective (`common/model/loss.py`) is the unweighted sum ($\lambda_d = \lambda_p = \lambda_e = 1.0$) of five strictly mask-filtered losses:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{post}} + \mathcal{L}_{\text{dur}} + \mathcal{L}_{\text{pitch}} + \mathcal{L}_{\text{energy}}$$
where:
$$\mathcal{L}_{\text{mel}} = \frac{1}{|\Omega_{\text{mel}}|} \sum_{(b,t) \in \Omega_{\text{mel}}} \left\|\mathbf{Y}_{b,t} - \hat{\mathbf{Y}}_{\text{mel}, b,t}\right\|_1, \qquad \mathcal{L}_{\text{post}} = \frac{1}{|\Omega_{\text{mel}}|} \sum_{(b,t) \in \Omega_{\text{mel}}} \left\|\mathbf{Y}_{b,t} - \hat{\mathbf{Y}}_{\text{post}, b,t}\right\|_1$$
$$\mathcal{L}_{\text{dur}} = \frac{1}{|\Omega_{\text{src}}|} \sum_{(b,i) \in \Omega_{\text{src}}} \left(\ln(d_{b,i} + 1) - \hat{d}_{\log, b,i}\right)^2$$
$$\mathcal{L}_{\text{pitch}} = \frac{1}{|\Omega_{\text{mel}}|} \sum_{(b,t) \in \Omega_{\text{mel}}} \left(z_{p, b,t} - \hat{z}_{p, b,t}\right)^2, \qquad \mathcal{L}_{\text{energy}} = \frac{1}{|\Omega_{\text{mel}}|} \sum_{(b,t) \in \Omega_{\text{mel}}} \left(z_{e, b,t} - \hat{z}_{e, b,t}\right)^2$$
Here $\Omega_{\text{src}}$ and $\Omega_{\text{mel}}$ denote the sets of valid (non-padded) token and mel-frame indices (`~src_masks` and `~mel_masks`).

---

## 6. Reference Implementation and Modifications

We used the official PyTorch FastSpeech 2 implementation (`ming024/FastSpeech2`, preserved under `original_fastspeech2/`) as our reference baseline and engineered comprehensive architectural, linguistic, and infrastructure modifications across `common/`, `English/`, and `Telugu/`.

| Component / Scope | Reference Baseline (`original_fastspeech2/`) | Our Modified Implementation (`common/`, `English/`, `Telugu/`) | Engineering Justification |
| :--- | :--- | :--- | :--- |
| **Device Management (`DEC-01`)** | Hardcoded global `device = torch.device("cuda")` at import time | Dynamic device inheritance from input tensors (`cuda`, `mps`, `cpu`) via `common/utils/device.py` | Enables training on NVIDIA CUDA GPUs and local inference/testing on Apple Silicon M2 (`mps`/`cpu`). |
| **Mask-Aware PostNet (`DEF-01`)** | `PostNet` ran `BatchNorm1d` across padded frames without masking | `common/model/postnet.py` zeros padded positions before and after every Conv1D+BN layer using `mel_masks` | Prevents zero-padded frames in variable-length batches from corrupting BatchNorm running statistics. |
| **Energy Control Routing (`DEC-02`)** | Bug in reference `VarianceAdaptor` multiplied energy by `p_control` in one branch | `common/model/variance_adaptor.py` strictly routes `p_control` ($\alpha_p$), `e_control` ($\alpha_e$), and `d_control` ($\alpha_d$) | Guarantees independent prosodic control during synthesis. |
| **Dynamic Decoder PosEnc (`DEC-09`)** | Crashed with `IndexError` when inference $L_{\text{mel}} > \text{max\_seq\_len}$ (1000) | `common/transformer/models.py` dynamically extends sinusoidal positional encodings when $L_{\text{mel}} > 1000$ | Supports arbitrarily long inference utterances without truncation. |
| **Centralized Path Manager** | Relied on `os.getcwd()` and relative `./preprocessed_data` paths | `common/paths.py` discovers `PROJECT_ROOT` via marker verification and resolves all config/asset paths portably | Allows running training, inference, and tests from any working directory or Google Colab mount. |
| **English Frontend (`English/`)** | Basic LJSpeech single-speaker cleaning | `English/text/normalizer.py` expands times (`10:45 AM`), dates (`12 November 2026`), numbers, plus CMUdict + `g2p_en` fallback and 88-profile conditioning | Handles complex Indian English multi-speaker utterances (`E1–E4`) with zero OOV failures. |
| **Telugu Frontend (`Telugu/`)** | **Not supported** (English/Mandarin only in reference repo) | Custom 101-token Telugu grapheme inventory (`Telugu/text/symbols.py`), Unicode NFC normalizer, Latin-$o \to$ `ం` repair, and `@sp` (ID 4) pause mapping | Eliminates external pronunciation lexicon errors by leveraging Telugu's phonetic *akshara* orthography. |

### 6.1 Deep-Dive: Telugu Linguistic & Grapheme Engineering (`Telugu/text/`)
Adapting FastSpeech 2 to Telugu required solving four language-specific challenges documented across `Telugu/text/normalizer.py`, `Telugu/text/symbols.py`, and `Telugu/inference.py`:
1. **Direct Alpha-Syllabic Grapheme Tokenization (101-Token Vocabulary)**:
   Because Telugu orthography is phonemic, we designed a deterministic 101-symbol vocabulary (`VOCAB_SIZE = 101`):
   - **Control & Pause Symbols (IDs 0–6)**: `<pad>` (0), `<unk>` (1), `<s>` (2), `</s>` (3), `@sp` (4), `<eps>` (5), `_` (6).
   - **Independent Vowels (*Achchulu*, IDs 19–34)**: `అ, ఆ, ఇ, ఈ, ఉ, ఊ, ఋ, ౠ, ఌ, ౡ, ఎ, ఏ, ఐ, ఒ, ఓ, ఔ`.
   - **Consonants (*Hallulu*, IDs 35–70)**: `క` through `ఱ` (including retroflexes `ట, ఠ, డ, ఢ, ణ, ళ` and aspirated stops).
   - **Dependent Vowel Matras (*Gunintalu*) & Virama (*Halant*, IDs 71–87)**: `ా, ి, ీ, ు, ూ, ృ, ౄ, ె, ే, ై, ొ, ో, ౌ, ్` (`U+0C4D`, which binds consonant clusters/geminates such as `ట్ట = ట + ్ + ట`).
   - **Syllabic Modifiers (IDs 88–90)**: Anusvara `ం` (`U+0C02`), Visarga `ః` (`U+0C03`), Candrabindu `ఁ` (`U+0C01`).
   - **Telugu Numerals (IDs 91–100)**: `౦, ౧, ౨, ౩, ౪, ౫, ౬, ౭, ౮, ౯`.
2. **Unicode NFC Canonicalization & Historical Latin-$o$ Repair**:
   Inspection of raw IndicTTS Telugu transcripts revealed 48 instances where OCR/transliteration artifacts encoded the Telugu anusvara `ం` (`U+0C02`) as a Latin lowercase `o` (`U+006F`) adjacent to Telugu characters. `Telugu/text/normalizer.py` applies contextual regex repair (`o` $\to$ `ం` within Telugu script runs) followed by `unicodedata.normalize("NFC", text)`.
3. **Elimination of Untrained Space/Punctuation IDs (IDs 7–18) via `@sp` Pause Mapping**:
   During MFA alignment (`option_1b`), inter-word boundaries and clausal pauses were aligned to `@sp` (token ID 4), meaning ordinary ASCII space/punctuation IDs (7–18) never received gradient updates during training. `Telugu/inference.py` maps all word spaces and clause-internal punctuation (`,`, `;`, `:`, `-`) to `@sp` (ID 4), collapses consecutive `@sp` tokens, and strips utterance-final punctuation so the model never encounters untrained token embeddings.

---

## 7. System Architecture & End-to-End Workflow

Figure 2 depicts the complete three-stage operational workflow connecting offline corpus preparation, dual-model training, and inference evaluation.

![FastSpeech 2 End-to-End Dual-Language Workflow](figures/system_workflow.svg)

---

## 8. Dataset Selection, Subsampling Strategy & Split Verification

The assignment specifies using **approximately 5 hours of audio for training and 1 hour of audio for validation** per language from **Svarah** (English) and **IndicTTS** (Telugu), with zero overlap between partitions.

### 8.1 English Dataset (`Svarah`) — `profile_stratified_v1` Strategy
- **Source Corpus**: AI4Bharat Svarah Indian English speech corpus (`9,656` total utterances, `~19.08` hours).
- **Subsampling & Stratification Strategy (`English/metadata/splits/profile_stratified_v1/`)**:
  To hit the exact 5-hour training and 1-hour validation target while preserving speaker diversity across Indian English accents, we grouped utterances by verified speaker voice profiles (`88` active profiles) and performed stratified sampling without replacement:
  - **Training Set**: **3,208 utterances** totaling **17,997.41 seconds (`4.9993 hours`)** across all 88 profiles.
  - **Validation Set**: **664 utterances** totaling **3,599.58 seconds (`0.9999 hours`)** stratified across the same speaker profiles.
  - **Leakage Verification (`split_audit.json`)**: `0` overlapping audio filenames, `0` overlapping utterance IDs, and `0` duplicate recordings between `train.txt` and `val.txt`.

### 8.2 Telugu Dataset (`IndicTTS`) — `option_1b` Stratified Two-Speaker Strategy
- **Source Corpus**: IIT Madras IndicTTS Telugu corpus (`data/telugu_parquet/`, native female speaker `spk_000` and native male speaker `spk_001`).
- **Subsampling & Stratification Strategy (`Telugu/metadata/splits/option_1b/`)**:
  We filtered utterances to the clean $1.5\text{ s} - 15.0\text{ s}$ duration window with 100% MFA alignment coverage and performed speaker-stratified allocation to achieve exact 5.0-hour training and 1.0-hour validation budgets:
  - **Training Set**: **2,300 utterances** totaling **18,000.52 seconds (`5.0001 hours`)** — **1,053 female (`spk_000`, $45.78\%$)** and **1,247 male (`spk_001`, $54.22\%$)**.
  - **Validation Set**: **421 utterances** totaling **3,600.18 seconds (`1.0001 hours`)** — **192 female (`spk_000`, $45.61\%$)** and **229 male (`spk_001`, $54.39\%$)**.
  - **Reserved Held-Out Test Pool**: **306 utterances** (`0.72 hours`), plus the 5 canonical assignment test sentences (`T1–T4`).
  - **Leakage Verification**: `0` overlapping utterance IDs or transcripts between training and validation manifests.

| Dataset Metric | English (`Svarah` — `profile_stratified_v1`) | Telugu (`IndicTTS` — `option_1b`) |
| :--- | :--- | :--- |
| **Language & Script** | Indian English (Latin / ARPAbet Phonemes) | Telugu (Brahmic Alpha-Syllabic *Akshara* Graphemes) |
| **Active Speakers** | 88 speaker profiles (`speakers.json`: 115 slots) | 2 native speakers (`spk_000` Female, `spk_001` Male) |
| **Training Utterances & Duration** | **3,208 utterances** (`17,997.4 s` / **4.999 h**) | **2,300 utterances** (`18,000.5 s` / **5.000 h**) |
| **Validation Utterances & Duration** | **664 utterances** (`3,599.6 s` / **1.000 h**) | **421 utterances** (`3,600.2 s` / **1.000 h**) |
| **Speaker Split Breakdown** | Stratified across 88 profiles | Train: 1,053 F / 1,247 M; Val: 192 F / 229 M |
| **Train / Val Overlap** | **0 utterances** (Verified disjoint) | **0 utterances** (Verified disjoint) |
| **Audio Specification** | $22,050\text{ Hz}$, 16-bit Mono PCM | $22,050\text{ Hz}$, 16-bit Mono PCM (resampled from 48 kHz) |
| **Vocabulary Size ($V$)** | `360` symbols (CMUdict ARPAbet + stress + punct) | `101` symbols (Vowels, Consonants, Matras, Virama, `@sp`) |

---

## 9. Dataset Preparation and Preprocessing Pipeline

Both datasets undergo an identical four-stage acoustic and phonetic preprocessing pipeline (`English/preprocess.py`, `Telugu/preprocess.py`, `config/*/preprocess.yaml`):

1. **Waveform Standardization**: All audio files are converted to single-channel 16-bit PCM at $f_s = 22,050\text{ Hz}$ (`max_wav_value = 32768.0`) using high-precision polyphase resampling (`soxr`/`librosa`).
2. **80-Channel Log-Mel Spectrogram Extraction**:
   Computed via Short-Time Fourier Transform (STFT) with FFT size $N_{\text{fft}} = 1024$, Hann window length $W = 1024$ ($46.44\text{ ms}$), hop length $H = 256$ ($11.61\text{ ms}$ per frame), and $80$ triangular mel filterbanks spanning $f_{\min} = 0\text{ Hz}$ to $f_{\max} = 8,000\text{ Hz}$.
3. **Montreal Forced Alignment (MFA) Duration Extraction**:
   - **English**: Aligned using MFA with the LibriSpeech ARPAbet pronunciation lexicon (`original_fastspeech2/lexicon/librispeech-lexicon.txt`).
   - **Telugu**: Aligned using MFA `option_1b` acoustic model training over the grapheme-to-grapheme dictionary (`research/Telugu_Dictionary_g2g.txt`), mapping inter-word pauses to `@sp`.
   - Interval boundaries $[t_{\text{start}}, t_{\text{end}}]$ from Praat `.TextGrid` files are converted into integer mel-frame durations $d_i$ such that $\sum_{i=1}^{L_{\text{src}}} d_i = L_{\text{mel}}$ exactly matches the spectrogram length.
4. **Frame-Level Pitch ($F_0$) and Energy Extraction & $z$-Score Normalization**:
   - **Pitch ($F_0$)**: Extracted via PyWorld `dio` + `stonemask` at frame step $256 / 22050\text{ s}$. Unvoiced frames ($F_0 = 0$) are linearly interpolated between nearest voiced boundaries prior to corpus-wide $z$-score normalization: $z_p = (F_0 - \mu_p) / \sigma_p$.
   - **Energy ($E$)**: Computed as the frame-wise Euclidean ($L_2$) norm of linear STFT magnitude bins, $E_t = \|\mathbf{S}_t\|_2$, and normalized via $z_e = (E_t - \mu_e) / \sigma_e$.

### Verified Corpus Normalization Statistics (`preprocessed_data/*/stats.json`)

| Statistic | English (`Svarah/stats.json`) | Telugu (`IndicTTS/stats.json`) | Physical Units & Role |
| :--- | :---: | :---: | :--- |
| **Pitch Mean ($\mu_p$)** | `177.9316 Hz` | `195.6199 Hz` | Corpus $F_0$ mean for $z$-score & inverse normalization |
| **Pitch Std ($\sigma_p$)** | `59.4528 Hz` | `46.4196 Hz` | Corpus $F_0$ standard deviation |
| **Normalized Pitch Range $[p_{\min}, p_{\max}]$** | `[-2.9928, 10.1252]` | `[-2.9401, 11.8867]` | Bounds for 256 linear quantization bins (`pitch_bins`) |
| **Energy Mean ($\mu_e$)** | `27.3145` | `28.1071` | Corpus STFT $L_2$-norm mean |
| **Energy Std ($\sigma_e$)** | `30.3573` | `26.6611` | Corpus STFT $L_2$-norm standard deviation |
| **Normalized Energy Range $[e_{\min}, e_{\max}]$** | `[-0.8998, 7.4643]` | `[-1.0542, 8.6770]` | Bounds for 256 linear quantization bins (`energy_bins`) |

---

## 10. Model Configuration & Architectural Hyperparameters

Table 4 documents the exact model configurations from `config/Svarah/model.yaml` and `config/IndicTTS/model.yaml`. Both models contain approximately $35.1\text{M}$ trainable parameters (differing only in the source vocabulary and speaker embedding table rows).

| Module | Hyperparameter | English (`Svarah`) | Telugu (`IndicTTS`) | Codebase Reference |
| :--- | :--- | :---: | :---: | :--- |
| **Text Encoder** | `encoder_layer` ($N_e$) | `4` | `4` | `common/transformer/models.py` |
| | `encoder_head` ($h$) / `encoder_hidden` ($d$) | `2` heads / `256` dim | `2` heads / `256` dim | `ScaledDotProductAttention` ($d_k = 128$) |
| | `conv_filter_size` / `conv_kernel_size` | `1024` / `[9, 1]` | `1024` / `[9, 1]` | `PositionwiseFeedForward` (Conv1D) |
| | `encoder_dropout` | `0.2` | `0.2` | Residual & attention dropout |
| **Variance Adaptor** | Predictor Conv1D Layers | `2` layers (`256` ch) | `2` layers (`256` ch) | `common/model/variance_predictor.py` |
| | `kernel_size` / `dropout` | `3` / `0.5` | `3` / `0.5` | Duration, Pitch & Energy predictors |
| | `pitch.feature` / `energy.feature` | `frame_level` | `frame_level` | Applied after `LengthRegulator` |
| | `pitch_quantization` / `energy_quantization` | `linear` (`256` bins) | `linear` (`256` bins) | `torch.bucketize` ($255$ boundaries) |
| **Mel Decoder** | `decoder_layer` ($N_d$) | **`6`** | **`6`** | `common/transformer/models.py` |
| | `decoder_head` ($h$) / `decoder_hidden` ($d$) | `2` heads / `256` dim | `2` heads / `256` dim | Matches encoder hidden dimension |
| | `conv_filter_size` / `conv_kernel_size` | `1024` / `[9, 1]` | `1024` / `[9, 1]` | `PositionwiseFeedForward` |
| | `decoder_dropout` | `0.2` | `0.2` | Residual & attention dropout |
| **PostNet** | Layers / Channels / Kernel Size | `5` / `512` / `5` | `5` / `512` / `5` | `common/model/postnet.py` (Mask-aware) |
| **Conditioning** | `n_src_vocab` ($V$) | `360` | `101` | `nn.Embedding(V, 256, padding_idx=0)` |
| | `multi_speaker` / `n_speaker` | `True` / `115` *(88 active)* | `True` / `2` | `nn.Embedding(n_speaker, 256)` |
| **Vocoder** | Architecture | HiFi-GAN Universal v1 | HiFi-GAN Universal v1 | `original_fastspeech2/hifigan/` |

---

## 11. Training Setup & Optimization Regime

Both models were trained independently on NVIDIA T4/L4 GPUs (`English/train.py`, `Telugu/train.py`) under identical optimization hyperparameters (`config/Svarah/train.yaml`, `config/IndicTTS/train.yaml`):

- **Total Training Steps**: $20,000$ optimizer updates:
  - **English (`Svarah`)**: $3,208$ train utterances at `batch_size = 16` ($200$ steps/epoch) $\implies$ **100 complete epochs** ($1\text{ h }23\text{ m }32\text{ s}$).
  - **Telugu (`IndicTTS`)**: $2,300$ train utterances at `batch_size = 16` ($\approx 143$ steps/epoch) $\implies$ **140 complete epochs** ($1\text{ h }36\text{ m }45\text{ s}$).
- **Optimizer**: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$, `weight_decay = 0.0`).
- **Noam Learning Rate Scheduler** (`common/model/optimizer.py`):
  $$\text{lr}(\text{step}) = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup\_steps}^{-1.5}\right), \qquad d_{\text{model}} = 256,\quad \text{warmup\_steps} = 4,000$$
  The learning rate increases linearly during the first $4,000$ steps to a peak of $\text{lr}_{\max} = 256^{-0.5} \times 4000^{-0.5} = 9.882 \times 10^{-4}$, then decays proportionally to $\text{step}^{-0.5}$, reaching $4.419 \times 10^{-4}$ at step $20,000$.
- **Gradient Clipping**: Global $L_2$-norm threshold $\text{grad\_clip\_thresh} = 1.0$.
- **Logging & Validation Cadence**: Training metrics logged every `100` steps (`201` records in `training_history.csv`); full validation sweep and checkpoint evaluation executed every `1,000` steps (`20` validation evaluations in `validation_history.csv`).

---

## 12. Training and Validation Loss Analysis

### 12.1 Quantitative Validation Trajectories Across All Milestones

*Methodological Note: Absolute validation loss values cannot be compared directly across English and Telugu as a direct measure of model quality, because the two models operate over different token granularities (360 ARPAbet symbols vs. 101 Telugu graphemes), speaker cardinalities (88 vs. 2 speakers), and corpus normalization variances.*

| Global Step | En Epoch | En Val Total | En Mel $L_1$ | En Post $L_1$ | En Pitch | En Energy | En Dur | Te Epoch | Te Val Total | Te Mel $L_1$ | Te Post $L_1$ | Te Pitch | Te Energy | Te Dur |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 5 | 2.8021 | 0.8243 | 0.9953 | 0.4524 | 0.2923 | 0.2378 | 7 | 2.5621 | 0.6853 | 0.7827 | 0.7292 | 0.2576 | 0.1072 |
| **3,000** | 15 | 2.4783 | 0.7219 | 0.7248 | 0.4648 | 0.2804 | 0.2864 | 21 | 2.2485 | 0.5562 | 0.5579 | 0.7932 | 0.2428 | 0.0984 |
| **5,000** | 25 | 2.4060 | 0.6934 | 0.6934 | 0.4747 | 0.2759 | 0.2686 | 35 | 2.1738 | 0.5248 | 0.5249 | 0.7940 | 0.2369 | 0.0932 |
| **8,000** | 40 | 2.3842 | 0.6812 | 0.6810 | 0.4669 | 0.2801 | 0.2750 | 56 | 2.1194 | 0.5108 | 0.5107 | 0.7685 | 0.2358 | 0.0936 |
| **10,000** | 50 | 2.3794 | 0.6767 | 0.6766 | 0.4670 | 0.2815 | 0.2775 | 70 | 2.1052 | 0.5072 | 0.5071 | 0.7630 | 0.2352 | 0.0927 |
| **13,000** | 65 | 2.3698 | 0.6741 | 0.6739 | 0.4638 | 0.2819 | 0.2761 | **91** | **2.0869** | **0.5039** | **0.5039** | **0.7481** | **0.2356** | **0.0954** |
| **15,000** | 75 | 2.3744 | 0.6775 | 0.6774 | 0.4656 | 0.2827 | 0.2712 | 105 | 2.1111 | 0.5069 | 0.5068 | 0.7648 | 0.2349 | 0.0978 |
| **19,000** | **95** | **2.3489** | **0.6686** | **0.6684** | **0.4592** | **0.2802** | **0.2725** | 133 | 2.1146 | 0.5065 | 0.5064 | 0.7671 | 0.2381 | 0.0965 |
| **20,000** | 100 | 2.3689 | 0.6746 | 0.6743 | 0.4596 | 0.2850 | 0.2754 | 140 | 2.1181 | 0.5070 | 0.5069 | 0.7688 | 0.2390 | 0.0964 |

### 12.2 Convergence Behaviour, Overfitting Analysis & Checkpoint Reconciliation
1. **Rapid Warmup Convergence (Steps 100–5,000)**:
   During the 4,000-step Noam warmup phase, training total loss dropped steeply from $5.84 \to 1.52$ (English) and $4.91 \to 1.18$ (Telugu). Notably, at Step 1,000, PostNet $L_1$ loss was higher than Pre-Net Mel $L_1$ loss ($0.9953$ vs. $0.8243$ in English; $0.7827$ vs. $0.6853$ in Telugu) while the 5-layer convolutional residual network initialized its feature detectors; by Step 5,000, PostNet converged tightly onto the Mel $L_1$ trajectory.
2. **English Best Observed Checkpoint (Step 19,000, Validation Loss `2.34890`)**:
   Because English conditions on 88 distinct voice profiles, acoustic reconstruction continued to refine steadily through late training, reaching its lowest observed validation total loss ($2.34890$), Mel $L_1$ loss ($0.6686$), and PostNet $L_1$ loss ($0.6684$) at **Step 19,000 (Cumulative Epoch 95)** before a minor stochastic fluctuation at Step 20,000 ($2.36890$). *(Provenance Note: Because the final Colab training session resumed from a saved checkpoint at Step 13,500 / Epoch 68, the internal checkpoint dictionary records session epoch `27` at Step 19,000, which corresponds to cumulative epoch $68 + 27 = 95$ in `validation_history.csv`).*
3. **Telugu Best Observed Checkpoint (Step 13,000, Validation Loss `2.08688`) & Post-13k Pitch Overfitting**:
   Telugu validation loss reached its lowest observed value at **Step 13,000 (Epoch 91)**: Total $= 2.08688$, Mel $= 0.5039$, PostNet $= 0.5039$, Pitch $= 0.7481$. Between Steps 13,000 and 20,000, training loss continued to fall ($0.91 \to 0.7791$), but validation total loss rose to $2.11808$ driven primarily by **validation pitch MSE drifting upward from $0.7481$ to $0.7688$** as the 2-layer pitch predictor began memorizing training-utterance $F_0$ micro-contours. Selecting `checkpoint_best.pth.tar` at Step 13,000 prevented pitch over-smoothing. *(Provenance Note: At Step 20,000, the single-step validation sweep logged inside `checkpoint_latest.pth.tar` is `2.094272`, whereas the full Epoch 140 aggregate recorded in `validation_history.csv` is `2.118075`).*

### 12.3 Training & Validation Loss Curves

![English Training Total Loss Curve (Steps 100–20,000)](figures/graph1_train_total_loss.png)

![Telugu Training Total Loss Curve (Steps 100–20,000)](figures/telugu_graph1_train_total_loss.png)

![English Validation Total Loss Curve (Steps 1,000–20,000)](figures/graph2_val_total_loss.png)

![Telugu Validation Total Loss Curve (Steps 1,000–20,000)](figures/telugu_graph2_val_total_loss.png)

![English Training Component Loss Breakdown](figures/graph3_train_loss_components.png)

![Telugu Training Component Loss Breakdown](figures/telugu_graph3_train_loss_components.png)

---

## 13. Inference Methodology & Prosody Inverse-Normalization

During inference (`English/inference.py`, `Telugu/inference.py`), ground-truth targets (`mels`, `p_targets`, `e_targets`, `d_targets`) are set to `None` and prosody control multipliers default to $\alpha_d = \alpha_p = \alpha_e = 1.0$. To ensure scientific clarity across all tables and plots, we strictly distinguish six prosodic quantities:

1. **Normalized Pitch ($z_p$, dimensionless $z$-score)**: Raw frame-level output of `pitch_predictor` (plotted directly on English pitch plots).
2. **Inverse-Normalized Physical Pitch ($F_0$, in Hz)**: Reconstructed via corpus `stats.json` (plotted directly on Telugu pitch plots and tabulated for both languages):
   $$F_{0, t}\;(\text{Hz}) = \hat{z}_{p, t} \cdot \sigma_p + \mu_p$$
3. **All-Frame Pitch vs. Voiced-Only Pitch**:
   - **All-Frame Pitch**: Mean and standard deviation computed across all $L_{\text{mel}}$ frames, including silent `@sp`/`sp` pauses and unvoiced consonants.
   - **Voiced-Only Pitch ($E_{\text{linear}} > 5.0$)**: Mean and standard deviation computed strictly over active voiced frames where physical STFT $L_2$-norm energy exceeds the silence floor ($E_{\text{linear}, t} > 5.0$).
4. **Normalized Energy ($z_e$, dimensionless $z$-score)** vs. **Inverse-Normalized Physical Energy ($E_{\text{linear}}$, STFT $L_2$-norm)**:
   $$E_{\text{linear}, t} = \hat{z}_{e, t} \cdot \sigma_e + \mu_e, \qquad t_{\text{sec}} = \frac{\text{frame\_idx} \times 256}{22,050}$$

---

## 14. Generated English Samples Analysis (E1, E2a, E2b, E3, E4)

All five English assignment sentences were synthesized using the best observed validation checkpoint (`Step 19,000`, `checkpoint_best.pth.tar`) conditioned on Speaker Profile `0` (Female Indian English) and vocoded via HiFi-GAN Universal at $22,050\text{ Hz}$.

### 14.1 Master Quantitative Summary Table — English Test Set

| ID | Tokens ($L_{\text{src}}$) | Mel Frames ($L_{\text{mel}}$) | Audio Dur (s) | Token Dur Mean ± Std [Min, Max] | All-Frame Pitch $z_p$ | All-Frame Pitch $F_0$ (Hz) | Voiced Frames ($E>5$) | Voiced Pitch $F_0$ (Hz) | All-Frame Energy $z_e$ | Physical Energy $E_{\text{linear}}$ ($L_2$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | 85 | 522 | 6.060 | 6.14 ± 2.05 [2, 11] | 1.111 ± 0.336 | 243.98 ± 19.98 | 505 / 522 | **243.93 ± 20.11** | 0.008 ± 0.444 | 27.54 ± 13.47 [2.38, 64.53] |
| **E2(a)** | 50 | 333 | 3.866 | 6.66 ± 2.40 [3, 12] | 0.903 ± 0.444 | 231.60 ± 26.42 | 312 / 333 | **230.10 ± 24.97** | -0.062 ± 0.468 | 25.44 ± 14.21 [2.86, 60.70] |
| **E2(b)** | 56 | 358 | 4.156 | 6.39 ± 2.61 [2, 12] | 0.882 ± 0.331 | 230.39 ± 19.69 | 331 / 358 | **230.36 ± 20.35** | -0.096 ± 0.473 | 24.39 ± 14.36 [2.02, 54.16] |
| **E3** | 37 | 240 | 2.786 | 6.49 ± 2.47 [3, 16] | 1.011 ± 0.321 | 238.05 ± 19.10 | 221 / 240 | **238.75 ± 19.44** | -0.021 ± 0.562 | 26.67 ± 17.05 [2.77, 74.45] |
| **E4** | 81 | 567 | 6.583 | 7.00 ± 4.75 [2, 40] | 1.016 ± 0.300 | 238.35 ± 17.85 | 489 / 567 | **237.70 ± 18.61** | -0.051 ± 0.566 | 25.78 ± 17.17 [0.05, 80.48] |

### 14.2 Individual Sample Evaluations (E1 through E4)

#### Sample E1 — Polysyllabic Morphological Stress Test
- **Exact Input Text**: *"The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."*
- **Phonetic Transcription ($85$ ARPAbet tokens)**: The 28-letter word *"antidisestablishmentarianism"* was decomposed via `g2p_en` into 28 phonemes (`AE2 N T AY0 D IH0 S AH0 S T AE2 B L IH0 SH M AH0 N T EH1 R IY0 AH0 N IH0 Z AH0 M`, tokens 2–29) carrying secondary stress on `AE2` and primary stress on `EH1`.
- **Duration Analysis**: Total duration is $522$ frames ($6.060\text{ s}$), with a compact mean of $6.14 \pm 2.05$ frames/token. Within *"antidisestablishmentarianism"*, the model allocated $166$ frames ($1.93\text{ s}$, averaging $5.93$ frames/phoneme), peaking at $10$ frames ($116\text{ ms}$) on the palato-alveolar fricative `SH` (token 16) and $8$ frames on stressed diphthongs/vowels (`AY0`, `IY0`). Utterance-final lengthening occurs on `"proceedings"` (`IH0 NG Z` = `[5, 10, 11]` frames).
- **Pitch & Energy Analysis**: Voiced $F_0$ averages $243.93 \pm 20.11\text{ Hz}$ ($z_p = 1.110 \pm 0.338$). Pitch reaches its global peak early on the compound word at frame $53$ ($0.62\text{ s}$, $298.8\text{ Hz}$, $z_p = 2.033$) and declines smoothly to $191.4\text{ Hz}$ ($z_p = 0.227$) at frame $521$. Physical energy averages $27.54 \pm 13.47$, peaking at $64.53$ ($z_e = 1.226$) at frame $392$ ($4.55\text{ s}$) on the stressed syllable `AA2 R` of *"parliamentary"*.
- **Pronunciation & Speech Quality**: Highly intelligible, artifact-free articulation of all 12 syllables of *"antidisestablishmentarianism"*. Because the Length Regulator enforces deterministic frame expansion, the model completely avoids the attention-skipping and syllable-looping failures typical of autoregressive TTS on long morphological chains.

#### Sample E2(a) — Contextual Homograph "conduct" (Noun Role)
- **Exact Input Text**: *"The orchestra's conduct during the ceremony was praised by the audience."*
- **Phonetic Transcription ($50$ ARPAbet tokens)**: `"conduct"` (tokens 11–17) is transcribed via CMUdict as `['K', 'AA1', 'N', 'D', 'AH0', 'K', 'T']`.
- **Duration Analysis**: Total duration is $333$ frames ($3.866\text{ s}$), averaging $6.66 \pm 2.40$ frames/token. The noun *"conduct"* spans $42$ frames ($0.488\text{ s}$) with per-phoneme durations `[6, 4, 10, 4, 5, 8, 5]` (first syllable `K AA1 N` = $20$ frames; second syllable `D AH0 K T` = $22$ frames). Longest tokens are the sibilants `S` in *"ceremony"* (token 24, $12$ frames) and utterance-final `S` in *"audience"* (token 49, $12$ frames).
- **Pitch & Energy Analysis**: Voiced $F_0$ averages $230.10 \pm 24.97\text{ Hz}$ ($z_p = 0.877 \pm 0.420$). The global pitch maximum occurs directly on the primary-stressed first syllable `AA1 N` of *"conduct"* at frame $70$ ($0.81\text{ s}$, $289.7\text{ Hz}$, $z_p = 1.881$), followed by a terminal declarative fall to $146.6\text{ Hz}$ ($z_p = -0.527$) at frame $332$. Physical energy averages $25.44 \pm 14.21$, peaking at $60.70$ ($z_e = 1.100$) at frame $14$ on `"orchestra's"` (`AO1 R`).
- **Pronunciation & Speech Quality**: Natural, clear Indian English pronunciation. First-syllable primary stress (`AA1`) combined with the $289.7\text{ Hz}$ pitch peak yields an authentic noun realization (/ˈkɑːndʌkt/) matching the syntactic context.

#### Sample E2(b) — Contextual Homograph "conduct" (Verb Role)
- **Exact Input Text**: *"The researchers will conduct an experiment to evaluate the proposed method."*
- **Phonetic Transcription ($56$ ARPAbet tokens)**: Because CMUdict is a static dictionary without POS disambiguation, `"conduct"` (tokens 12–18) receives the **identical** phoneme sequence `['K', 'AA1', 'N', 'D', 'AH0', 'K', 'T']` as in E2(a).
- **Duration Analysis**: Total duration is $358$ frames ($4.156\text{ s}$), averaging $6.39 \pm 2.61$ frames/token. Here `"conduct"` spans $46$ frames ($0.534\text{ s}$) with durations `[10, 5, 8, 3, 7, 8, 5]` ($+4$ frames longer than E2a due to pre-boundary lengthening on initial `K` [$10$ frames] after auxiliary `"will"`). Longest utterance tokens are `CH` ($12$ frames) and `ER0 Z` ($[11, 11]$ frames) in *"researchers"* and `OW1` ($11$ frames) in *"proposed"*.
- **Pitch & Energy Analysis**: Voiced $F_0$ averages $230.36 \pm 20.35\text{ Hz}$ ($z_p = 0.882 \pm 0.342$), with a narrower pitch range ($179.8 - 272.9\text{ Hz}$) than E2(a). Over `"conduct"`, mean pitch is lower ($z_p = 1.022$, peak $1.423$ vs. $1.881$ in E2a). Physical energy averages $24.39 \pm 14.36$, peaking at $54.16$ ($z_e = 0.884$) at frame $50$ on *"researchers"*.
- **Pronunciation & Speech Quality**: Segmentally crisp and fluent across all words (*"experiment"*, *"evaluate"*, *"proposed"*), but *"conduct"* retains noun-like first-syllable vowel quality (`AA1`) rather than shifting to verb-stressed /kənˈdʌkt/ because of the static lexicon lookup.

#### Sample E3 — Orthographic Split Anomaly ("rhyth m")
- **Exact Input Text**: *"The rhyth m of the orchestra changed unexpectedly."*
- **Phonetic Transcription ($37$ ARPAbet tokens)**: Preserving the exact prompt text containing a space inside `"rhyth m"` caused the tokenizer to process `"rhyth"` (OOV $\to$ `g2p_en`: `R AY1 DH`, tokens 2–4) and `"m"` (letter-name lookup: `EH1 M`, tokens 5–6) as two distinct words rather than canonical `R IH1 DH AH0 M`.
- **Duration Analysis**: Shortest English utterance at $240$ frames ($2.786\text{ s}$), averaging $6.49 \pm 2.47$ frames/token. The fragmented sequence `R AY1 DH EH1 M` consumes $4 + 10 + 7 + 6 + 7 = 34$ frames ($0.395\text{ s}$), with $20$ frames ($232\text{ ms}$) across the `DH EH1 M` juncture. Utterance-final `IY0` in *"unexpectedly"* undergoes strong phrase-final lengthening ($16$ frames / $186\text{ ms}$).
- **Pitch & Energy Analysis**: Voiced $F_0$ averages $238.75 \pm 19.44\text{ Hz}$ ($z_p = 1.023 \pm 0.327$), peaking at frame $116$ ($1.35\text{ s}$, $273.7\text{ Hz}$, $z_p = 1.611$) on `"changed"` (`CH EY1 N JH D`) and reaching its minimum ($192.9\text{ Hz}$, $z_p = 0.251$) at frame $213$. Physical energy averages $26.67 \pm 17.05$ with a sharp peak of $74.45$ ($z_e = 1.553$) at frame $32$ ($0.37\text{ s}$) on the open vowel `EH1` of `"m"`.
- **Pronunciation & Speech Quality**: Directly demonstrates how non-autoregressive TTS faithfully renders front-end tokenization: the split `"rhyth m"` is audibly pronounced as /raɪð ɛm/, whereas the remainder of the sentence (*"of the orchestra changed unexpectedly"*) is synthesized with smooth, natural prosody.

#### Sample E4 — Temporal Numerals, Dates & Code-Mixed OOV ("andaroo")
- **Exact Input Text**: *"The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."*
- **Phonetic Transcription ($81$ ARPAbet tokens)**: `English/text/normalizer.py` expanded `"10:45 AM"` and `"12 November 2026"` into words, while the code-mixed Telugu-English word `"andaroo"` (meaning *"everyone"*) was handled by `g2p_en` as `['AE0', 'N', 'D', 'AA1', 'R', 'OW0']` (tokens 64–69). Two explicit `sp` pause tokens were inserted at the colon (token 20, $7$ frames) and comma (token 61, $40$ frames / $0.464\text{ s}$).
- **Duration Analysis**: Longest English utterance at $567$ frames ($6.583\text{ s}$), averaging $7.00 \pm 4.75$ frames/token. The comma pause `sp` at token 61 receives $40$ frames ($464\text{ ms}$), cleanly separating the temporal clause from the concluding clause *"so andaroo should arrive on time"*.
- **Pitch & Energy Analysis**: Voiced $F_0$ averages $237.70 \pm 18.61\text{ Hz}$ ($z_p = 1.005 \pm 0.313$, across $489$ voiced frames), peaking at $292.0\text{ Hz}$ ($z_p = 1.918$) at frame $29$ (`"meeting"`). During the 40-frame comma pause (frames 403–442), physical energy drops to near-zero ($0.05\text{ }L_2$, $z_e = -0.898$) before recovering cleanly on *"so andaroo"* ($46.2\text{ }L_2$). Global energy peaks at $80.48$ ($z_e = 1.751$) at frame $178$ ($2.07\text{ s}$).
- **Pronunciation & Speech Quality**: Clear clause boundary pausing and smooth code-mixed rendering of *"andaroo"* (/ændɑːroʊ/). Minor regex interaction in the colon/hyphen normalizer contracted the middle syllables of `"forty-five"` and `"six"` in `"twenty-six"`, which is documented as a text-normalization edge case in Section 18.

---

## 15. Generated Telugu Samples Analysis (T1, T2a, T2b, T3, T4)

All five Telugu assignment sentences were synthesized using the best observed validation checkpoint (`Step 13,000`, `checkpoint_best.pth.tar`) conditioned on Speaker `spk_001` (Male Native Telugu) in `telugu_digits` mode using the verified text manifest `Results/Telugu/official_samples.json`.

### 15.1 Master Quantitative Summary Table — Telugu Test Set

| ID | Tokens ($L_{\text{src}}$) | `@sp` Pauses (Frames) | Mel Frames ($L_{\text{mel}}$) | Audio Dur (s) | Token Dur Mean ± Std [Min, Max] | All-Frame Pitch $z_p$ | All-Frame Pitch $F_0$ (Hz) | Voiced Frames ($E>5$) | Voiced Pitch $F_0$ (Hz) | All-Frame Energy $z_e$ | Physical Energy $E_{\text{linear}}$ ($L_2$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1** | 67 | 5 (68f) | 464 | 5.387 | 6.93 ± 4.32 [1, 18] | -0.073 ± 0.968 | 192.25 ± 44.93 | 377 / 464 | **197.25 ± 39.30** | 0.289 ± 1.102 | 35.82 ± 29.37 [-0.22, 116.54] |
| **T2(a)** | 33 | 5 (74f) | 283 | 3.286 | 8.58 ± 5.57 [1, 20] | -0.118 ± 0.887 | 190.15 ± 41.19 | 206 / 283 | **196.00 ± 37.46** | 0.160 ± 1.081 | 32.37 ± 28.82 [-0.25, 114.51] |
| **T2(b)** | 39 | 5 (91f) | 291 | 3.378 | 7.46 ± 5.77 [1, 26] | -0.278 ± 0.953 | 182.69 ± 44.22 | 201 / 291 | **186.67 ± 44.03** | -0.126 ± 0.877 | 24.76 ± 23.37 [-0.20, 83.00] |
| **T3** | 32 | 3 (37f) | 208 | 2.415 | 6.50 ± 4.68 [1, 17] | -0.475 ± 0.783 | 173.55 ± 36.32 | 165 / 208 | **179.57 ± 31.53** | -0.086 ± 0.772 | 25.81 ± 20.57 [0.01, 102.67] |
| **T4** | 89 | 13 (144f) | 692 | 8.034 | 7.78 ± 4.55 [1, 22] | -0.089 ± 0.806 | 191.50 ± 37.40 | 541 / 692 | **195.47 ± 33.15** | 0.168 ± 0.946 | 32.59 ± 25.23 [-0.32, 106.46] |

### 15.2 Individual Sample Evaluations (T1 through T4)

#### Sample T1 — Polysyllabic Technical Compound & Complex Conjuncts
- **Exact Input Text**: న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.
- **Grapheme Tokenization ($67$ tokens, $0$ `<unk>`)**: The 20-grapheme transliterated compound `న్యూరోఫార్మకోలాజికల్` (*"neuropharmacological"*, tokens 0–19: `['న', '్', 'య', 'ూ', 'ర', 'ో', 'ఫ', 'ా', 'ర', '్', 'మ', 'క', 'ో', 'ల', 'ా', 'జ', 'ి', 'క', 'ల', '్']`) is tokenized as a single uninterrupted lexical unit followed by 5 inter-word `@sp` pauses (`[6, 18, 13, 13, 18]` frames; $68$ pause frames total).
- **Duration Analysis**: Total duration is $464$ frames ($5.387\text{ s}$), averaging $6.93 \pm 4.32$ frames/token. The 20-token compound `న్యూరోఫార్మకోలాజికల్` spans $120$ frames ($1.393\text{ s}$), with virama (`్`) closures taking $1-2$ frames and long vowel matras (`ా`, `ో`, `ూ`) taking $5-7$ frames. Word-final vowels before `@sp` pauses exhibit strong pre-boundary lengthening (`ో` in `పరిశోధనలో` = $18$ frames; `ి` in `ప్రభావాన్ని` = $18$ frames).
- **Pitch & Energy Analysis**: All-frame pitch is $192.25 \pm 44.93\text{ Hz}$ ($z_p = -0.073 \pm 0.968$); voiced-only pitch ($377$ active frames) is $197.25 \pm 39.30\text{ Hz}$. Pitch peaks early on `న్యూరోఫార్మకోలాజికల్` at frame $22$ ($0.26\text{ s}$, $281.5\text{ Hz}$, $z_p = 1.851$) and dips to $87.5 - 115\text{ Hz}$ during the 18-frame `@sp` pauses. Physical energy averages $35.82 \pm 29.37$, peaking at $116.54$ ($z_e = 3.317$) at frame $343$ ($3.98\text{ s}$) on the open `ా` vowel of `ప్రభావాన్ని`.
- **Pronunciation & Speech Quality**: AI-assisted subjective rating: Clarity $4.0/5$, Fluency $3.5/5$, Naturalness $3.0/5$ (Mean $3.50/5$). All complex conjuncts (`న్యూ`, `ర్మ`, `ప్ర`, `న్ని`, `శ్లే`) are articulated accurately without phoneme dropping, though the 20-grapheme English loanword exhibits a slightly mechanical syllable-timed cadence across frames 1–120.

#### Sample T2(a) — Contextual Homograph "పట్టు" (Physical Grip / Hold Meaning)
- **Exact Input Text**: తాడుపై అతని పట్టు చాలా బలంగా ఉంది.
- **Grapheme Tokenization ($33$ tokens, $0$ `<unk>`)**: Contains 5 inter-word `@sp` tokens (`[13, 20, 11, 20, 10]` frames; $74$ pause frames total). The target word `పట్టు` (*"grip/hold"*, tokens 12–16) is tokenized as `['ప', 'ట', '్', 'ట', 'ు']`.
- **Duration Analysis**: Total duration is $283$ frames ($3.286\text{ s}$), averaging $8.58 \pm 5.57$ frames/token. The target word `పట్టు` receives **exactly $20$ frames ($0.232\text{ s}$)** distributed as `[5, 4, 2, 7, 2]`: $5$ frames for onset `ప`, $4 + 2 + 7 = 13$ frames for the geminate retroflex stop closure and release `ట్టు` (`ట + ్ + ట`), and $2$ frames for short vowel `ు`. Diphthong `ై` in `తాడుపై` ($18$ frames) and long `ా` in `బలంగా` ($18$ frames) receive prominent vowel durations.
- **Pitch & Energy Analysis**: All-frame pitch is $190.15 \pm 41.19\text{ Hz}$; voiced-only pitch ($206$ frames) is $196.00 \pm 37.46\text{ Hz}$. Across `పట్టు` (frames 111–130), mean pitch is $199.2\text{ Hz}$ (range $169.4 - 249.7\text{ Hz}$). Utterance pitch peaks at frame $99$ ($1.15\text{ s}$, $329.3\text{ Hz}$) at the onset transition following `@sp`. Physical energy averages $32.37 \pm 28.82$, peaking at $114.51$ ($z_e = 3.241$) at frame $30$ (`తాడుపై`).
- **Pronunciation & Speech Quality**: Highest-rated Telugu sample (Clarity $4.5/5$, Fluency $4.0/5$, Naturalness $3.5/5$, Mean $4.00/5$). Crisp retroflex gemination on `పట్టు` and clear nasalization (`ం`) on `బలంగా ఉంది`.

#### Sample T2(b) — Contextual Homograph "పట్టు" (Silk Fabric Meaning)
- **Exact Input Text**: ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.
- **Grapheme Tokenization ($39$ tokens, $0$ `<unk>`)**: Contains 5 inter-word `@sp` tokens (`[26, 20, 17, 11, 17]` frames; $91$ pause frames total). The target word `పట్టు` (*"silk"*, tokens 4–8) is tokenized as the **identical** sequence `['ప', 'ట', '్', 'ట', 'ు']`.
- **Duration Analysis**: Total duration is $291$ frames ($3.378\text{ s}$), averaging $7.46 \pm 5.77$ frames/token. Remarkably, `పట్టు` in T2(b) also receives **identically $20$ frames ($0.232\text{ s}$)** with sub-token durations `[5, 5, 1, 7, 2]` (differing by just $\pm 1$ frame between implosion `ట` and virama `్` compared to `[5, 4, 2, 7, 2]` in T2a). The utterance-final verb `వచ్చింది` (`['వ', 'చ', '్', 'చ', 'ి', 'ం', 'ద', 'ి']`, tokens 31–38) is compressed into $28$ frames ($0.325\text{ s}$), with the geminate palatal affricate `చ్చ` (`చ + ్ + చ`) allocated only $3 + 2 + 1 = 6$ frames.
- **Pitch & Energy Analysis**: All-frame pitch is $182.69 \pm 44.22\text{ Hz}$; voiced-only pitch ($201$ frames) is $186.67 \pm 44.03\text{ Hz}$. Over `పట్టు` (frames 62–81), mean pitch is $206.7\text{ Hz}$ ($+7.5\text{ Hz}$ higher than T2a due to its earlier sentence position as modifier of `చీరను`). Physical energy is lower overall ($24.76 \pm 23.37$, peak $83.00$ at frame $158$ on `ధరించి`), with reduced coda energy on `వచ్చింది`.
- **Pronunciation & Speech Quality**: AI-assisted rating: Clarity $2.5/5$, Fluency $2.5/5$, Naturalness $2.0/5$ (Mean $2.33/5$). While `ఆమె పట్టు చీరను` is clear, rapid duration allocation and lower energy on `ధరించి` and `వచ్చింది` cause mild perceptual softening of the palatal affricates.

#### Sample T3 — Short Declarative Baseline & Sibilant Geminate (`నిశ్శబ్దంగా`)
- **Exact Input Text**: పులి అడవిలో నిశ్శబ్దంగా నడిచింది.
- **Grapheme Tokenization ($32$ tokens, $0$ `<unk>`)**: Shortest Telugu utterance with 3 `@sp` pauses (`[8, 16, 13]` frames; $37$ pause frames total). Contains the complex double-conjunct word `నిశ్శబ్దంగా` (`['న', 'ి', 'శ', '్', 'శ', 'బ', '్', 'ద', 'ం', 'గ', 'ా']`, tokens 12–22), which combines a geminate palatal sibilant (`శ్శ`) and a voiced labial-dental stop cluster (`బ్ద`).
- **Duration Analysis**: Total duration is $208$ frames ($2.415\text{ s}$), averaging $6.50 \pm 4.68$ frames/token. `నిశ్శబ్దంగా` spans $79$ frames ($0.917\text{ s}$), with `శ్శ` receiving $8 + 1 + 7 = 16$ frames, `బ్ద` receiving $6 + 2 + 5 = 13$ frames, anusvara `ం` receiving $12$ frames, and final `గా` receiving $1 + 16 = 17$ frames.
- **Pitch & Energy Analysis**: Exhibits the most compact pitch contour in the Telugu test set: all-frame pitch is $173.55 \pm 36.32\text{ Hz}$ and voiced-only pitch ($165$ frames) is $179.57 \pm 31.53\text{ Hz}$ (peaking at $275.5\text{ Hz}$ at frame $160$). Physical energy averages $25.81 \pm 20.57$, peaking at $102.67$ ($z_e = 2.797$) at frame $44$ on `అడవిలో`, but tapering to $< 8.0$ on the final syllable `ది` of `నడిచింది`.
- **Pronunciation & Speech Quality**: AI-assisted rating: Clarity $3.5/5$, Fluency $3.0/5$, Naturalness $2.5/5$ (Mean $3.00/5$). Natural rhythm and accurate realization of `నిశ్శబ్దంగా`, with slight perceptual attenuation on the utterance-final syllable `ది`.

#### Sample T4 — Long Compound Utterance with Native Telugu Digits
- **Exact Input Text**: ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.
- **Grapheme Tokenization ($89$ tokens, $0$ `<unk>`)**: Longest test utterance across both languages ($89$ tokens, $13$ `@sp` pauses totaling $144$ pause frames). In `telugu_digits` mode, ASCII numerals `2026`, `12`, and `10:45` map to native Telugu digit tokens `['౨', '౦', '౨', '౬']` (tokens 10–13), `['౧', '౨']` (tokens 22–23), and `['౧', '౦', '@sp', '౪', '౫']` (tokens 36–40).
- **Duration Analysis**: Spans $692$ mel frames ($8.034\text{ s}$), averaging $7.78 \pm 4.55$ frames/token. However, each Telugu digit character (`౨`, `౦`, `౬`, `౧`, `౪`, `౫`) is a single token in the vocabulary and receives only $3 - 11$ frames ($35 - 128\text{ ms}$) per digit: `౨౦౨౬` (`2026`) is compressed into $11 + 8 + 7 + 9 = 35$ frames ($0.406\text{ s}$) and `౧౦ @sp ౪౫` (`10:45`) into $7 + 3 + 5 + 6 + 9 = 30$ frames ($0.348\text{ s}$), whereas spoken Telugu words for those numbers require 12–18 graphemes. Major clausal boundaries at `తేదీ` (`@sp` token 30 = $22$ frames) and the comma after `ప్రారంభమవుతుంది` (`@sp` token 64 = $21$ frames) provide clear phrasal segmentation.
- **Pitch & Energy Analysis**: All-frame pitch is $191.50 \pm 37.40\text{ Hz}$; voiced-only pitch ($541$ frames) is $195.47 \pm 33.15\text{ Hz}$ (peaking at $331.3\text{ Hz}$ at frame $657$). Physical energy averages $32.59 \pm 25.23$, peaking at $106.46$ ($z_e = 2.939$) at frame $169$ (`నవంబర్`).
- **Pronunciation & Speech Quality**: AI-assisted rating: Clarity $2.0/5$, Fluency $2.0/5$, Naturalness $1.5/5$ (Mean $1.83/5$). While pure lexical words (`సమావేశం`, `ప్రారంభమవుతుంది`, `సమయానికి`) are well-formed, single-token digit compression on `2026`, `12వ`, and `10:45` causes perceptual rushing, motivating our `--digit_mode words` lexical expansion recommendation in Section 19.

---

## 16. Variance Adaptor Plot Analysis (All 30 Contours)

Below is the complete visual and quantitative inspection of all **30 Variance Adaptor plots** (`10 Duration`, `10 Pitch`, and `10 Energy` contours) generated from the ten canonical assignment sentences.

### 16.1 English Sample E1 — Variance Adaptor Contours
- **Duration Plot (`E1_duration.png`)**: Across $85$ ARPAbet tokens ($522$ frames), durations remain tightly bounded between $2$ and $11$ frames ($\mu = 6.14, \sigma = 2.05$). Stressed vowels (`AY0` [8f], `IY0` [8–10f], `EY1` [10f]) and fricatives/nasals (`SH` [10f], `M` [10f], `S` [10f], final `Z` [11f]) form clear local peaks, while unstressed schwas (`AH0` [2–4f]) and stop closures (`T`, `D`, `B` [3–5f]) form troughs.
- **Pitch Plot (`E1_pitch.png`)**: Normalized pitch $z_p$ begins high on *"The antidisestablishmentarianism"* (peaking at $z_p = 2.033$ / $298.8\text{ Hz}$ at frame $53$), resets to local accent peaks on *"movement gained"* ($z_p = 1.48$), *"momentum"* ($z_p = 1.45$), and *"parliamentary"* ($z_p = 1.38$), and exhibits classic declarative declination down to $z_p = 0.227$ ($191.4\text{ Hz}$) at frame $521$.
- **Energy Plot (`E1_energy.png`)**: Normalized energy $z_e$ ($\mu = 0.008, \sigma = 0.444$) oscillates rhythmically at syllable rate, rising to $+0.8$ to $+1.23$ on vocalic nuclei (global peak $z_e = 1.226$ / $64.53\text{ }L_2$ at frame $392$ on *"parliamentary"*) and dipping to $-0.75$ to $-0.82$ during voiceless stop closures (`T`, `P`, `K`).

![Sample E1 Duration Contour](figures/E1_duration.png)

![Sample E1 Pitch Contour](figures/E1_pitch.png)

![Sample E1 Energy Contour](figures/E1_energy.png)

### 16.2 English Sample E2(a) — Variance Adaptor Contours ("conduct" as Noun)
- **Duration Plot (`E2a_duration.png`)**: Across $50$ tokens ($333$ frames, $\mu = 6.66 \pm 2.40$), sibilant fricatives (`S` in *"ceremony"* [12f], final `S` in *"audience"* [12f], `Z` in *"orchestra's"* [11f]) and nasal codas (`N` in *"conduct"* [10f] and *"audience"* [10f]) dominate token durations. The noun *"conduct"* (`K AA1 N D AH0 K T`, tokens 11–17) spans $42$ frames (`[6, 4, 10, 4, 5, 8, 5]`).
- **Pitch Plot (`E2a_pitch.png`)**: Exhibits a pronounced focal pitch accent on *"conduct"*, rising sharply to the utterance maximum $z_p = 1.881$ ($289.7\text{ Hz}$) at frame $70$ on the primary-stressed first syllable `AA1 N`, followed by secondary peaks on *"ceremony"* ($z_p = 1.34$) and *"praised"* ($z_p = 1.29$), and a steep terminal fall to $z_p = -0.527$ ($146.6\text{ Hz}$) at frame $332$.
- **Energy Plot (`E2a_energy.png`)**: Energy peaks strongly on the initial stressed vowel `AO1` of *"orchestra's"* ($z_e = 1.100$ / $60.70\text{ }L_2$ at frame $14$) and on *"ceremony"* ($z_e = 0.89$) and *"audience"* ($z_e = 0.94$), with sharp $-0.80$ troughs marking plosive closures (`K`, `T`, `P`).

![Sample E2a Duration Contour](figures/E2a_duration.png)

![Sample E2a Pitch Contour](figures/E2a_pitch.png)

![Sample E2a Energy Contour](figures/E2a_energy.png)

### 16.3 English Sample E2(b) — Variance Adaptor Contours ("conduct" as Verb)
- **Duration Plot (`E2b_duration.png`)**: Across $56$ tokens ($358$ frames, $\mu = 6.39 \pm 2.61$), prominent durations appear on *"researchers"* (`CH ER0 Z` = `[12, 11, 11]` frames), *"proposed"* (`OW1 Z` = `[11, 9]` frames), and final `D` in *"method"* ($10$ frames). Verb *"conduct"* (tokens 12–18) spans $46$ frames (`[10, 5, 8, 3, 7, 8, 5]`), with initial `K` lengthened to $10$ frames after `"will"`.
- **Pitch Plot (`E2b_pitch.png`)**: Unlike E2(a), pitch in E2(b) starts at $z_p = 1.597$ ($272.9\text{ Hz}$) on `"The researchers"` and remains flatter across the verb phrase `"will conduct"` (local peak $z_p = 1.423$, mean $1.022$), reflecting reduced prosodic prominence in predicate verb position before declining to $z_p = 0.031$ ($179.8\text{ Hz}$) at frame $357$.
- **Energy Plot (`E2b_energy.png`)**: Energy is slightly more uniform ($\mu = -0.096 \pm 0.473$, max $z_e = 0.884$ / $54.16\text{ }L_2$ at frame $50$ on *"researchers"*), with 8 distinct syllabic peaks corresponding to the 8 lexical words of the sentence.

![Sample E2b Duration Contour](figures/E2b_duration.png)

![Sample E2b Pitch Contour](figures/E2b_pitch.png)

![Sample E2b Energy Contour](figures/E2b_energy.png)

### 16.4 English Sample E3 — Variance Adaptor Contours ("rhyth m" Split)
- **Duration Plot (`E3_duration.png`)**: Across $37$ tokens ($240$ frames, $\mu = 6.49 \pm 2.47$), the split tokens `R AY1 DH` (`[4, 10, 7]`) and `EH1 M` (`[6, 7]`) occupy $34$ frames at the start of the plot, while utterance-final `IY0` in *"unexpectedly"* reaches the maximum duration of $16$ frames ($186\text{ ms}$).
- **Pitch Plot (`E3_pitch.png`)**: Shows a double-peaked contour across the separated fragments `"rhyth"` ($z_p = 1.36$) and `"m"` ($z_p = 1.28$), a mid-sentence focal peak on `"changed"` at frame $116$ ($z_p = 1.611$ / $273.7\text{ Hz}$), and a trough of $z_p = 0.251$ ($192.9\text{ Hz}$) at frame $213$ on `"unexpectedly"`.
- **Energy Plot (`E3_energy.png`)**: Displays a prominent spike at frame $32$ ($z_e = 1.553$ / $74.45\text{ }L_2$) directly on the isolated vowel `EH1` of `"m"`, visually confirming the acoustic salience of the orthographic split.

![Sample E3 Duration Contour](figures/E3_duration.png)

![Sample E3 Pitch Contour](figures/E3_pitch.png)

![Sample E3 Energy Contour](figures/E3_energy.png)

### 16.5 English Sample E4 — Variance Adaptor Contours (Numbers, Dates & "andaroo")
- **Duration Plot (`E4_duration.png`)**: Across $81$ tokens ($567$ frames, $\mu = 7.00 \pm 4.75$), the standout feature is the 40-frame ($464\text{ ms}$) comma pause `sp` at token 61, alongside stressed diphthong `AY1` ($18$ frames at token 59; $12$ frames at token 74 in *"arrive"*).
- **Pitch Plot (`E4_pitch.png`)**: Starts with a high peak on *"meeting"* ($z_p = 1.918$ / $292.0\text{ Hz}$ at frame $29$), maintains a steady plateau ($z_p \approx 0.8 - 1.4$) across the date/time expression, drops during the 40-frame comma pause (frames 403–442), and executes a clear pitch reset ($z_p = 1.52$) at the onset of the second clause *"so andaroo should arrive on time"*.
- **Energy Plot (`E4_energy.png`)**: Reaches its global maximum at frame $178$ ($z_e = 1.751$ / $80.48\text{ }L_2$) on the time expression, drops flat to the floor ($z_e = -0.898$ / $0.05\text{ }L_2$) across the 40-frame comma pause around $4.7 - 5.1\text{ s}$, and resumes rhythmic syllabic pulses through $6.58\text{ s}$.

![Sample E4 Duration Contour](figures/E4_duration.png)

![Sample E4 Pitch Contour](figures/E4_pitch.png)

![Sample E4 Energy Contour](figures/E4_energy.png)

### 16.6 Telugu Sample T1 — Variance Adaptor Contours
- **Duration Plot (`T1_duration.png`)**: Across $67$ grapheme tokens ($464$ frames, $\mu = 6.93 \pm 4.32$), the first 20 bars represent the uninterrupted compound `న్యూరోఫార్మకోలాజికల్` (viramas `్` at $1-2$ frames; consonants/matras at $4-11$ frames), followed by five tall bars of $6, 18, 13, 13, 18$ frames corresponding to the `@sp` word boundaries and pre-boundary lengthened vowels (`ో`, `న`, `ి` at $18$ frames).
- **Pitch Plot (`T1_pitch.png`)**: Plotted in physical Hz against the corpus mean reference line ($195.6\text{ Hz}$). $F_0$ peaks at $281.5\text{ Hz}$ ($0.26\text{ s}$) on `న్యూరోఫార్మకోలాజికల్`, stays near $200 - 245\text{ Hz}$ on voiced lexical segments, and exhibits five sharp downward V-troughs ($87.5 - 125\text{ Hz}$) aligning with the five `@sp` inter-word pauses.
- **Energy Plot (`T1_energy.png`)**: Physical STFT $L_2$-norm energy ($\mu = 35.82 \pm 29.37$, reference line $28.1$) shows six distinct word-level envelopes separated by zero-energy troughs ($\approx 0.0\text{ }L_2$) at the five `@sp` pauses, reaching its global maximum of $116.54\text{ }L_2$ at $3.98\text{ s}$ on `ప్రభావాన్ని`.

![Sample T1 Duration Contour](figures/T1_duration.png)

![Sample T1 Pitch Contour](figures/T1_pitch.png)

![Sample T1 Energy Contour](figures/T1_energy.png)

### 16.7 Telugu Sample T2(a) — Variance Adaptor Contours ("పట్టు" as Grip)
- **Duration Plot (`T2_a_duration.png`)**: Across $33$ tokens ($283$ frames, $\mu = 8.58 \pm 5.57$), the five `@sp` pauses (`[13, 20, 11, 20, 10]` frames) and word-final long vowels (`ై` [18f], `ి` [18f], `ా` [14f, 18f]) stand out clearly. Tokens 12–16 (`ప, ట, ్, ట, ు`) show the characteristic geminate duration profile `[5, 4, 2, 7, 2]` totaling $20$ frames ($0.232\text{ s}$).
- **Pitch Plot (`T2_a_pitch.png`)**: Voiced $F_0$ averages $196.00\text{ Hz}$ (essentially matching the $195.6\text{ Hz}$ corpus mean dashed line), with a sharp onset spike ($329.3\text{ Hz}$ at $1.15\text{ s}$) coming out of the second `@sp` pause into `పట్టు` (where voiced pitch averages $199.2\text{ Hz}$), followed by steady declination across `చాలా బలంగా ఉంది`.
- **Energy Plot (`T2_a_energy.png`)**: Six crisp word envelopes separated by five zero-energy `@sp` intervals. Energy peaks at $114.51\text{ }L_2$ ($0.35\text{ s}$) on `తాడుపై` and reaches $24.7\text{ }L_2$ mean across `పట్టు` ($1.28 - 1.51\text{ s}$).

![Sample T2a Duration Contour](figures/T2_a_duration.png)

![Sample T2a Pitch Contour](figures/T2_a_pitch.png)

![Sample T2a Energy Contour](figures/T2_a_energy.png)

### 16.8 Telugu Sample T2(b) — Variance Adaptor Contours ("పట్టు" as Silk)
- **Duration Plot (`T2_b_duration.png`)**: Across $39$ tokens ($291$ frames, $\mu = 7.46 \pm 5.77$), the first `@sp` pause after `ఆమె` is $26$ frames ($0.302\text{ s}$), followed immediately by `పట్టు` (tokens 4–8) with durations `[5, 5, 1, 7, 2]` — totaling the **exact same $20$ frames ($0.232\text{ s}$)** as in T2(a). Tokens 31–38 (`వచ్చింది`) show compressed consonant durations (`[5, 3, 2, 1, 2, 6, 2, 7]`).
- **Pitch Plot (`T2_b_pitch.png`)**: Voiced $F_0$ averages $186.67\text{ Hz}$. Over `పట్టు` ($0.72 - 0.95\text{ s}$), pitch averages $206.7\text{ Hz}$ before peaking at $322.5\text{ Hz}$ at $1.53\text{ s}$ on `చీరను` and dropping below the $195.6\text{ Hz}$ reference line during the final two words `వేడుకకు వచ్చింది` ($\approx 150 - 175\text{ Hz}$).
- **Energy Plot (`T2_b_energy.png`)**: Overall energy ($\mu = 24.76 \pm 23.37$) is lower than T2(a), peaking at $83.00\text{ }L_2$ at $1.83\text{ s}$ on `ధరించి` and tapering to a modest $20 - 38\text{ }L_2$ envelope over `వచ్చింది` ($2.9 - 3.38\text{ s}$), visually explaining the softer coda articulation noted in Section 18.

![Sample T2b Duration Contour](figures/T2_b_duration.png)

![Sample T2b Pitch Contour](figures/T2_b_pitch.png)

![Sample T2b Energy Contour](figures/T2_b_energy.png)

### 16.9 Telugu Sample T3 — Variance Adaptor Contours
- **Duration Plot (`T3_duration.png`)**: Across $32$ tokens ($208$ frames, $\mu = 6.50 \pm 4.68$), word-final vowels (`ి` in `పులి` [17f], `ో` in `అడవిలో` [16f], `ా` in `నిశ్శబ్దంగా` [16f]) and the three `@sp` pauses (`[8, 16, 13]` frames) create a highly regular four-word rhythmic structure.
- **Pitch Plot (`T3_pitch.png`)**: The calmest pitch contour in the Telugu set (voiced mean $179.57 \pm 31.53\text{ Hz}$), remaining predominantly between $150\text{ Hz}$ and $215\text{ Hz}$ just below the $195.6\text{ Hz}$ corpus mean, with a single expressive accent peak ($275.5\text{ Hz}$ at $1.86\text{ s}$) at the onset of `నడిచింది`.
- **Energy Plot (`T3_energy.png`)**: Four well-separated word bursts matching the four words `పులి`, `అడవిలో` (global peak $102.67\text{ }L_2$ at $0.51\text{ s}$), `నిశ్శబ్దంగా` (sustained double burst across `శ్శ` and `బ్దం`), and `నడిచింది` (tapering at $2.3 - 2.41\text{ s}$).

![Sample T3 Duration Contour](figures/T3_duration.png)

![Sample T3 Pitch Contour](figures/T3_pitch.png)

![Sample T3 Energy Contour](figures/T3_energy.png)

### 16.10 Telugu Sample T4 — Variance Adaptor Contours
- **Duration Plot (`T4_duration.png`)**: Across $89$ tokens ($692$ frames, $8.034\text{ s}$), the 13 `@sp` pauses (peaking at $22$ frames at token 30 after `తేదీ` and $21$ frames at token 64 at the comma) segment the long sentence into multi-word phrases, while the native digit tokens (`౨, ౦, ౨, ౬`, `౧, ౨`, `౧, ౦`, `౪, ౫`) appear as short $3-11$ frame bars.
- **Pitch Plot (`T4_pitch.png`)**: Sustains a stable voiced baseline of $195.47 \pm 33.15\text{ Hz}$ across the entire 8-second span without long-horizon drift, punctuated by periodic `@sp` pause dips ($100 - 120\text{ Hz}$) and a late emphatic peak ($331.3\text{ Hz}$ at $7.63\text{ s}$) on `సిద్ధంగా`.
- **Energy Plot (`T4_energy.png`)**: Maintains consistent vocal effort ($\mu = 32.59 \pm 25.23$, max $106.46\text{ }L_2$ at $1.96\text{ s}$ on `నవంబర్`) across all 14 lexical groups through $8.034\text{ s}$.

![Sample T4 Duration Contour](figures/T4_duration.png)

![Sample T4 Pitch Contour](figures/T4_pitch.png)

![Sample T4 Energy Contour](figures/T4_energy.png)

---

## 17. Comparative Analysis: English vs. Telugu

### 17.1 Contextual Word Analysis: Homographs & Semantic Disambiguation

A core scientific question raised by test pairs `E2(a)/E2(b)` and `T2(a)/T2(b)` is whether FastSpeech 2 disambiguates lexical homographs based on sentence meaning.

#### Case A: English E2(a) vs. E2(b) — Homograph *"conduct"*

| Metric | E2(a) Noun (*"orchestra's conduct"*) | E2(b) Verb (*"will conduct"*) | Empirical Comparison |
| :--- | :---: | :---: | :--- |
| **Input Phoneme Sequence** | `['K', 'AA1', 'N', 'D', 'AH0', 'K', 'T']` | `['K', 'AA1', 'N', 'D', 'AH0', 'K', 'T']` | **100% Identical** (`AA1` primary stress on syllable 1) |
| **Per-Phoneme Durations** | `[6, 4, 10, 4, 5, 8, 5]` | `[10, 5, 8, 3, 7, 8, 5]` | +4f on initial `K` in E2(b); syllable 1/2 ratio unchanged |
| **Total Word Duration** | **42 frames (`0.488 s`)** | **46 frames (`0.534 s`)** | +4 frames (`+46.4 ms`) in E2(b) |
| **Word Pitch Mean / Peak ($z_p$)** | **Mean `1.204` / Peak `1.881`** | **Mean `1.022` / Peak `1.423`** | Higher pitch accent in noun phrase E2(a) |
| **Word Energy Mean ($z_e$)** | `-0.267` | `-0.234` | Statistically equivalent ($\Delta z_e = 0.033$) |

- **Scientific Conclusion**: Standard English distinguishes the noun /ˈkɒndʌkt/ (syllable 1 stress) from the verb /kənˈdʌkt/ (syllable 2 stress, reduced first vowel `AH0`). Because CMUdict lookup is purely lexical and lacks a syntactic Part-of-Speech (POS) tagger, both sentences fed the noun-stressed sequence `K AA1 N D AH0 K T` into the encoder. While multi-head self-attention modulated the pitch peak ($1.881$ vs. $1.423$) and initial stop closure ($6$ vs. $10$ frames) due to different surrounding tokens (*"orchestra's"* vs. *"will"*), **these experiments do not demonstrate semantic disambiguation of contextual meanings**.

#### Case B: Telugu T2(a) vs. T2(b) — Homograph *"పట్టు"*

| Metric | T2(a) Physical Grip (*"అతని పట్టు"*) | T2(b) Silk Fabric (*"పట్టు చీరను"*) | Empirical Comparison |
| :--- | :---: | :---: | :--- |
| **Input Grapheme Sequence** | `['ప', 'ట', '్', 'ట', 'ు']` | `['ప', 'ట', '్', 'ట', 'ు']` | **100% Identical** (5 grapheme tokens) |
| **Per-Grapheme Durations** | `[5, 4, 2, 7, 2]` | `[5, 5, 1, 7, 2]` | Identical geminate closure+release (`4+2+7 = 5+1+7 = 13f`) |
| **Total Word Duration** | **20 frames (`0.232 s`)** | **20 frames (`0.232 s`)** | **Identical to the exact mel frame (`0.232 s`)** |
| **Word Pitch Mean / Range (Hz)** | **Mean `199.2 Hz`** (`169.4–249.7 Hz`) | **Mean `206.7 Hz`** (`164.6–272.8 Hz`) | `+7.5 Hz` higher in sentence-initial phrase T2(b) |
| **Word Physical Energy ($L_2$)** | `24.7` | `25.2` | Statistically equivalent ($\Delta E = 0.5\text{ }L_2$) |

- **Scientific Conclusion**: Unlike English *"conduct"*, Telugu *"పట్టు"* is a true homophone across both meanings (/paʈʈu/). The Duration Predictor allocated **identically 20 mel frames ($0.232\text{ s}$)** in both sentences, with the geminate retroflex stop `ట్టు` occupying exactly $13$ frames in both cases. The $+7.5\text{ Hz}$ higher mean pitch in T2(b) is a positional declination effect (occurring at token index 4 after 1 word in T2b vs. token index 12 after 2 words in T2a). Thus, **these experiments do not demonstrate semantic disambiguation of contextual meanings**.

### 17.2 Typological & Architectural Comparison Across Languages
1. **Frontend Simplicity vs. Phonetic Precision**: Telugu's alpha-syllabic *akshara* script enabled a 101-token grapheme vocabulary with **zero OOV (`<unk>`) tokens** across all test sentences without requiring an external lexicon, whereas English required a 134,000-word CMUdict lexicon plus a neural `g2p_en` fallback to handle OOV compounds (`E1`), typos (`E3`), and code-mixed words (`E4`).
2. **Pause Modeling**: Telugu explicitly models inter-word boundaries via `@sp` (ID 4), producing distinct zero-energy valleys between words in `T1–T4` (Figures 26, 29, 32, 35, 38) and a higher standard deviation in physical energy ($\sigma_E \approx 20.6 - 29.4\text{ }L_2$). English models continuous connected speech with `sp` inserted only at punctuation (such as the 40-frame comma pause in `E4`), yielding smoother inter-word energy transitions ($\sigma_E \approx 13.5 - 17.2\text{ }L_2$).
3. **Speaker Conditioning & Pitch Baselines**: Across the test sets, English Speaker Profile `0` (female) exhibited a mean voiced pitch of $236.1\text{ Hz}$, whereas Telugu Speaker `spk_001` (male) exhibited a mean voiced pitch of $191.0\text{ Hz}$. *(Methodological Caveat: This $\approx 45\text{ Hz}$ difference reflects the empirical acoustic baselines of the two selected training speaker profiles; we do not claim that this observation proves biological causality or language-level pitch differences).*

---

## 18. Limitations and Acoustic Failure Analysis

### 18.1 Subjective AI-Assisted Evaluation (Telugu Test Set)
*Disclosure: In compliance with scientific reporting standards, the ratings below are explicitly documented as **AI-assisted subjective acoustic assessments** on a 1–5 scale, NOT human listener Mean Opinion Scores (MOS).*

| Sample ID | Clarity (1–5) | Fluency (1–5) | Naturalness (1–5) | Overall Mean | Primary Acoustic & Prosodic Observations |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **T1** | 4.0 | 3.5 | 3.0 | **3.50** | Accurate conjunct articulation; 20-grapheme loanword has a steady, slightly mechanical cadence. |
| **T2(a)** | 4.5 | 4.0 | 3.5 | **4.00** | Crisp geminate stop closure on `పట్టు`; natural phrasing; mildly flat declarative declination. |
| **T2(b)** | 2.5 | 2.5 | 2.0 | **2.33** | Clear opening phrase, but consonant boundary softening and lower coda energy on `ధరించి` & `వచ్చింది`. |
| **T3** | 3.5 | 3.0 | 2.5 | **3.00** | Clean sibilant/stop cluster `నిశ్శబ్దంగా`; low terminal energy ($E < 8.0$) softens final syllable `ది`. |
| **T4** | 2.0 | 2.0 | 1.5 | **1.83** | Single-token Telugu digit mode (`౨౦౨౬`, `౧౨వ`, `౧౦:౪౫`) compresses multi-syllable numbers into $3-11$f/digit. |
| **Mean** | **3.30** | **3.00** | **2.50** | **2.93** | *Honest AI-assisted diagnostic baseline across all five Telugu test sentences.* |

### 18.2 Root-Cause Failure Analysis
1. **Single-Token Digit Compression in Telugu (`T4`)**: In `telugu_digits` mode, each Telugu numeral `౦–౯` is represented as a single grapheme token. Because the Duration Predictor typically assigns $3-11$ mel frames per token, a 4-digit year (`2026` $\to$ `౨౦౨౬`) receives only $35$ frames ($0.406\text{ s}$), whereas its spoken realization (*"రెండు వేల ఇరవై ఆరు"*) spans 18 graphemes ($\approx 1.4\text{ s}$). This causes perceived rushing on numerical spans.
2. **Regex Colon/Hyphen Interaction in English (`E4`)**: In `English/text/normalizer.py`, expanding `"10:45 AM"` and `"2026"` interacted with hyphenated compound rules, slightly contracting the phoneme sequences for `"forty-five"` and `"twenty-six"`.
3. **Homograph Stress Ambiguity (`E2b`)**: Static dictionary lookup without syntactic POS tagging assigns noun stress (`AA1`) to verb uses of homographs like *"conduct"*.
4. **Universal Vocoder Timbre Smoothing**: Using the pretrained `generator_universal.pth.tar` HiFi-GAN without domain fine-tuning on Svarah or IndicTTS introduces mild high-frequency phase buzz during low-energy unvoiced-to-voiced transitions (`T2b`, `T3`).

---

## 19. Possible Improvements & Future Directions

1. **Full Cardinal/Ordinal Lexical Number Expansion for Telugu (`T4`)**: Extend `Telugu/text/normalizer.py` with a context-aware Telugu number-to-word transducer converting years (`2026` $\to$ `రెండు వేల ఇరవై ఆరు`), ordinals (`12వ` $\to$ `పన్నెండవ`), and times (`10:45` $\to$ `పది గంటల నలభై ఐదు నిమిషాలు`) into full grapheme sequences prior to tokenization.
2. **POS-Aware Homograph Frontend for English (`E2a/E2b`)**: Integrate a lightweight Part-of-Speech tagger (or neural G2P model conditioned on BERT contextual embeddings) to select verb-stressed /kənˈdʌkt/ (`K AH0 N D AH1 K T`) when *"conduct"* functions as a verb (`VB`).
3. **Pitch & Energy Continuous Wavelet / Flow Predictors**: Replace deterministic $L_2$-regressed 256-bin pitch quantization with a Continuous Wavelet Transform (CWT) pitch spectrogram or conditional flow-matching variance adaptor to eliminate post-13k pitch over-smoothing.
4. **Corpus-Specific HiFi-GAN Fine-Tuning**: Fine-tune the HiFi-GAN generator and multi-period/multi-scale discriminators for 50k steps on ground-truth Svarah and IndicTTS mel-spectrograms to sharpen consonant bursts in `వచ్చింది` (`T2b`) and low-energy codas (`T3`).

---

## 20. Reproducibility, Dependencies & Execution Guide

### 20.1 Centralized Path Management & Environment Setup
All scripts and modules resolve paths dynamically against `PROJECT_ROOT` via [`common/paths.py`](../common/paths.py) without relying on `os.getcwd()`. Small inference metadata files (`preprocessed_data/{Svarah,IndicTTS}/{stats.json, speakers.json}`, $2,790\text{ bytes}$ total) are tracked directly in Git so model instantiation and inference work immediately upon downloading the checkpoints.

```bash
# 1. Create and activate Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the complete 252-test regression suite across all 23 test modules
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

### 20.2 Preprocessing, Training & Inference Commands

```bash
# --- ENGLISH (SVARAH) PIPELINE ---
# 1. Preprocess Svarah corpus and extract MFA durations + mel/pitch/energy features
.venv/bin/python -m English.preprocess -p config/Svarah/preprocess.yaml

# 2. Train English FastSpeech 2 (20,000 steps, batch_size=16, val/save every 1,000 steps)
.venv/bin/python -m English.train \
  -p config/Svarah/preprocess.yaml -m config/Svarah/model.yaml -t config/Svarah/train.yaml \
  --preprocessed_dir preprocessed_data/Svarah \
  --split_dir English/metadata/splits/profile_stratified_v1 \
  --checkpoint_dir checkpoints/Svarah_profile_stratified_v1 \
  --log_dir output/logs/Svarah_profile_stratified_v1 --total_step 20000

# 3. Synthesize English assignment samples (E1, E2a, E2b, E3, E4) using Best Checkpoint (Step 19,000)
.venv/bin/python -m English.inference \
  --mode assignment --speaker_id 0 \
  --checkpoint checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar \
  -p config/Svarah/preprocess.yaml -m config/Svarah/model.yaml -t config/Svarah/train.yaml \
  --vocoder_weights original_fastspeech2/hifigan/generator_universal.pth.tar \
  --output_dir Results/English/test_generated_audios

# --- TELUGU (INDICTTS) PIPELINE ---
# 1. Preprocess IndicTTS Telugu corpus (NFC + Latin-o repair + @sp pause alignment)
.venv/bin/python -m Telugu.preprocess -p config/IndicTTS/preprocess.yaml

# 2. Train Telugu FastSpeech 2 (20,000 steps, batch_size=16, val/save every 1,000 steps)
.venv/bin/python -m Telugu.train \
  -p config/IndicTTS/preprocess.yaml -m config/IndicTTS/model.yaml -t config/IndicTTS/train.yaml \
  --preprocessed_dir preprocessed_data/IndicTTS \
  --checkpoint_dir checkpoints/IndicTTS --log_dir output/log/IndicTTS --total_step 20000

# 3. Synthesize Telugu assignment samples (T1, T2_a, T2_b, T3, T4) using Best Checkpoint (Step 13,000)
.venv/bin/python -m Telugu.inference \
  --mode assignment --samples_file Results/Telugu/official_samples.json \
  --speaker spk_001 --digit_mode telugu_digits \
  --checkpoint checkpoints/IndicTTS/checkpoint_best.pth.tar \
  -p config/IndicTTS/preprocess.yaml -m config/IndicTTS/model.yaml -t config/IndicTTS/train.yaml \
  --vocoder_weights original_fastspeech2/hifigan/generator_universal.pth.tar \
  --output_dir Results/Telugu/test_generated_audios \
  --plots_dir Results/Telugu/variance_adaptor_plots
```

---

## 21. Assignment Compliance Matrix

Table 9 maps every requirement from `FastSpeech2_Assignment(4).pdf` to its corresponding repository artifact, report section, and PDF page reference.

| Assignment Requirement (`FastSpeech2_Assignment.pdf`) | Repository Deliverable / Evidence | Report Section | PDF Page(s) | Status |
| :--- | :--- | :--- | :---: | :---: |
| **1. Separate English & Telugu FastSpeech 2 implementations** | `English/`, `Telugu/`, `common/`, `config/{Svarah,IndicTTS}/` | Sections 5, 6, 7, 10 | pp. 3–8 | **COMPLETE** |
| **2. Document changes made to reference GitHub repo (esp. Telugu)** | Table 1 & Section 6.1 (`101`-token graphemes, NFC, Latin-$o \to$ `ం`, `@sp` ID 4, mask-aware PostNet) | Section 6 (Table 1 & §6.1) | pp. 4–5 | **COMPLETE** |
| **3. Datasets: ~5h train & ~1h val for Svarah (En) and IndicTTS (Te)** | En: `3,208` train (`4.999h`) / `664` val (`1.000h`); Te: `2,300` train (`5.000h`) / `421` val (`1.000h`) | Section 8 (Table 2) | pp. 6–7 | **COMPLETE** |
| **4. Specify data selection, subsampling & zero-overlap verification** | `profile_stratified_v1` (88 profiles) & `option_1b` (1,053 F / 1,247 M train; 192 F / 229 M val; 0 overlap) | Sections 8.1, 8.2, 9 | pp. 6–7 | **COMPLETE** |
| **5. Trained checkpoints & training/validation loss curves** | En Best Step 19k (`2.3489`), Te Best Step 13k (`2.0869`), Figures 3–8 (`Results/*/training_loss.png`, `validation_loss.png`) | Sections 1, 11, 12 (Table 5, Figs 3–8) | pp. 1, 8–10 | **COMPLETE** |
| **6. Generate speech for all 10 test sentences (`E1–E4`, `T1–T4`)** | `Results/English/test_generated_audios/E*.wav`, `Results/Telugu/test_generated_audios/T*.wav` | Sections 13, 14, 15 (Tables 6 & 7) | pp. 10–14 | **COMPLETE** |
| **7. Observations on pronunciation & speech quality for each sentence** | Individual sample evaluations for all 10 utterances (`E1..E4`, `T1..T4`) + Table 8 | Sections 14.2, 15.2, 18.1 | pp. 11–14, 25–26 | **COMPLETE** |
| **8. 30 Variance Adaptor plots (duration, pitch, energy) + explanations** | `Results/{English,Telugu}/variance_adaptor_plots/{duration,pitch,energy}/*.png` + analytical text | Section 16 (§16.1–§16.10, Figs 9–38) | pp. 15–24 | **COMPLETE** |
| **9. Comparative observations (`conduct` & `పట్టు`), limitations & improvements** | Homograph tables (`E2a/E2b`, `T2a/T2b`), failure analysis & 4 concrete improvements | Sections 17, 18, 19 | pp. 25–26 | **COMPLETE** |
| **10. Reproducibility details, dependencies, configs & CLI commands** | `README.md`, `requirements.txt`, `common/paths.py`, `252/252` unit tests, Section 20 | Section 20 | pp. 26–27 | **COMPLETE** |

---

## 22. Conclusion

This project delivered a rigorous, fully reproducible dual-language implementation of **FastSpeech 2** for English (`Svarah`) and Telugu (`IndicTTS`). By pairing stratified ~5-hour training and ~1-hour validation splits with Montreal Forced Alignment, frame-level pitch/energy conditioning, mask-aware PostNet refinement, and a custom 101-token Telugu grapheme frontend, both models converged smoothly over 20,000 steps. Selecting the best observed validation checkpoints at **Step 19,000** for English ($2.34890$) and **Step 13,000** for Telugu ($2.08688$) produced intelligible, prosodically expressive 22,050 Hz speech across all ten canonical test sentences and all 30 variance adaptor plots, while empirical homograph analysis of *"conduct"* and *"పట్టు"* rigorously established the boundary between acoustic self-attention modulation and true semantic disambiguation.

---

## 23. References

1. **Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T. Y.** (2020). *FastSpeech 2: Fast and high-quality end-to-end text to speech*. In International Conference on Learning Representations (ICLR 2021), arXiv:2006.04558.
2. **Ren, Y., Ruan, Y., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T. Y.** (2019). *FastSpeech: Fast, robust and controllable text to speech*. In Advances in Neural Information Processing Systems (NeurIPS), 32.
3. **Kong, J., Kim, J., & Bae, J.** (2020). *HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis*. In Advances in Neural Information Processing Systems (NeurIPS), 33, 17022–17033.
4. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). *Attention is all you need*. In Advances in Neural Information Processing Systems (NeurIPS), 30.
5. **Baby, A., Thomas, A. L., Nishanthi, N. L., & Murthy, H. A.** (2016). *Resources for Indian languages: IndicTTS database for text-to-speech synthesis*. In Proc. Workshop on Speech and Language Processing for Dravidian Languages (pp. 37–43).
6. **McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M.** (2017). *Montreal Forced Aligner: Trainable text-speech alignment using Kaldi*. In Proc. Interspeech 2017 (pp. 498–502).
7. **Morise, M., Yokomori, F., & Ozawa, K.** (2016). *WORLD: A vocoder-based high-quality speech synthesis system for real-time applications*. IEICE Transactions on Information and Systems, E99-D(7), 1877–1884.
