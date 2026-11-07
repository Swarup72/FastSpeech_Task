# FastSpeech 2 Implementation Decision Register (`IMPLEMENTATION_DECISIONS.md`)

**Project Root**: `/Users/swaruppal/FastSpeech2_Project`  
**Shared Implementation Directory**: `/Users/swaruppal/FastSpeech2_Project/common`  
**Last Updated**: Stage 2, Step 1.3G (Architecture Review & Specification)

This register tracks all architectural, bug-fix, configuration, and portability decisions for the shared `common/` FastSpeech 2 implementation across the English (Svarah) and Telugu (IndicTTS) TTS pipelines.

---

## 1. Active Architectural & Implementation Decisions

| Decision ID | Component | Category | Decision Summary | Original Reference (`original_fastspeech2/`) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DEC-01** | `utils.tools` & `LengthRegulator` | **Fix for Correctness** | Remove module-global `device` variables. Derive tensor device dynamically from inputs (`lengths.device` in `get_mask_from_lengths`, `x.device` in `LengthRegulator.LR`) so `cpu`, `mps:0`, and `cuda` work identically. | `utils/tools.py:15, 96`; `model/modules.py:14, 180` | **Approved (Ready for Implementation)** |
| **DEC-02** | `VarianceAdaptor` | **Fix for Correctness** | Pass `e_control` instead of `p_control` to `self.get_energy_embedding(...)` in both `phoneme_level` and `frame_level` branches. | `model/modules.py:124, 146` | **Approved (Ready for Implementation)** |
| **DEC-03** | `VarianceAdaptor` | **Fix for Correctness** | Multiply `d_control` *inside* `torch.round(...)` and cast `duration_rounded` to `torch.long`: `torch.clamp(torch.round((torch.exp(log_d) - 1) * d_control), min=0).long()`. Prevents `int()` floor truncation of 1-frame phonemes when `d_control < 1.0` and unifies `duration_rounded` dtype (`int64`). | `model/modules.py:132–135, 187` | **Approved (Ready for Implementation)** |
| **DEC-04** | `VarianceAdaptor` | **Fix for Correctness** | Validate `pitch_min > 0` / `energy_min > 0` when `"log"` quantization is configured, raising `ValueError` instead of silently producing `NaN` bin boundaries. Default quantization to `"linear"`. | `model/modules.py:51, 63` | **Approved (Ready for Implementation)** |
| **DEC-05** | `FastSpeech2` & `PostNet` | **Make Configurable / Fix** | Pass `n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"]` (and optional `model_config.get("postnet", {})` overrides) to `PostNet(...)` in `FastSpeech2.__init__`. | `model/fastspeech2.py:27`; `transformer/Layers.py:72–78` | **Approved (Ready for Implementation)** |
| **DEC-06** | `Encoder` & `FastSpeech2` | **Make Configurable** | Decouple `Encoder` from global `from text.symbols import symbols` by accepting `n_src_vocab` via constructor argument or `model_config["n_src_vocab"]`, allowing distinct vocabulary sizes for English and Telugu. | `transformer/Models.py:7, 40` | **Approved (Ready for Implementation)** |
| **DEC-07** | `MultiHeadAttention` & `FFTBlock` | **Fix for Correctness** | Guard `if mask is not None:` before `mask.repeat(n_head, 1, 1)` in `MultiHeadAttention.forward` and before `enc_output.masked_fill(mask.unsqueeze(-1), 0)` in `FFTBlock.forward`. | `transformer/SubLayers.py:46`; `transformer/Layers.py:25, 28` | **Approved (Ready for Implementation)** |
| **DEC-08** | `ScaledDotProductAttention` | **Fix for Correctness** | Guard against all-masked key rows (`mask.all(dim=-1, keepdim=True)`) producing `0/0 = NaN` in `Softmax(dim=2)` by zeroing out `NaN` attention weights (`torch.nan_to_num(attn, nan=0.0)`) and rejecting zero-length input sequences (`src_lens <= 0`) at model entry. | `transformer/Modules.py:20–22` | **Approved (Ready for Implementation)** |
| **DEC-09** | `Encoder` & `Decoder` | **Fix for Correctness** | Return `(enc_output, enc_slf_attn_list)` from `Encoder.forward` and `(dec_output, mask, dec_slf_attn_list)` from `Decoder.forward` when `return_attns=True`, while preserving single/2-tuple returns when `return_attns=False`. | `transformer/Models.py:73, 97–100, 139, 168–171` | **Approved (Ready for Implementation)** |
| **DEC-10** | `FastSpeech2Loss` | **Fix for Correctness** | When `pitch_feature_level == "frame_level"` or `energy_feature_level == "frame_level"`, slice `pitch_predictions`, `pitch_targets`, `energy_predictions`, and `energy_targets` along dim 1 to `:mel_masks.shape[1]` to match Decoder-truncated `mel_masks`. Also raise `ValueError` on empty masked selections. | `model/loss.py:43–64` | **Approved (Ready for Implementation)** |
| **DEC-11** | `VariancePredictor` | **Fix for Correctness** | Use `padding=(self.kernel - 1) // 2` in both `conv1d_1` and `conv1d_2` (replacing hardcoded `padding=1` in `conv1d_2`). | `model/modules.py:230` | **Approved (Ready for Implementation)** |
| **DEC-12** | `FastSpeech2` & `VarianceAdaptor` | **Make Configurable** | Allow `stats` dict (`{"pitch": [...], "energy": [...]}`) and `n_speaker` (`int`) to be passed directly in constructor args or `model_config` for in-memory instantiation/testing, falling back to `<preprocessed_path>/stats.json` and `speakers.json` when omitted. | `model/modules.py:41–46`; `model/fastspeech2.py:31–37` | **Approved (Ready for Implementation)** |
| **DEC-13** | `common/utils/tools.py` | **Portability / Hygiene** | Keep core tensor utilities (`get_mask_from_lengths`, `pad`, `expand`, `to_device`) free of top-level `matplotlib` imports so `common.model` and `common.transformer` can be imported in lightweight/headless environments. | `utils/tools.py:4–5` | **Approved (Ready for Implementation)** |
| **DEC-14** | All `common/` Modules | **Preserve Unchanged** | Preserve exact submodule hierarchy, `state_dict` parameter names, Post-LN residual connections, interleaved `FFTBlock` padding zeroing, sequential pitch-then-energy conditioning, and 5-term unweighted loss structure (`mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss`). | All `model/` and `transformer/` files | **Approved (Baseline Invariant)** |

---

## 2. Deferred / Pending Decisions

| Decision ID | Component | Question / Trade-off | Current Default | Trigger to Revisit |
| :--- | :--- | :--- | :--- | :--- |
| **DEF-01** | `PostNet.forward` | Should `PostNet` zero out padded frames (`mel_masks`) between its 5 `Conv1d + BatchNorm1d` layers? | Accept optional `mask=None` in `PostNet.forward(x, mask=None)`, defaulting to `None` (unmasked, matching original repo). | Revisit if batch validation metrics show sensitivity to batch padding length. |
| **DEF-02** | `LengthRegulator.expand` | How to handle an inference utterance where all predicted token durations round to `0` (`sum(d_rounded) == 0`)? | If `expanded.shape[0] == 0`, fallback to 1 zero-frame (`torch.zeros(1, H, device=batch.device)`) so downstream `Conv1d` layers do not crash. | Revisit during inference/synthesis script testing. |
| **DEF-03** | `English/` & `Telugu/` Configs | Exact `n_src_vocab` and `multi_speaker` settings for Svarah (English) and IndicTTS (Telugu). | Configured per-language in `English/config/model.yaml` and `Telugu/config/model.yaml`. | Finalize during dataset & text frontend stage. |

