# Telugu Dataset Split and Selection Re-Audit (`SPRINGLab/IndicTTS_Telugu`)

**Audit Date:** 2026-11-05  
**Dataset:** `SPRINGLab/IndicTTS_Telugu` (8,576 source utterances, 36.7122 decoded hours)  
**Target Files Audited (Read-Only — No Files Modified):**
- [`Telugu/metadata/partition_dataset.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/partition_dataset.py)
- [`Telugu/metadata/split_summary.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/split_summary.json)
- [`Telugu/metadata/text_manifest.json`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/text_manifest.json)
- [`Telugu/metadata/train.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/train.txt), [`Telugu/metadata/val.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/val.txt), [`Telugu/metadata/test.txt`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/test.txt)
- Local Parquet shards: `data/telugu_parquet/data/train-00000-of-00004.parquet` .. `train-00003-of-00004.parquet`

---

## Executive Summary

Your concern that the current training set (`2,330` utterances) is smaller in utterance count than the test set (`2,366` utterances) is **completely valid and exposes two structural flaws in how `partition_dataset.py` constructed the current split**:

1. **Unbounded Remainder Dumping into `test.txt` (Causing a 45.2% / 8.9% / 45.9% Split and 92.6% Male Bias in Test)**:
   - After filtering to `5,153` clean utterances (`11.9009` actual hours), the clean pool is heavily Male-skewed (`3,813` Male utterances [`8.4995h`] vs. `1,340` Female utterances [`3.4014h`]) because the Female speaker recorded long multi-sentence story paragraphs (mean `21.30s`) while the Male speaker recorded shorter sentences (mean `10.16s`).
   - [`partition_dataset.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/partition_dataset.py#L134-L144) selected fixed quotas for `train` (`~2.50` est hours/speaker) and `val` (`~0.50` est hours/speaker), and **dumped all remaining clean utterances (`female_rem + male_rem`) into `test.txt`**.
   - Because `2,190` Male utterances (`4.9014h`) and only `176` Female utterances (`0.4454h`) were left over, `test.txt` ballooned to **`2,366` utterances (`5.3468h` — 45.9% of the clean pool)** and became **92.56% Male by utterance count (`2,190 / 2,366`)** and **91.67% Male by duration (`4.9014h / 5.3468h`)**.

2. **Character-Rate Estimated Filtering Instead of Actual Decoded Audio Duration Filtering (Allowing 148 Utterances $> 15\text{s}$ Up to `42.70s` into the Current Splits)**:
   - Because [`partition_dataset.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/partition_dataset.py#L51-L55) was executed in Step 2.2A *before* the 4 Parquet audio shards were downloaded in Step 2.2A-R, it applied the $[1.0\text{s}, 15.0\text{s}]$ filter to **character-rate estimated durations** (`len(norm_text) / rate`) rather than actual decoded WAV durations.
   - Cross-checking all 8,576 WAV headers in the Parquet shards reveals that **148 utterances inside the current `train.txt`, `val.txt`, and `test.txt` actually exceed `15.0s` of real audio** — including a **`42.70s` utterance in `train.txt`** (`66` utterances $> 15\text{s}$, `7` utterances $> 20\text{s}$) and a **`34.33s` utterance in `val.txt`** (`16` utterances $> 15\text{s}$). A `42.70s` utterance (`~3,678` mel frames) poses a severe $O(T^2)$ self-attention CUDA OOM risk during training.
   - Conversely, **149 clean utterances** (`133` Female, `16` Male) that were excluded by character-rate estimation actually have **true decoded durations strictly within $[1.0\text{s}, 15.0\text{s}]$**. Filtering by **true decoded WAV duration** increases the clean $[1.0\text{s}, 15.0\text{s}]$ Female pool from `1,340` utterances (`3.4014h`) to **`1,435` utterances (`3.6644h`)** while eliminating every single $> 15.0\text{s}$ outlier.

---

## 1 & 2. Why Were 3,423 Utterances Excluded, and What Exact Quality Criteria Caused Their Exclusion?

In [`Telugu/metadata/partition_dataset.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/partition_dataset.py#L65-L111), each of the `8,576` source records (`36.7122` actual hours) was evaluated against four sequential rules. In total, **`3,423` utterances (`24.8113` actual hours)** were excluded, leaving **`5,153` utterances (`11.9009` actual hours)**:

| Exclusion Reason | Code Rule in `partition_dataset.py` | Total Excluded (Utts) | Actual Duration (Hours) | Est. Duration (Hours) | Female Excluded (Utts / Actual h) | Male Excluded (Utts / Actual h) | True Decoded Duration Breakdown of Excluded Pool |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`too_long_super_15s`** | `est_dur > 15.0s` (`len(norm_text)/rate > 15.0`) | **3,199** | **24.6113 h** | 25.4462 h | 2,585 (`20.3978h`) | 614 (`4.2136h`) | `3,071` are truly $> 15.0\text{s}$ (up to `151.60s`); **`128` (`120` F, `8` M) are actually $\le 15.0\text{s}$!** |
| **`cross_gender_duplicate_title`** | Identical `norm_text` spoken by both Female (`0`) and Male (`1`) | **198** | **0.1872 h** | 0.1629 h | 101 (`0.1131h`) | 97 (`0.0742h`) | 90 shared story headings (e.g., *"భువన సుందరి."*, *"నావికుడు సింద్ బాద్."*) quarantined to prevent cross-split transcript leakage |
| **`too_short_sub_1s`** | `est_dur < 1.0s` (`len(norm_text)/rate < 1.0`) | **21** | **0.0099 h** | 0.0048 h | 13 (`0.0063h`) | 8 (`0.0035h`) | All `21` (`13` F, `8` M) actually have decoded durations in $[1.0\text{s}, 15.0\text{s}]$ due to leading/trailing breath pauses |
| **`intra_gender_duplicate_transcript`** | Duplicate `norm_text` within the same speaker (1st kept, rest dropped) | **5** | **0.0029 h** | 0.0018 h | 4 (`0.0024h`) | 1 (`0.0005h`) | Exact duplicate transcripts within a single speaker |
| **Total Excluded** | — | **3,423** | **24.8113 h** | **25.6157 h** | **2,703 (`20.5196h`)** | **720 (`4.2918h`)** | — |
| **Selected Clean Pool (`none`)** | `1.0s <= est_dur <= 15.0s` & unique `norm_text` | **5,153** | **11.9009 h** | **10.5481 h** | **1,340 (`3.4014h`)** | **3,813 (`8.4995h`)** | `5,004` are truly in $[1.0\text{s}, 15.0\text{s}]$; **`148` are actually $> 15.0\text{s}$ (up to `42.70s`)**; `1` is `< 1.0s` (`0.95s`) |
| **Full Source Corpus** | — | **8,576** | **36.7122 h** | **36.1638 h** | **4,043 (`23.9210h`)** | **4,533 (`12.7912h`)** | — |

### Why 64% of Female Utterances Were Excluded (`2,585 / 4,043`)
- In `SPRINGLab/IndicTTS_Telugu`, the Female speaker recorded **unsegmented multi-sentence folktale paragraphs** rather than isolated single sentences:
  - **All Female utterances (`n = 4,043`, `23.9210h`)**: min `0.81s`, 10th percentile `9.05s`, median **`19.50s`**, mean **`21.30s`**, 90th percentile **`34.81s`**, max **`151.60s`**.
  - **All Male utterances (`n = 4,533`, `12.7912h`)**: min `0.62s`, 10th percentile `4.73s`, median **`9.59s`**, mean **`10.16s`**, 90th percentile **`16.12s`**, max **`78.00s`**.
- Because over half of the Female corpus exceeds `19.50s` per recording, applying a `15.0s` upper bound removes `64%` of Female utterances (`20.40h` of `23.92h`) while removing only `13.5%` of Male utterances (`4.21h` of `12.79h`).

### Critical Finding: Estimated vs. Actual Duration Mismatch in the Current Split
Inspecting the actual decoded WAV durations inside the current `train.txt`, `val.txt`, and `test.txt` reveals:
- **`train.txt` (`2,330` utts)**: min `1.28s`, median `8.23s`, mean `8.43s`, 95th percentile `13.97s`, **max `42.70s`** — contains **`66` utterances $> 15.0\text{s}$** and **`7` utterances $> 20.0\text{s}$**!
- **`val.txt` (`457` utts)**: min `1.41s`, median `8.47s`, mean `8.65s`, 95th percentile `14.27s`, **max `34.33s`** — contains **`16` utterances $> 15.0\text{s}$** and **`2` utterances $> 20.0\text{s}$**!
- **`test.txt` (`2,366` utts)**: min `0.95s`, median `7.92s`, mean `8.14s`, 95th percentile `14.09s`, **max `20.28s`** — contains **`66` utterances $> 15.0\text{s}$**, `1` utterance $> 20.0\text{s}$, and `1` utterance `< 1.0s` (`0.95s`).

---

## 3. Why Does the Test Set Contain More Utterances (`2,366`) Than the Training Set (`2,330`)?

Look at lines 134–144 of [`Telugu/metadata/partition_dataset.py`](file:///Users/swaruppal/FastSpeech2_Project/Telugu/metadata/partition_dataset.py#L134-L144):
```python
# Female split
female_train, female_rem = select_split(female_pool, TARGET_TRAIN_HOURS_PER_SPEAKER * 3600)
female_val, female_test = select_split(female_rem, TARGET_VAL_HOURS_PER_SPEAKER * 3600)

# Male split
male_train, male_rem = select_split(male_pool, TARGET_TRAIN_HOURS_PER_SPEAKER * 3600)
male_val, male_test = select_split(male_rem, TARGET_VAL_HOURS_PER_SPEAKER * 3600)
```
1. `select_split` greedily accumulated `2.50` estimated hours of Female (`972` utts, `2.4548h` actual) and `2.50` estimated hours of Male (`1,358` utts, `3.0010h` actual) for `train.txt` (`2,330` utts total).
2. Next, it accumulated `0.50` estimated hours of Female (`192` utts, `0.5013h` actual) and `0.50` estimated hours of Male (`265` utts, `0.5971h` actual) for `val.txt` (`457` utts total).
3. Finally, **it placed the entire unallocated remainder (`female_test` + `male_test`) into `test.txt` without a quota or cap**.
4. Because the Male clean pool had `3,813` utterances (`8.4995h` actual) and shorter average utterance durations (`7.95s` vs. Female `9.14s`), `3,813 - 1,358 - 265 = 2,190` Male utterances (`4.9014h`) were left over and dumped into `test.txt` alongside the `176` remaining Female utterances (`0.4454h`).
5. Consequently, `test.txt` received **`2,366` utterances (`45.91%` of all clean utterances)** vs. `train.txt`'s **`2,330` utterances (`45.22%`)**.

---

## 4. What Strategy Was Used to Create the Split?

- **Strategy**: **Speaker-stratified, text-deduplicated pseudo-random shuffling (`random.Random(42)`) followed by fixed-hour quota greedy accumulation** for Train and Validation, with an **unbounded remainder assignment** for Test.
- It was **not** split sequentially by source ID or by story/chapter boundary: after deduplication and duration filtering, `female_pool` and `male_pool` were shuffled across all source IDs using `seed=42`.

---

## 5. Is There Any Speaker, Recording, or Transcript Leakage Between Splits?

| Leakage Dimension | Finding | Audit Status |
| :--- | :--- | :---: |
| **Source ID / Recording Overlap** | `0` shared utterance IDs or WAV files between `train.txt` (`2,330`), `val.txt` (`457`), and `test.txt` (`2,366`). Total = `5,153` unique recordings. | **PASS (Zero Overlap)** |
| **Exact Normalized Transcript Overlap** | `0` shared transcripts across `train ∩ val`, `train ∩ test`, and `val ∩ test`. All `90` cross-speaker story titles (`198` rows) and `5` intra-speaker duplicate texts were removed before splitting. | **PASS (Zero Overlap)** |
| **Speaker Overlap (Multi-Speaker Closed-Set)** | `SPRINGLab/IndicTTS_Telugu` contains **only 2 speakers total** (`speaker 0` = 1 Female voice artist at 16 kHz; `speaker 1` = 1 Male voice artist at 48 kHz). Because FastSpeech 2 uses a closed-set lookup table `speaker_emb = nn.Embedding(2, 256)`, both speakers **must** appear in Train, Val, and Test. | **EXPECTED (2-Speaker Corpus)** |
| **Narrative / Story Context Overlap** | Because the corpus recordings are sequential paragraphs from folktales (e.g., `train_telugufemale_00001..00015` narrate the story of *Mitravindaka*) and shuffling was performed at the **utterance level**, different sentences from the same story appear across Train, Val, and Test (though with zero text overlap and consistent studio acoustics per speaker). | **BENIGN (Standard Single-Studio Corpus)** |

---

## 6. Is the Test Set Disproportionately Male-Dominated?

**Yes — severely.** Below is the exact gender breakdown of the current split:

| Split | Total Utterances (% of Clean Pool) | Total Actual Hours (% of Clean Pool) | Female Utterances (% of Split) | Female Actual Hours (% of Split) | Male Utterances (% of Split) | Male Actual Hours (% of Split) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Current `train.txt`** | 2,330 (**45.22%**) | 5.4557 h (**45.84%**) | 972 (41.72%) | 2.4548 h (44.99%) | 1,358 (58.28%) | 3.0010 h (55.01%) |
| **Current `val.txt`** | 457 (**8.87%**) | 1.0984 h (**9.23%**) | 192 (42.01%) | 0.5013 h (45.64%) | 265 (57.99%) | 0.5971 h (54.36%) |
| **Current `test.txt`** | **2,366 (45.91%)** | **5.3468 h (44.93%)** | **176 (7.44%)** | **0.4454 h (8.33%)** | **2,190 (92.56%)** | **4.9014 h (91.67%)** |
| **Total Clean Pool** | **5,153 (100.0%)** | **11.9009 h (100.0%)** | **1,340 (26.00%)** | **3.4014 h (28.58%)** | **3,813 (74.00%)** | **8.4995 h (71.42%)** |

Because `test.txt` is **92.56% Male by utterance count** (`91.67%` by duration), any unweighted evaluation metric on `test.txt` measures almost exclusively the Male voice and masks synthesis quality on the Female voice.

---

## 7 & 8. Can We Create a More Balanced Split, and Would Using More of the 36.71-Hour Corpus Help or Hurt?

### Why We Should NOT Include Utterances $> 18\text{s}$–$20\text{s}$ from the 36.71-Hour Corpus
 Let us look at the exact deduplicated clean pool available at different **true decoded audio duration** thresholds across all 8,576 recordings:

| True Decoded Audio Duration Window | Total Clean Utterances | Total Actual Hours | Clean Female Utterances (Hours) | Clean Male Utterances (Hours) | Max Mel Frames (`sr=22050, hop=256`) | MFA Alignment & GPU Training Risk |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$[1.0\text{s}, 12.0\text{s}]$** | 4,426 | 8.9963 h | 1,046 (`2.2101h`) | 3,380 (`6.7862h`) | 1,033 frames | Very safe, but Female pool drops to `2.21h` (too small for `2.5h` Female Train) |
| **$[1.0\text{s}, 15.0\text{s}]$ (Standard)** | **5,153** | **11.6975 h** | **1,435 (`3.6644h`)** | **3,718 (`8.0331h`)** | **1,292 frames** | **Optimal safety**: matches English Svarah max length, zero OOM risk on 16 GB Colab T4, high IndicMFA accuracy, and **adds +95 clean Female utterances (`+0.263h`)** over the estimated-duration filter |
| **$[1.0\text{s}, 18.0\text{s}]$ (Moderately Relaxed)** | **5,757** | **14.4482 h** | **1,844 (`5.5306h`)** | **3,913 (`8.9176h`)** | **1,550 frames** | **Feasible if more hours are desired**: increases attention matrix memory by $1.44\times$ over 15s (`1,550²` vs `1,292²`), still fits on 16 GB T4 with `batch_size=16`, boosts Female pool by `+51%` (`5.53h`) |
| **$[1.0\text{s}, 20.0\text{s}]$** | 6,136 | 16.4487 h | 2,118 (`6.9759h`) | 4,018 (`9.4727h`) | 1,722 frames | Borderline: $1.78\times$ attention memory; multi-clause paragraphs start hurting MFA beam search |
| **$[1.0\text{s}, 25.0\text{s}]$** | 6,854 | 20.9131 h | 2,644 (`10.2542h`) | 4,210 (`10.6589h`) | 2,153 frames | High risk: $2.78\times$ attention memory; frequent MFA alignment dropouts on multi-sentence paragraphs |
| **Unfiltered $[1.0\text{s}, 151.6\text{s}]$** | 8,372 | 36.5219 h | 3,937 (`23.8052h`) | 4,435 (`12.7166h`) | **13,058 frames** | **Unusable without audio/text re-segmentation**: $102\times$ attention memory (`13,058²`), guaranteed CUDA OOM and MFA failure |

---

## 9. Quantitative Comparison: Existing Split vs. 70–80% Train / 10–15% Val / 10–15% Test Alternatives

Below is a side-by-side comparison of the **Existing Split** against three clean alternatives filtered by **true decoded WAV duration** (zero $> 15\text{s}$ outliers in Options 1 & 2; zero $> 18\text{s}$ outliers in Option 3) that follow the **70–80% Train / 10–15% Validation / 10–15% Test** proportion:

| Metric / Criterion | Existing Split (`est_dur <= 15s`, Unbounded Test) | **Option 1 (RECOMMENDED): True $[1\text{s}, 15\text{s}]$ Speaker-Balanced 75% / 12.5% / 12.5%** | **Option 1B: True $[1\text{s}, 15\text{s}]$ Speaker-Balanced ~73% / 13.5% / 13.5% (Exact 1.0h Val)** | **Option 2: True $[1\text{s}, 15\text{s}]$ Full-Pool 75% / 12.5% / 12.5% (Unbalanced Gender)** | **Option 3: True $[1\text{s}, 18\text{s}]$ Speaker-Balanced 75% / 12.5% / 12.5%** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Duration Filter Basis** | Character-rate estimate (`148` utts $>15\text{s}$, max **`42.70s`**) | **True WAV duration $\in [1.0\text{s}, 15.0\text{s}]$** (`0` utts $>15\text{s}$) | **True WAV duration $\in [1.0\text{s}, 15.0\text{s}]$** (`0` utts $>15\text{s}$) | **True WAV duration $\in [1.0\text{s}, 15.0\text{s}]$** (`0` utts $>15\text{s}$) | **True WAV duration $\in [1.0\text{s}, 18.0\text{s}]$** (`0` utts $>18\text{s}$) |
| **Active Split Pool** | 5,153 utts (`11.9009h`) | **3,148 utts (`7.3285h`)** (+ `2,005` Male reserve [`4.3690h`]) | **3,146 utts (`7.3288h`)** (+ `2,007` Male reserve [`4.3687h`]) | **5,153 utts (`11.6975h`)** (`0` reserve) | **4,282 utts (`11.0611h`)** (+ `1,475` Male reserve [`3.3872h`]) |
| **Split Ratio (Train / Val / Test)** | 45.8% / 9.2% / **44.9%** (Inverted!) | **75.0% / 12.5% / 12.5%** | **72.7% / 13.65% / 13.65%** | **75.0% / 12.5% / 12.5%** | **75.0% / 12.5% / 12.5%** |
| **Train Total** | 2,330 utts (`5.4557h`) | **2,374 utts (`5.4965h`)** | **2,302 utts (`5.3288h`)** | **3,886 utts (`8.7716h`)** | **3,204 utts (`8.2955h`)** |
| — *Train Female / Male* | 972 (`2.45h`) / 1,358 (`3.00h`) [45% / 55%] | **1,088 (`2.75h`) / 1,286 (`2.75h`) [50.0% / 50.0%]** | **1,055 (`2.66h`) / 1,247 (`2.66h`) [50.0% / 50.0%]** | 1,085 (`2.75h`) / 2,801 (`6.02h`) [31.3% / 68.7%] | **1,374 (`4.15h`) / 1,830 (`4.15h`) [50.0% / 50.0%]** |
| **Validation Total** | 457 utts (`1.0984h`) | **387 utts (`0.9160h`)** | **422 utts (`1.0000h`)** | **640 utts (`1.4635h`)** | **539 utts (`1.3826h`)** |
| — *Val Female / Male* | 192 (`0.50h`) / 265 (`0.60h`) [46% / 54%] | **176 (`0.46h`) / 211 (`0.46h`) [50.0% / 50.0%]** | **192 (`0.50h`) / 230 (`0.50h`) [50.0% / 50.0%]** | 179 (`0.46h`) / 461 (`1.00h`) [31.4% / 68.6%] | **235 (`0.69h`) / 304 (`0.69h`) [50.0% / 50.0%]** |
| **Test Total** | **2,366 utts (`5.3468h`)** | **387 utts (`0.9160h`)** | **422 utts (`1.0000h`)** | **627 utts (`1.4625h`)** | **539 utts (`1.3830h`)** |
| — *Test Female / Male* | **176 (`0.45h`) / 2,190 (`4.90h`) [8.3% / 91.7%!]** | **171 (`0.46h`) / 216 (`0.46h`) [50.0% / 50.0%]** | **188 (`0.50h`) / 234 (`0.50h`) [50.0% / 50.0%]** | 171 (`0.46h`) / 456 (`1.00h`) [31.3% / 68.7%] | **235 (`0.69h`) / 304 (`0.69h`) [50.0% / 50.0%]** |
| **Assignment Target (~5h Train, ~1h Val)** | Yes (`5.46h` / `1.10h`), but Test is `5.35h` | **Yes (`5.50h` Train / `0.92h` Val / `0.92h` Test)** | **Yes (`5.33h` Train / `1.00h` Val / `1.00h` Test)** | Exceeds (`8.77h` Train / `1.46h` Val), gender-skewed | Exceeds (`8.30h` Train / `1.38h` Val), balanced |

---

## Clear Audit Recommendation

We **strongly recommend replacing the current split before running full Telugu IndicMFA alignment** with **Option 1 (or Option 1B)**:

1. **Filter by True Decoded Audio Duration ($1.0\text{s} \le \text{actual\_dur} \le 15.0\text{s}$)**:
   - Eliminates all `148` hidden $> 15.0\text{s}$ utterances (including the `42.70s` training outlier and `34.33s` validation outlier) that slipped through character-rate estimation.
   - Recovers `133` clean Female utterances (`0.4625h`) that were falsely excluded by character-rate estimation, expanding the clean $\le 15.0\text{s}$ Female pool from `1,340` (`3.4014h`) to **`1,435` utterances (`3.6644h`)**.
2. **Adopt the Speaker-Balanced 75% / 12.5% / 12.5% Split (**Option 1** or **Option 1B**)**:
   - Instead of dumping `2,005` excess Male utterances (`4.37h`) into `test.txt` (which made `test.txt` larger than `train.txt` and `92.6%` Male), place the excess Male utterances into a documented `unused_male_reserve` category in `split_summary.json`.
   - Every split (`train.txt`, `val.txt`, and `test.txt`) achieves an **exact 50.0% Female / 50.0% Male duration balance**, satisfies the **73–75% Train / 12.5–13.6% Val / 12.5–13.6% Test** split ratio, and meets the assignment's **~5.3–5.5h Train / ~0.92–1.00h Val** requirement.
   - *(Alternatively, if you prefer ~8.30 hours of 50/50 gender-balanced training data and are comfortable with a `18.0s` max utterance length, **Option 3** provides `3,204` training utterances [`8.30h`], `539` validation utterances [`1.38h`], and `539` test utterances [`1.38h`] at a 75% / 12.5% / 12.5% ratio.)*

> [!IMPORTANT]
> **No dataset or split files have been modified.** All existing files (`Telugu/metadata/train.txt`, `val.txt`, `test.txt`, `split_summary.json`, `text_manifest.json`) remain untouched awaiting your decision on which option to apply.

