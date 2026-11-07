"""
Builds the 50-utterance representative pilot corpus from Telugu/metadata/train.txt.
Criteria:
- 25 female, 25 male utterances.
- Duration diversity: short (1-4s), medium (4-8s), long (8-14s).
- Phonetic/orthographic diversity: vowels, consonants, matras, virama, anusvara, visarga, conjuncts.
- Includes corrected transcript example (train_telugumale_02568).
- Resamples audio from original sample rate (16k/48k) to 22050 Hz using Kaiser-windowed sinc resampler.
- Generates corpus directory with .wav and .lab files for IndicMFA.
"""
import os
from pathlib import Path
import sys
import json
import duckdb
import soundfile as sf
import numpy as np

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import PROJECT_ROOT
from Telugu.alignment.resample import resample_audio
from Telugu.text.normalizer import normalize_text

def select_pilot():
    with open("Telugu/metadata/train.txt", "r", encoding="utf-8") as f:
        train_lines = [line.strip().split("|") for line in f if line.strip()]

    print(f"Total training utterances available: {len(train_lines)}")
    female_pool = [line for line in train_lines if line[5] == "female"]
    male_pool = [line for line in train_lines if line[5] == "male"]

    # Target: 25 female, 25 male
    # Must include train_telugumale_02568
    male_target_special = "train_telugumale_02568"
    special_male = [line for line in male_pool if line[0] == male_target_special]
    assert len(special_male) == 1, f"Special utterance {male_target_special} not found!"

    # Criteria scorers to ensure maximum orthographic coverage
    def score_diversity(line):
        text = line[3]
        score = 0
        if "ః" in text: score += 50 # Visarga is rare
        if "ఋ" in text or "ృ" in text: score += 30 # Vocalic R
        if "ఐ" in text or "ై" in text: score += 15
        if "ఔ" in text or "ౌ" in text: score += 20
        if "ం" in text: score += 10 # Anusvara
        if "్" in text: score += 10 # Virama / Conjunct
        if any(c in text for c in "ఙఞఠఢఝ"): score += 20 # Rare aspirates/nasals
        return score

    # Select stratified by duration bins:
    # Short: dur < 4.0s (7 female, 7 male)
    # Medium: 4.0s <= dur < 8.0s (11 female, 11 male)
    # Long: 8.0s <= dur <= 14.5s (7 female, 7 male including special)

    def select_bins(pool, is_male=False):
        short = sorted([l for l in pool if float(l[4]) < 4.0], key=score_diversity, reverse=True)
        med = sorted([l for l in pool if 4.0 <= float(l[4]) < 8.0], key=score_diversity, reverse=True)
        long = sorted([l for l in pool if float(l[4]) >= 8.0], key=score_diversity, reverse=True)

        selected = []
        if is_male:
            # Add special to long bin
            selected.append(special_male[0])
            long = [l for l in long if l[0] != male_target_special]
            selected.extend(long[:6]) # 7 long total
            selected.extend(med[:11]) # 11 med total
            selected.extend(short[:7]) # 7 short total
        else:
            selected.extend(long[:7])
            selected.extend(med[:11])
            selected.extend(short[:7])
        return selected

    selected_f = select_bins(female_pool, is_male=False)
    selected_m = select_bins(male_pool, is_male=True)
    all_selected = selected_f + selected_m

    print(f"Selected: {len(selected_f)} female, {len(selected_m)} male = {len(all_selected)} total.")
    assert len(all_selected) == 50

    # Verify coverage
    all_text = " ".join(l[3] for l in all_selected)
    print("\nCoverage in 50 pilot utterances:")
    print("Visarga (ః) count:", all_text.count("ః"))
    print("Anusvara (ం) count:", all_text.count("ం"))
    print("Virama (్) count:", all_text.count("్"))
    print("Vocalic R (ఋ / ృ) count:", all_text.count("ఋ") + all_text.count("ృ"))

    # Connect to DuckDB to extract audio
    con = duckdb.connect()
    corpus_dir = "data/telugu_mfa_pilot/corpus"
    os.makedirs(corpus_dir, exist_ok=True)

    pilot_manifest = []

    print("\nExtracting and resampling 50 pilot utterances...")
    shards = [
        "data/telugu_parquet/data/train-00000-of-00004.parquet",
        "data/telugu_parquet/data/train-00001-of-00004.parquet",
        "data/telugu_parquet/data/train-00002-of-00004.parquet",
        "data/telugu_parquet/data/train-00003-of-00004.parquet"
    ]

    for idx, item in enumerate(all_selected, 1):
        uid, spk_id, raw_t, norm_t, est_dur, spk = item
        clean_text = normalize_text(raw_t) # Applies Latin 'o' correction and NFC normalization
        wav_name = f"{uid}.wav"

        # Find in shards
        row = None
        for sf_path in shards:
            res = con.execute(f"SELECT audio.bytes, gender FROM '{sf_path}' WHERE audio.path = '{wav_name}'").fetchone()
            if res:
                row = res
                break
        
        assert row is not None, f"Could not find {wav_name} in any shard!"
        audio_bytes, gender = row
        
        import io
        bio = io.BytesIO(audio_bytes)
        wav_data, sr = sf.read(bio)
        
        # Resample to 22050 Hz
        resampled_data, target_sr = resample_audio(wav_data, sr, 22050)
        
        # Save WAV
        wav_out = os.path.join(corpus_dir, wav_name)
        sf.write(wav_out, resampled_data, 22050)
        
        # Format text for IndicMFA: space-separated elementary characters
        chars = [c for c in clean_text if c not in " .,!?;:'\"-"]
        spaced_text = " ".join(chars)
        
        lab_out = os.path.join(corpus_dir, f"{uid}.lab")
        with open(lab_out, "w", encoding="utf-8") as lf:
            lf.write(spaced_text)

        actual_dur = len(resampled_data) / 22050.0
        pilot_manifest.append({
            "id": uid,
            "speaker": spk,
            "gender": int(gender),
            "raw_text": raw_t,
            "normalized_text": clean_text,
            "spaced_mfa_text": spaced_text,
            "original_sr": sr,
            "target_sr": 22050,
            "actual_duration_seconds": actual_dur,
            "estimated_duration_seconds": float(est_dur)
        })

    # Save pilot manifest
    with open("research/telugu_mfa_pilot_manifest.json", "w", encoding="utf-8") as pf:
        json.dump(pilot_manifest, pf, indent=2, ensure_ascii=False)

    print(f"\nAll 50 pilot utterances saved to {corpus_dir}")
    print("Pilot manifest written to research/telugu_mfa_pilot_manifest.json")

    # Summary table
    f_durs = [m["actual_duration_seconds"] for m in pilot_manifest if m["gender"] == 0]
    m_durs = [m["actual_duration_seconds"] for m in pilot_manifest if m["gender"] == 1]
    print(f"Female: 25 utts, total={sum(f_durs):.2f}s, min={min(f_durs):.2f}s, max={max(f_durs):.2f}s, mean={np.mean(f_durs):.2f}s")
    print(f"Male:   25 utts, total={sum(m_durs):.2f}s, min={min(m_durs):.2f}s, max={max(m_durs):.2f}s, mean={np.mean(m_durs):.2f}s")
    print(f"Total:  50 utts, total={sum(f_durs)+sum(m_durs):.2f}s ({(sum(f_durs)+sum(m_durs))/60.0:.2f} mins)")

if __name__ == "__main__":
    select_pilot()

