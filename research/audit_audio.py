import json, urllib.request, io, os
import soundfile as sf
import numpy as np

with open("research/telugu_sample_100_rows.json") as f:
    rows = json.load(f)

# Pick 15 female (from 0..99) and 15 male (from 100..199)
selected_rows = rows[:15] + rows[100:115]
print(f"Selected {len(selected_rows)} audio samples (15 female, 15 male). Downloading and analyzing...")

results = []
os.makedirs("research/scratch_audio", exist_ok=True)

for idx, r in enumerate(selected_rows):
    row_data = r["row"]
    gender = "female" if row_data["gender"] == 0 else "male"
    audio_src = row_data["audio"][0]["src"]
    text = row_data["text"]
    row_num = r["row_idx"]
    
    req = urllib.request.Request(audio_src, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        audio_bytes = resp.read()
    
    with io.BytesIO(audio_bytes) as bio:
        data, sr = sf.read(bio)
        bio.seek(0)
        sf_info = sf.info(bio)
    
    duration = len(data) / sr
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data**2)))
    rms_db = 20 * np.log10(rms) if rms > 0 else -100
    is_clipped = peak >= 0.999
    is_silent = rms < 1e-4
    
    results.append({
        "sample_idx": idx + 1,
        "row_idx": row_num,
        "gender": gender,
        "sr": sr,
        "channels": 1 if len(data.shape) == 1 else data.shape[1],
        "subtype": sf_info.subtype,
        "duration": duration,
        "samples": len(data),
        "peak": peak,
        "rms": rms,
        "rms_db": rms_db,
        "clipped": is_clipped,
        "silent": is_silent,
        "text": text[:35]
    })
    txt_prefix = text[:25]
    print(f"[{idx+1:02d}/30] row={row_num} {gender:6s} dur={duration:.2f}s sr={sr} peak={peak:.3f} rms={rms_db:.1f}dB | {txt_prefix}")

durations = [r["duration"] for r in results]
female_durs = [r["duration"] for r in results if r["gender"] == "female"]
male_durs = [r["duration"] for r in results if r["gender"] == "male"]

all_srs = set(r["sr"] for r in results)
all_channels = set(r["channels"] for r in results)
all_subtypes = set(r["subtype"] for r in results)
clipped_count = sum(r["clipped"] for r in results)
silent_count = sum(r["silent"] for r in results)

print("\n--- Summary Audio Statistics (30 sample cohort) ---")
print("Sampling Rates:", all_srs)
print("Channels:", all_channels)
print("Subtypes / Bit Depth:", all_subtypes)
print(f"Duration All: min={min(durations):.2f}s, max={max(durations):.2f}s, mean={np.mean(durations):.2f}s, median={np.median(durations):.2f}s")
print(f"Female Durations: min={min(female_durs):.2f}s, max={max(female_durs):.2f}s, mean={np.mean(female_durs):.2f}s")
print(f"Male Durations: min={min(male_durs):.2f}s, max={max(male_durs):.2f}s, mean={np.mean(male_durs):.2f}s")
print(f"Clipped samples count: {clipped_count}")
print(f"Silent samples count: {silent_count}")
print("Corrupted samples count: 0 (all 30 parsed successfully)")

with open("research/telugu_audio_audit_30_samples.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved detailed results to research/telugu_audio_audit_30_samples.json")
