"""
Test script to run a single forced alignment using IndicMFA on train_telugufemale_01731.
"""
import os
from pathlib import Path
import sys
import io
import shutil
import subprocess

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import duckdb
import soundfile as sf
import torch
import torchaudio

from common.paths import PROJECT_ROOT
from Telugu.alignment.resample import resample_audio
from Telugu.text.normalizer import normalize_text

target_id = "train_telugufemale_01731"
wav_filename = f"{target_id}.wav"
text = "నాకది రాదే అన్నాడు బ్రాహ్మడు."
norm_text = normalize_text(text)

# 1. Extract audio from parquet
con = duckdb.connect()
row = con.execute(f"""
    SELECT audio.bytes
    FROM 'data/telugu_parquet/data/train-00000-of-00004.parquet'
    WHERE audio.path = '{wav_filename}'
""").fetchone()

assert row is not None, f"Audio {wav_filename} not found in shard 0!"
audio_bytes = row[0]

# Decode audio
wav_tensor, orig_sr = sf.read(io.BytesIO(audio_bytes))
print(f"Original audio: shape={wav_tensor.shape}, sr={orig_sr}")

# Resample to 22050 Hz using our band-limited Kaiser-windowed sinc resampler
wav_tensor = torch.from_numpy(wav_tensor).float()
if wav_tensor.ndim == 1:
    wav_tensor = wav_tensor.unsqueeze(0)

resampled_wav, target_sr = resample_audio(wav_tensor, orig_sr, 22050)
print(f"Resampled audio: shape={resampled_wav.shape}, sr={target_sr}")

# Setup test directory
test_dir = "data/mfa_test"
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

corpus_dir = os.path.join(test_dir, "corpus")
os.makedirs(corpus_dir, exist_ok=True)
out_dir = os.path.join(test_dir, "output")
os.makedirs(out_dir, exist_ok=True)

# Save WAV
out_wav_path = os.path.join(corpus_dir, f"{target_id}.wav")
sf.write(out_wav_path, resampled_wav, 22050)

# Format text for IndicMFA
# Option A: Split every character by space (ignoring whitespace between words)
# e.g., 'న ా క ద ి   ర ా ద ే ...'
# Let's inspect what happens if characters are space separated:
# AI4Bharat training.md: 'Preprocess the corresponding text transcripts by splitting the words into individual letters.'
# Example: 'அ த ற ் க ు த ் த...'
# In their example: words are split into letters separated by spaces.
# Between words, is there a space or does each letter act as a word?
# If each letter is separated by a space, MFA treats each letter as a word!
chars = []
for c in norm_text:
    if c in " .,!?;:'\"-":
        continue
    chars.append(c)

spaced_text = " ".join(chars)
print(f"Spaced text for MFA: {spaced_text}")

out_lab_path = os.path.join(corpus_dir, f"{target_id}.lab")
with open(out_lab_path, "w", encoding="utf-8") as f:
    f.write(spaced_text)

# Check dictionary coverage for every token in spaced_text
with open("research/Telugu_Dictionary_g2g.txt", "r", encoding="utf-8") as f:
    dict_content = f.read()

tokens = spaced_text.split()
missing = [t for t in tokens if t not in dict_content]
print(f"Tokens: {len(tokens)}, Missing in dictionary: {missing}")

# Run MFA align
mfa_bin = "/opt/anaconda3/envs/mfa_env/bin/mfa"
cmd = [
    mfa_bin,
    "align",
    "--clean",
    "--use_mp",
    "--single_speaker",
    "--no_textgrid_cleanup",
    corpus_dir,
    "research/Telugu_Dictionary_g2g.txt",
    "data/telugu_mfa_model/Telugu_Acoustic_Model.zip",
    out_dir
]

print("\nRunning command:", " ".join(cmd))
env = os.environ.copy()
env["PATH"] = "/opt/anaconda3/envs/mfa_env/bin:" + env.get("PATH", "")
res = subprocess.run(cmd, capture_output=True, text=True, env=env)
print(f"Exit code: {res.returncode}")
print("STDOUT:", res.stdout[-1000:])
if res.stderr:
    print("STDERR:", res.stderr[-1000:])

tg_path = os.path.join(out_dir, f"{target_id}.TextGrid")
if os.path.exists(tg_path):
    print(f"\nSUCCESS! TextGrid generated at {tg_path} ({os.path.getsize(tg_path)} bytes)")
    with open(tg_path, "r", encoding="utf-8") as f:
        print("First 30 lines of TextGrid:")
        for _ in range(30):
            print(f.readline().rstrip())
else:
    print(f"\nFAILED! No TextGrid found at {tg_path}")
