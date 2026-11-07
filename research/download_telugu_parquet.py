"""
Script to download remaining Telugu parquet shards (1, 2, 3) to data/telugu_parquet/data/
"""
import os
import time
from huggingface_hub import hf_hub_download

repo_id = "SPRINGLab/IndicTTS_Telugu"
dest_dir = "data/telugu_parquet/data"
os.makedirs(dest_dir, exist_ok=True)

shards = [
    "data/train-00001-of-00004.parquet",
    "data/train-00002-of-00004.parquet",
    "data/train-00003-of-00004.parquet",
]

for shard in shards:
    basename = os.path.basename(shard)
    target_path = os.path.join(dest_dir, basename)
    if os.path.exists(target_path):
        print(f"{target_path} already exists ({os.path.getsize(target_path)} bytes). Skipping.")
        continue
    print(f"Downloading {shard} -> {target_path}...")
    t0 = time.time()
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=shard,
        repo_type="dataset",
        local_dir="data/telugu_parquet",
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(downloaded) / (1024 * 1024)
    print(f"Downloaded {basename}: {size_mb:.2f} MB in {elapsed:.2f}s ({size_mb/elapsed:.2f} MB/s)")

print("All parquet shards ready in", dest_dir)

