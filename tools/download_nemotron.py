import os
import sys
import time
import urllib.request

URL = "https://huggingface.co/mradermacher/OpenMath-Nemotron-14B-Kaggle-i1-GGUF/resolve/main/OpenMath-Nemotron-14B-Kaggle.i1-Q4_K_M.gguf"
DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(DEST_DIR, exist_ok=True)
DEST_PATH = os.path.join(DEST_DIR, "OpenMath-Nemotron-14B-Kaggle.i1-Q4_K_M.gguf")
PART_PATH = DEST_PATH + ".part"

def download_loop():
    print(f"Target destination: {DEST_PATH}")
    if os.path.exists(DEST_PATH):
        print(f"File already exists: {DEST_PATH}")
        return

    while True:
        downloaded = 0
        if os.path.exists(PART_PATH):
            downloaded = os.path.getsize(PART_PATH)
            print(f"Resuming download from byte {downloaded} ({round(downloaded / (1024**2), 2)} MB)...")

        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        if downloaded > 0:
            req.add_header("Range", f"bytes={downloaded}-")

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                content_range = resp.headers.get("Content-Range")
                total_size = None
                if content_range:
                    total_size = int(content_range.split("/")[-1])
                else:
                    cl = resp.headers.get("Content-Length")
                    if cl:
                        total_size = downloaded + int(cl)

                mode = "ab" if downloaded > 0 else "wb"
                chunk_size = 1024 * 1024
                start_time = time.time()
                last_log = start_time

                with open(PART_PATH, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            # Finished!
                            os.rename(PART_PATH, DEST_PATH)
                            print(f"Successfully finished download: {DEST_PATH}")
                            return
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_log >= 15:
                            pct = (downloaded / total_size * 100) if total_size else 0.0
                            mb = downloaded / (1024 * 1024)
                            speed = (mb / (now - start_time)) if (now - start_time) > 0 else 0
                            print(f"Progress: {mb:.1f} MB / {total_size/(1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s", flush=True)
                            last_log = now
        except Exception as e:
            print(f"Network glitch ({e}). Retrying in 5 seconds from byte {os.path.getsize(PART_PATH) if os.path.exists(PART_PATH) else 0}...", flush=True)
            time.sleep(5)
            continue

if __name__ == "__main__":
    download_loop()