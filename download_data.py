"""Step 1: download UCF101 and the official split files, and extract only the chosen classes.

Run:  python download_data.py --exp scene
      python download_data.py --exp motion   (reuses the downloaded .rar, extracts the other classes)
"""
import exp_arg  # noqa: F401  (must be first: sets the experiment)

import os
import shutil
import subprocess
import sys
import time
import zipfile

import requests
from tqdm import tqdm

import config as C

RAR_PATH = C.DATA_DIR / "UCF101.rar"
ZIP_PATH = C.DATA_DIR / "splits.zip"


VERIFY_SSL = True


def _get(url, **kw):
    """GET that falls back to no certificate check if the UCF server's certificate fails
    (Colab tutorials use wget --no-check-certificate for the same reason; public dataset)."""
    global VERIFY_SSL
    try:
        return requests.get(url, stream=True, timeout=60, verify=VERIFY_SSL, **kw)
    except requests.exceptions.SSLError:
        print("  SSL certificate check failed on the UCF server; continuing without verification.")
        requests.packages.urllib3.disable_warnings()
        VERIFY_SSL = False
        return requests.get(url, stream=True, timeout=60, verify=False, **kw)


def download(url, dest, max_retries=100):
    """Resumable download: keeps a .part file and continues from where it stopped
    if the connection drops (also across separate runs of this script)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already downloaded: {dest.name}")
        return
    print(f"Downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = None
    bar = None
    for attempt in range(1, max_retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and done >= total:
            break
        try:
            r = _get(url, headers={"Range": f"bytes={done}-"} if done else {})
            if r.status_code == 416:          # range not satisfiable: file already complete
                break
            r.raise_for_status()
            if done and r.status_code != 206:  # server ignored the range: start over
                print("  Server does not support resume; restarting download.")
                done = 0
                tmp.unlink(missing_ok=True)
            if total is None:
                total = done + int(r.headers.get("content-length", 0))
            if bar is None:
                bar = tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=dest.name)
            with open(tmp, "ab") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
            if tmp.stat().st_size >= total:
                break
        except (requests.exceptions.RequestException, ConnectionError) as e:
            wait = min(5 * attempt, 60)
            print(f"\n  Connection dropped ({type(e).__name__}). Resuming in {wait}s "
                  f"(attempt {attempt}/{max_retries})...")
            time.sleep(wait)
    else:
        if bar:
            bar.close()
        sys.exit("Download kept failing. Run the script again later: it will resume from the saved part.")
    if bar:
        bar.close()
    tmp.replace(dest)


def extract_splits():
    if (C.SPLIT_DIR / "trainlist01.txt").exists():
        return
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(C.DATA_DIR)
    print("Split files extracted.")


def find_7zip():
    for cand in (shutil.which("7z"), shutil.which("7za"),
                 r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if cand and os.path.exists(cand):
            return cand
    return None


def class_has_videos(c):
    return any((C.VIDEO_DIR / c).glob("*.avi"))


def extract_classes():
    missing = [c for c in C.CLASSES if not class_has_videos(c)]
    if not missing:
        print("All class folders already extracted.")
        return
    print(f"Extracting {len(missing)} classes from {RAR_PATH.name} (takes a few minutes)...")

    attempts = []
    seven = find_7zip()
    if seven:   # 7-Zip: folder names as filters extract the whole folder
        attempts.append(("7-Zip", [seven, "x", str(RAR_PATH), f"-o{C.DATA_DIR}", "-y", "-aos"]
                         + [f"UCF-101\\{c}" if os.name == "nt" else f"UCF-101/{c}" for c in missing]))
    if shutil.which("tar"):   # Windows 10/11 ship tar.exe (libarchive), which can read RAR
        attempts.append(("tar", ["tar", "-xf", str(RAR_PATH), "-C", str(C.DATA_DIR)]
                         + [f"UCF-101/{c}" for c in missing]))
    if shutil.which("unrar"):
        attempts.append(("unrar", ["unrar", "x", "-o+", "-inul", str(RAR_PATH)]
                         + [f"UCF-101/{c}/*" for c in missing] + [str(C.DATA_DIR) + os.sep]))

    for name, cmd in attempts:
        print(f"  trying {name}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        still_missing = [c for c in missing if not class_has_videos(c)]
        if not still_missing:
            print(f"  extracted with {name}")
            return
        print(f"  {name} did not extract everything (exit code {res.returncode}). "
              f"{(res.stderr or res.stdout).strip()[-300:]}")

    print("\nCould not extract the .rar automatically.")
    print("Fix: install 7-Zip (PowerShell: winget install 7zip.7zip), then run this script again.")
    print(f"Or open {RAR_PATH} in 7-Zip/WinRAR and copy these folders from UCF-101\\ into "
          f"{C.VIDEO_DIR}:\n  " + "\n  ".join(c for c in missing if not class_has_videos(c)))
    sys.exit(1)


if __name__ == "__main__":
    download(C.SPLIT_URL, ZIP_PATH)
    extract_splits()
    download(C.UCF_URL, RAR_PATH)   # about 6.5 GB
    extract_classes()
    for c in C.CLASSES:
        print(f"  {c:16s} {len(list((C.VIDEO_DIR / c).glob('*.avi')))} videos")
    print(f"Done. Next: python extract_features.py --exp {C.EXP}")
