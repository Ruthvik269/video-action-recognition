"""Step 2: turn video frames into InceptionV3 features (cached, so this runs once per experiment).

- Training videos: 40 evenly spaced frames, so training can pick a different 20-frame
  clip every epoch (temporal augmentation).
- Validation/test videos: 20 evenly spaced frames, exactly what the model sees at prediction time.

Run:  python extract_features.py --exp scene
      python extract_features.py --exp motion
"""
import exp_arg  # noqa: F401  (must be first: sets the experiment)

import json
import os
import sys

import numpy as np
from tqdm import tqdm

import config as C
from utils import get_cnn, get_splits, video_to_features


def _key(path):
    return os.path.normcase(os.path.abspath(path))


def reusable_features(name, seq_len):
    """Features of the same videos already extracted by OTHER experiments, keyed by video path.
    The same video with the same frame count gives identical features, so they can be copied."""
    found = {}
    for other in C.FEAT_ROOT.iterdir():
        if not other.is_dir() or other == C.FEAT_DIR:
            continue
        x_path, p_path = other / f"{name}_X.npy", other / f"{name}_paths.json"
        if not (x_path.exists() and p_path.exists()):
            continue
        X = np.load(x_path, mmap_mode="r")
        paths = json.loads(p_path.read_text())
        if X.ndim != 3 or X.shape[1] != seq_len or len(paths) != len(X):
            continue
        for i, p in enumerate(paths):
            found.setdefault(_key(p), (X, i))
    return found


def build_feature_set(items, name, cnn, seq_len, dtype=np.float32):
    x_path, y_path, p_path = (C.FEAT_DIR / f"{name}_X.npy", C.FEAT_DIR / f"{name}_y.npy",
                              C.FEAT_DIR / f"{name}_paths.json")
    if x_path.exists() and p_path.exists():
        cached = np.load(x_path, mmap_mode="r")
        if cached.ndim == 3 and len(cached) > 0 and cached.shape[1] == seq_len:
            print(f"{name}: using cached features ({len(cached)} videos)")
            return
        print(f"{name}: cached features are empty or outdated, re-extracting")
    reuse = reusable_features(name, seq_len)
    n_reuse = sum(_key(p) in reuse for p, _ in items)
    if n_reuse:
        print(f"{name}: reusing {n_reuse} of {len(items)} videos already extracted by other experiments")
    X, y, paths = [], [], []
    for path, label in tqdm(items, desc=f"Extracting {name} ({seq_len} frames/video)"):
        if _key(path) in reuse:
            arr, i = reuse[_key(path)]
            feats = np.asarray(arr[i])
        else:
            feats = video_to_features(path, cnn, seq_len=seq_len)
        if feats is None:
            print("  skipped unreadable video:", path)
            continue
        X.append(feats.astype(dtype)); y.append(label); paths.append(path)
    if not X:
        sys.exit(f"\nNo readable videos for '{name}'. Check that {C.VIDEO_DIR} contains the class "
                 f"folders with .avi files (run python download_data.py --exp {C.EXP} first).")
    np.save(x_path, np.array(X, dtype=dtype))
    np.save(y_path, np.array(y))
    p_path.write_text(json.dumps(paths))
    print(f"{name}: {len(X)} videos -> {x_path.name}")


if __name__ == "__main__":
    print(f"Experiment: {C.EXP}  ({', '.join(C.CLASSES)})")
    missing = [c for c in C.CLASSES if not any((C.VIDEO_DIR / c).glob("*.avi"))]
    if missing:
        sys.exit(f"No videos found for: {', '.join(missing)} in {C.VIDEO_DIR}\n"
                 f"Run python download_data.py --exp {C.EXP} first (it extracts them from the .rar).")
    train, val, test = get_splits()
    print(f"Train {len(train)} | Val {len(val)} | Test {len(test)} videos")
    cnn = get_cnn()
    # float16 halves the size of the large 40-frame training cache with no effect on accuracy
    build_feature_set(train, "train_dense", cnn, C.DENSE_LEN, dtype=np.float16)
    build_feature_set(val, "val", cnn, C.SEQ_LEN)
    build_feature_set(test, "test", cnn, C.SEQ_LEN)
    print(f"Done. Next: python train.py --exp {C.EXP}")
