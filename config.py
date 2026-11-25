"""Central settings for the whole project.

Two experiments are defined:
  scene  - actions with distinctive scenes/objects (horse, guitar, keyboard...)
  motion - exercise actions filmed in similar settings, where the movement decides the class
Choose one with --exp on each script (default: scene). Each experiment has its own
features folder and outputs folder.
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Experiments ----
EXPERIMENTS = {
    "scene": [
        "Basketball", "Biking", "HorseRiding", "JumpRope", "PlayingGuitar",
        "Punch", "PushUps", "TennisSwing", "Typing", "WalkingWithDog",
    ],
    "motion": [
        "BodyWeightSquats", "BoxingPunchingBag", "BoxingSpeedBag", "HandstandWalking", "JumpingJack",
        "JumpRope", "Lunges", "PullUps", "PushUps", "WallPushups",
    ],
}
# 25 classes: every action from the two experiments above (18) plus 7 new ones
EXPERIMENTS["mixed25"] = sorted(set(EXPERIMENTS["scene"]) | set(EXPERIMENTS["motion"]) | {
    "CricketBowling", "CricketShot",                    # same pitch, only the movement differs
    "PlayingDhol", "PlayingSitar", "PlayingTabla",      # Indian instruments, next to PlayingGuitar
    "Archery",                                          # easy, scene-based
    "TaiChi",                                           # slow, smooth motion
})
EXPERIMENT_INFO = {
    "scene": "Scene-heavy actions: the objects and setting (horse, guitar, keyboard, court) mostly give the answer away.",
    "motion": "Motion-heavy exercise actions in similar indoor settings: the movement over time decides the class.",
    "mixed25": "25 mixed actions: all scene and motion classes plus cricket, Indian instruments, archery and tai chi. "
               "Includes fine-grained pairs such as CricketBowling vs CricketShot and PlayingTabla vs PlayingDhol.",
}
EXP = os.environ.get("AR_EXP", "scene")
if EXP not in EXPERIMENTS:
    raise SystemExit(f"Unknown experiment '{EXP}'. Choose from: {', '.join(EXPERIMENTS)}")
CLASSES = EXPERIMENTS[EXP]

# ---- Paths ----
DATA_DIR  = Path(os.environ.get("AR_DATA_DIR", ROOT / "data"))
VIDEO_DIR = DATA_DIR / "UCF-101"            # class folders with .avi files
SPLIT_DIR = DATA_DIR / "ucfTrainTestlist"   # official split files
FEAT_ROOT = DATA_DIR / "features"
FEAT_DIR  = FEAT_ROOT / EXP                 # cached CNN features for this experiment
OUT_ROOT  = Path(os.environ.get("AR_OUT_DIR", ROOT / "outputs"))
OUT_DIR   = OUT_ROOT / EXP                  # models + results.json for this experiment

UCF_URL   = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
SPLIT_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
VAL_MAX_GROUP = 10   # training-list groups 8..10 are held out for validation

# ---- Frames and CNN ----
SEQ_LEN   = 20       # frames the model sees per video
DENSE_LEN = 40       # frames cached per TRAINING video, for temporal augmentation
IMG_SIZE  = 299      # InceptionV3 input size
FEAT_DIM  = 2048     # InceptionV3 pooled feature size
_w = os.environ.get("AR_CNN_WEIGHTS", "imagenet")
CNN_WEIGHTS = None if _w.lower() == "none" else _w

# ---- Training ----
EPOCHS        = 100   # upper limit; early stopping usually ends much sooner
BATCH_SIZE    = 32
LEARNING_RATE = 3e-4
SEED          = 42

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
for d in (DATA_DIR, FEAT_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- One-time migration from the earlier single-experiment layout ----
# Earlier versions saved scene features in data/features/ and results in outputs/.
for src_dir, dst_dir in ((FEAT_ROOT, FEAT_ROOT / "scene"), (OUT_ROOT, OUT_ROOT / "scene")):
    for f in list(src_dir.glob("*.npy")) + list(src_dir.glob("*.json")) + list(src_dir.glob("*.keras")):
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not (dst_dir / f.name).exists():
            shutil.move(str(f), str(dst_dir / f.name))
