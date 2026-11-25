# Video Action Recognition with CNN + LSTM (UCF101)

A pretrained **InceptionV3** CNN turns each of 20 sampled frames into a 2048-d feature vector, and a two-layer **LSTM** learns how those features change over time to classify the action. A CNN-only baseline (average of frame features) is trained too, to show what the LSTM adds. Results are shown in a **Streamlit dashboard** with a live video-upload demo.

```
video → 20 frames → InceptionV3 (frozen) → 20 × 2048 features → LSTM → Dense → softmax
```

## Two experiments
| Experiment | Classes | Purpose |
|---|---|---|
| `scene` | Basketball, Biking, HorseRiding, JumpRope, PlayingGuitar, Punch, PushUps, TennisSwing, Typing, WalkingWithDog | Scene-heavy: objects and setting give the answer away |
| `motion` | BodyWeightSquats, BoxingPunchingBag, BoxingSpeedBag, HandstandWalking, JumpingJack, JumpRope, Lunges, PullUps, PushUps, WallPushups | Motion-heavy: similar gym settings, the movement decides |
| `mixed25` | All 18 actions above + CricketBowling, CricketShot, PlayingDhol, PlayingSitar, PlayingTabla, Archery, TaiChi | One model for 25 classes, with fine-grained pairs (cricket bowling vs shot, tabla vs dhol) |

The question: **does the LSTM help more when motion, not scene, decides the action?**

## Models
Both models share a per-frame projection (feature dropout → Dense 256 → dropout). The only difference is how the 20 frames are combined:
- **CNN + BiLSTM:** a bidirectional LSTM (128 units) reads the frames in order.
- **Baseline:** the frames are averaged, so their order is lost.

**Temporal augmentation:** 40 frames are cached for each training video. Every epoch, each video gives a different ordered 20-frame clip (one random frame from each of 20 segments). Validation and test always use 20 evenly spaced frames.

## Project files
| File | What it does |
|---|---|
| `config.py` | Settings and the two experiments' class lists |
| `exp_arg.py` | Reads `--exp scene` / `--exp motion` |
| `utils.py` | Dataset split, frame sampling, CNN features, both models |
| `download_data.py` | Step 1: downloads UCF101 and extracts the experiment's classes |
| `extract_features.py` | Step 2: CNN features, cached in `data/features/<exp>/` |
| `train.py` | Step 3: trains both models, saves `outputs/<exp>/results.json` |
| `app.py` | Step 4: the dashboard |

## Setup (VS Code)
Use **Python 3.10–3.13**. In the VS Code terminal:
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Extracting the .rar needs **7-Zip** on Windows (`winget install 7zip.7zip`) or `unrar` on macOS/Linux.

## Run
```bash
python download_data.py --exp scene
python extract_features.py --exp scene
python train.py --exp scene

python download_data.py --exp motion     # reuses the downloaded .rar
python extract_features.py --exp motion
python train.py --exp motion

python download_data.py --exp mixed25    # extracts the 7 new classes
python extract_features.py --exp mixed25 # reuses features of the 18 known actions
python train.py --exp mixed25

streamlit run app.py
```
Features are shared between experiments: a video already processed for one experiment is copied, not re-extracted.

Feature extraction is the slow step on CPU (40 frames per training video). It runs once per experiment; training afterwards takes a few minutes.

## Dashboard
- **Overview:** architecture, augmentation, dataset split per class
- **Results:** accuracy, BiLSTM vs baseline, training curves, confusion matrices for both models, per-class F1
- **Scene vs motion:** BiLSTM gain over the baseline in each experiment
- **Test video explorer:** browse test videos, filter by class or mistakes
- **Try your own video:** upload a clip and get a live prediction (uses the experiment selected in the sidebar)

## Dataset split
Official UCF101 split 1: groups 1–7 for testing, 8–25 for training. Groups 8–10 are held out from training for validation. A group is one person in one setting, so no person appears in both training and test.
