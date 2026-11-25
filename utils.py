"""Shared code: dataset splits, frame sampling, CNN feature extraction and the two models."""
import os
import re

import cv2
import numpy as np

import config as C


# ---------------- Dataset splits ----------------
def read_split(filename):
    """Read an official UCF101 split file and keep only the chosen classes."""
    items = []
    with open(C.SPLIT_DIR / filename) as f:
        for line in f:
            rel = line.strip().split(" ")[0]
            if not rel:
                continue
            cls = rel.split("/")[0]
            if cls in C.CLASS_TO_IDX:
                items.append((str(C.VIDEO_DIR / rel), C.CLASS_TO_IDX[cls]))
    return items


def group_id(path):
    """UCF101 file names look like v_Biking_g08_c02.avi; the group is the person/setting."""
    return int(re.search(r"_g(\d+)_", os.path.basename(path)).group(1))


def get_splits():
    """Official split 1: groups 1-7 test, 8-25 train. Groups 8-10 of train become validation."""
    train_all = read_split("trainlist01.txt")
    test = read_split("testlist01.txt")
    val = [x for x in train_all if group_id(x[0]) <= C.VAL_MAX_GROUP]
    train = [x for x in train_all if group_id(x[0]) > C.VAL_MAX_GROUP]
    return train, val, test


# ---------------- Video frames ----------------
def load_video_frames(path, seq_len=C.SEQ_LEN, size=C.IMG_SIZE):
    """Return `seq_len` evenly spaced RGB frames, shape (seq_len, size, size, 3), or None."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, seq_len).astype(int)
    out = [cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), (size, size)) for i in idx]
    return np.array(out, dtype=np.float32)


# ---------------- CNN feature extractor ----------------
def get_cnn():
    from tensorflow import keras
    cnn = keras.applications.InceptionV3(
        weights=C.CNN_WEIGHTS, include_top=False, pooling="avg",
        input_shape=(C.IMG_SIZE, C.IMG_SIZE, 3),
    )
    cnn.trainable = False
    return cnn


def frames_to_features(frames, cnn):
    from tensorflow import keras
    x = keras.applications.inception_v3.preprocess_input(frames.copy())
    return cnn(x, training=False).numpy()          # (SEQ_LEN, 2048)


def video_to_features(path, cnn, seq_len=C.SEQ_LEN):
    frames = load_video_frames(path, seq_len=seq_len)
    return None if frames is None else frames_to_features(frames, cnn)


# ---------------- Temporal augmentation ----------------
def random_clip_indices(rng, dense_len=C.DENSE_LEN, seq_len=C.SEQ_LEN):
    """Split the dense sequence into seq_len equal segments and pick one random frame from each.
    Keeps the clip ordered and spread over the whole video, but every epoch sees different frames."""
    edges = np.linspace(0, dense_len, seq_len + 1)
    lo, hi = edges[:-1].astype(int), np.maximum(edges[1:].astype(int), edges[:-1].astype(int) + 1)
    return rng.integers(lo, hi)


def center_clip_indices(dense_len=C.DENSE_LEN, seq_len=C.SEQ_LEN):
    return np.linspace(0, dense_len - 1, seq_len).astype(int)


# ---------------- Models ----------------
# Both models share the same per-frame projection (2048 -> 256) and regularisation, so the
# ONLY difference between them is how the 20 frames are combined: BiLSTM vs averaging.
LSTM_ARCH = ["Feature dropout 0.3", "Dense 256 per frame", "BiLSTM 128", "Dropout 0.5"]
BASELINE_ARCH = ["Feature dropout 0.3", "Dense 256 per frame", "Average over frames", "Dropout 0.5"]


def _frame_projection(inp):
    from tensorflow.keras import layers, regularizers
    x = layers.Dropout(0.3)(inp)                                   # drop random feature values
    x = layers.Dense(256, activation="relu",                       # applied to each frame
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    return layers.Dropout(0.3)(x)


def _compile(model):
    from tensorflow import keras
    model.compile(optimizer=keras.optimizers.Adam(C.LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model(num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = keras.Input(shape=(C.SEQ_LEN, C.FEAT_DIM), name="frame_features")
    x = _frame_projection(inp)
    x = layers.Bidirectional(layers.LSTM(128, dropout=0.2))(x)     # reads the sequence both ways
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return _compile(keras.Model(inp, out, name="CNN_BiLSTM"))


def build_baseline_model(num_classes):
    """Same as above but the frames are averaged, so their order is thrown away."""
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = keras.Input(shape=(C.SEQ_LEN, C.FEAT_DIM), name="frame_features")
    x = _frame_projection(inp)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return _compile(keras.Model(inp, out, name="CNN_AvgPool_Baseline"))
