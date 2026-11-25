"""Step 3: train the CNN+BiLSTM model and the CNN-only baseline, evaluate both on the test set,
and save everything the dashboard needs to outputs/<experiment>/.

Both models are trained the same way, with temporal augmentation: every epoch, each training
video gives a different ordered 20-frame clip picked from its 40 cached frames.

Run:  python train.py --exp scene
      python train.py --exp motion
"""
import exp_arg  # noqa: F401  (must be first: sets the experiment)

import json
import random
import time
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

import config as C
from utils import (BASELINE_ARCH, LSTM_ARCH, build_baseline_model, build_lstm_model,
                   random_clip_indices)


class AugmentedClips(keras.utils.PyDataset):
    """Yields batches of 20-frame clips, re-sampled from the 40-frame cache every epoch."""

    def __init__(self, X_dense, y, batch_size, seed, **kwargs):
        super().__init__(**kwargs)
        self.X, self.y, self.bs = X_dense, y, batch_size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(y))

    def __len__(self):
        return int(np.ceil(len(self.y) / self.bs))

    def __getitem__(self, i):
        idx = self.order[i * self.bs:(i + 1) * self.bs]
        clips = np.stack([self.X[j, random_clip_indices(self.rng, self.X.shape[1])] for j in idx])
        return clips.astype(np.float32), self.y[idx]

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.y))


def load(name):
    X = np.load(C.FEAT_DIR / f"{name}_X.npy")
    y = np.load(C.FEAT_DIR / f"{name}_y.npy")
    paths = json.loads((C.FEAT_DIR / f"{name}_paths.json").read_text())
    return X, y, paths


def train_model(model, X_dense, y_tr, X_va, y_va):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ]
    t0 = time.time()
    h = model.fit(AugmentedClips(X_dense, y_tr, C.BATCH_SIZE, C.SEED),
                  validation_data=(X_va, y_va), epochs=C.EPOCHS, callbacks=callbacks, verbose=2)
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items() if k != "learning_rate"}
    return hist, time.time() - t0


def evaluate(model, X, y):
    probs = model.predict(X, verbose=0)
    pred = probs.argmax(1)
    labels = list(range(len(C.CLASSES)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "report": classification_report(y, pred, labels=labels, target_names=C.CLASSES,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "pred": pred.tolist(),
        "probs": np.round(probs, 4).tolist(),
    }


if __name__ == "__main__":
    random.seed(C.SEED); np.random.seed(C.SEED); tf.random.set_seed(C.SEED)
    print(f"Experiment: {C.EXP}")

    try:
        X_tr, y_tr, _ = load("train_dense")
        X_va, y_va, _ = load("val")
        X_te, y_te, p_te = load("test")
    except FileNotFoundError:
        raise SystemExit(f"Features not found. Run python extract_features.py --exp {C.EXP} first.")
    print("Train", X_tr.shape, "Val", X_va.shape, "Test", X_te.shape)
    if min(len(X_tr), len(X_va), len(X_te)) == 0 or X_tr.ndim != 3:
        raise SystemExit(f"Features are empty. Delete data/features/{C.EXP} and run "
                         f"python extract_features.py --exp {C.EXP} again.")

    results = {
        "experiment": C.EXP,
        "description": C.EXPERIMENT_INFO[C.EXP],
        "classes": C.CLASSES,
        "config": {"seq_len": C.SEQ_LEN, "dense_len_train": C.DENSE_LEN, "img_size": C.IMG_SIZE,
                   "feat_dim": C.FEAT_DIM, "cnn": "InceptionV3 (ImageNet, frozen)",
                   "augmentation": f"random ordered {C.SEQ_LEN}-of-{C.DENSE_LEN} frames per epoch",
                   "epochs_max": C.EPOCHS, "batch_size": C.BATCH_SIZE, "learning_rate": C.LEARNING_RATE},
        "arch": {"lstm": LSTM_ARCH, "baseline": BASELINE_ARCH},
        "counts": {name: [Counter(y.tolist()).get(i, 0) for i in range(len(C.CLASSES))]
                   for name, y in (("train", y_tr), ("val", y_va), ("test", y_te))},
        "test_paths": p_te,
        "test_labels": y_te.tolist(),
        "models": {},
    }

    for key, builder in (("lstm", build_lstm_model), ("baseline", build_baseline_model)):
        print(f"\n===== Training {key} =====")
        tf.random.set_seed(C.SEED)
        model = builder(len(C.CLASSES))
        hist, secs = train_model(model, X_tr, y_tr, X_va, y_va)
        ev = evaluate(model, X_te, y_te)
        model.save(C.OUT_DIR / f"{key}_model.keras")
        results["models"][key] = {"name": model.name, "params": int(model.count_params()),
                                  "train_seconds": round(secs, 1), "epochs_run": len(hist["loss"]),
                                  "history": hist, **ev}
        print(f"{key}: test accuracy {ev['accuracy'] * 100:.2f}%")

    (C.OUT_DIR / "results.json").write_text(json.dumps(results))
    L, B = results["models"]["lstm"]["accuracy"], results["models"]["baseline"]["accuracy"]
    print(f"\n[{C.EXP}] CNN+BiLSTM {L * 100:.1f}%  vs  baseline {B * 100:.1f}%  "
          f"({(L - B) * 100:+.1f} pts)")
    print("Saved to", C.OUT_DIR, "\nNext: streamlit run app.py")
