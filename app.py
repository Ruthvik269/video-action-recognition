"""Step 4: results dashboard and live demo.

Run:  streamlit run app.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
from utils import frames_to_features, load_video_frames

st.set_page_config(page_title="Video Action Recognition", page_icon="🎬", layout="wide")

LSTM_COLOR, BASE_COLOR = "#2563eb", "#9ca3af"
EXP_LABEL = {"scene": "Scene-heavy actions", "motion": "Motion-heavy actions", "mixed25": "25 mixed actions"}


def results_path(exp):
    return C.OUT_ROOT / exp / "results.json"


def mtime(p):
    return p.stat().st_mtime if p.exists() else 0


# ---------------- Cached loaders ----------------
@st.cache_data
def load_results(exp, mt):
    p = results_path(exp)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner="Loading CNN (first time only)...")
def load_cnn():
    from utils import get_cnn
    return get_cnn()


@st.cache_resource(show_spinner="Loading trained model...")
def load_lstm(exp, mt):
    from tensorflow import keras
    return keras.models.load_model(C.OUT_ROOT / exp / "lstm_model.keras")


@st.cache_data(show_spinner=False)
def preview_frames(path, n=8, mt=0):
    frames = load_video_frames(path, seq_len=n, size=180)
    return None if frames is None else frames.astype("uint8")


def show_frames(path, n=8):
    frames = preview_frames(str(path), n, mtime(Path(path)))
    if frames is None:
        st.error("Could not read frames from this video.")
        return
    for i, (col, fr) in enumerate(zip(st.columns(n), frames)):
        col.image(fr, caption=f"frame {i + 1}", use_container_width=True)


def prob_chart(probs, classes, highlight=None, top_k=None):
    df = pd.DataFrame({"Action": classes, "Probability": np.asarray(probs) * 100})
    df = df.sort_values("Probability", ascending=True)
    if top_k:
        df = df.tail(top_k)
    colors = [LSTM_COLOR if a == highlight else "#93c5fd" for a in df["Action"]]
    fig = go.Figure(go.Bar(x=df["Probability"], y=df["Action"], orientation="h", marker_color=colors,
                           text=[f"{p:.1f}%" for p in df["Probability"]], textposition="outside"))
    fig.update_layout(height=40 * len(df) + 60, margin=dict(l=10, r=40, t=10, b=30),
                      xaxis=dict(range=[0, 110], title="Probability (%)"))
    return fig


def pts(x):
    return f"{x * 100:.1f}%"


# ---------------- Sidebar ----------------
available = [e for e in C.EXPERIMENTS if results_path(e).exists()]
with st.sidebar:
    st.title("🎬 Action Recognition")
    st.caption("CNN (InceptionV3) + BiLSTM on UCF101")
    if available:
        exp = st.radio("Experiment", available, format_func=lambda e: EXP_LABEL.get(e, e))
        R = load_results(exp, mtime(results_path(exp)))
        st.caption(R.get("description", ""))
        a, b = st.columns(2)
        a.metric("BiLSTM", pts(R["models"]["lstm"]["accuracy"]))
        b.metric("Baseline", pts(R["models"]["baseline"]["accuracy"]))
        classes = R["classes"]
    else:
        exp, R, classes = C.EXP, None, C.CLASSES
        st.warning("No results yet. Run the training steps first (see Overview).")
    st.markdown("**Actions in this experiment**")
    st.markdown("\n".join(f"- {c}" for c in classes))

tab_over, tab_res, tab_cmp, tab_test, tab_try = st.tabs(
    ["📘 Overview", "📊 Results", "🔬 Scene vs motion", "🎞️ Test video explorer", "🎥 Try your own video"])

# ---------------- Overview ----------------
with tab_over:
    st.header("Video Action Recognition with CNN + LSTM")
    if R:
        st.subheader(f"Dataset: {EXP_LABEL.get(exp, exp)} (official UCF101 split 1)")
        counts = pd.DataFrame({"Action": classes, **{k.capitalize(): v for k, v in R["counts"].items()}})
        m1, m2, m3 = st.columns(3)
        m1.metric("Training videos", int(counts["Train"].sum()))
        m2.metric("Validation videos", int(counts["Val"].sum()))
        m3.metric("Test videos", int(counts["Test"].sum()))
        fig = px.bar(counts.melt(id_vars="Action", var_name="Split", value_name="Videos"),
                     x="Action", y="Videos", color="Split", barmode="stack",
                     color_discrete_map={"Train": LSTM_COLOR, "Val": "#93c5fd", "Test": "#f59e0b"})
        fig.update_layout(height=380, margin=dict(t=10))
        if len(classes) > 15:
            fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Test = groups 1–7, validation = groups 8–10, training = groups 11–25. "
                   "A group is one person in one setting, so no person appears in both training and test.")
    else:
        st.subheader("How to generate results")
        st.code("python download_data.py --exp scene\npython extract_features.py --exp scene\n"
                "python train.py --exp scene\n\n(repeat with --exp motion)\n\nstreamlit run app.py",
                language="bash")

# ---------------- Results ----------------
with tab_res:
    if not R:
        st.info("Run `python train.py` first; results appear here automatically.")
    else:
        L, B = R["models"]["lstm"], R["models"]["baseline"]
        st.caption(f"Experiment: **{EXP_LABEL.get(exp, exp)}**. Switch experiments in the sidebar.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CNN + BiLSTM accuracy", pts(L["accuracy"]),
                  f"{(L['accuracy'] - B['accuracy']) * 100:+.1f} pts vs baseline")
        c2.metric("Baseline accuracy (frames averaged)", pts(B["accuracy"]))
        c3.metric("Macro F1 (CNN + BiLSTM)", f"{L['report']['macro avg']['f1-score']:.3f}")
        c4.metric("Test videos", len(R["test_labels"]))

        left, right = st.columns([1, 2])
        with left:
            st.subheader("Does modelling time help?")
            fig = go.Figure(go.Bar(x=["Frames averaged<br>(baseline)", "CNN + BiLSTM"],
                                   y=[B["accuracy"] * 100, L["accuracy"] * 100],
                                   marker_color=[BASE_COLOR, LSTM_COLOR],
                                   text=[pts(B["accuracy"]), pts(L["accuracy"])], textposition="outside"))
            fig.update_layout(yaxis=dict(range=[0, 105], title="Test accuracy (%)"), height=380, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Training curves")
            which = st.radio("Model", ["CNN + BiLSTM", "Baseline"], horizontal=True, key="curves")
            h = (L if which == "CNN + BiLSTM" else B)["history"]
            ep = list(range(1, len(h["loss"]) + 1))
            ca, cb = st.columns(2)
            for col, metric, title in ((ca, "accuracy", "Accuracy"), (cb, "loss", "Loss")):
                fig = go.Figure()
                fig.add_scatter(x=ep, y=h[metric], name="train", line=dict(color=LSTM_COLOR))
                fig.add_scatter(x=ep, y=h[f"val_{metric}"], name="validation", line=dict(color="#f59e0b"))
                fig.update_layout(title=title, xaxis_title="Epoch", height=330, margin=dict(t=40, b=10),
                                  legend=dict(orientation="h", y=-0.25))
                col.plotly_chart(fig, use_container_width=True)

        st.subheader("Confusion matrix")
        cm_model = st.radio("Model", ["CNN + BiLSTM", "Baseline"], horizontal=True, key="cm_model")
        M = L if cm_model == "CNN + BiLSTM" else B
        cm = np.array(M["confusion_matrix"], dtype=float)
        norm = st.toggle("Show as percentage of each true class", value=True)
        z = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100 if norm else cm
        fig = px.imshow(z, x=classes, y=classes, color_continuous_scale="Blues", text_auto=".0f",
                        labels=dict(x="Predicted", y="True", color="%" if norm else "videos"), aspect="auto")
        fig.update_layout(height=max(560, 30 * len(classes)), margin=dict(t=10))
        if len(classes) > 15:
            fig.update_traces(textfont_size=9)
            fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes))
               for j in range(len(classes)) if i != j and cm[i, j] > 0]
        if off:
            st.markdown("**Most common mistakes:** " + "; ".join(
                f"{t} → {p} ({int(n)}×)" for n, t, p in sorted(off, reverse=True)[:5 if len(classes) > 15 else 3]))

        st.subheader("Per-class F1 score")
        f1 = pd.DataFrame({"Action": classes,
                           "CNN + BiLSTM": [L["report"][c]["f1-score"] for c in classes],
                           "Baseline": [B["report"][c]["f1-score"] for c in classes]})
        fig = px.bar(f1.melt(id_vars="Action", var_name="Model", value_name="F1"), x="Action", y="F1",
                     color="Model", barmode="group",
                     color_discrete_map={"CNN + BiLSTM": LSTM_COLOR, "Baseline": BASE_COLOR})
        fig.update_layout(yaxis_range=[0, 1.05], height=380, margin=dict(t=10))
        if len(classes) > 15:
            fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# ---------------- Scene vs motion ----------------
with tab_cmp:
    st.header("Does the LSTM matter more when motion decides the action?")
    st.markdown(
        "Both experiments use the same models and training. **Scene-heavy** actions can often be "
        "recognised from a single frame (a horse, a guitar), so averaging the frames works well. "
        "**Motion-heavy** exercises look alike frame by frame (a person in a gym), so the order of "
        "the frames should matter more.")
    allR = {e: load_results(e, mtime(results_path(e))) for e in available if e in ("scene", "motion")}
    if len(allR) < 2:
        missing = [e for e in ("scene", "motion") if e not in allR]
        st.info("Train both experiments to see this comparison. Still missing: "
                + ", ".join(f"`python train.py --exp {e}`" for e in missing))
    if allR:
        rows = []
        for e, r in allR.items():
            for key, label in (("baseline", "Frames averaged (baseline)"), ("lstm", "CNN + BiLSTM")):
                rows.append({"Experiment": EXP_LABEL.get(e, e), "Model": label,
                             "Accuracy": r["models"][key]["accuracy"] * 100})
        df = pd.DataFrame(rows)
        fig = px.bar(df, x="Experiment", y="Accuracy", color="Model", barmode="group", text_auto=".1f",
                     color_discrete_map={"Frames averaged (baseline)": BASE_COLOR, "CNN + BiLSTM": LSTM_COLOR})
        fig.update_layout(yaxis=dict(range=[0, 105], title="Test accuracy (%)"), height=420, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(len(allR))
        for col, (e, r) in zip(cols, allR.items()):
            gap = (r["models"]["lstm"]["accuracy"] - r["models"]["baseline"]["accuracy"]) * 100
            col.metric(f"{EXP_LABEL.get(e, e)}: BiLSTM gain over baseline", f"{gap:+.1f} pts")
            col.caption(", ".join(r["classes"]))
        if len(allR) == 2:
            g = {e: (r["models"]["lstm"]["accuracy"] - r["models"]["baseline"]["accuracy"]) * 100
                 for e, r in allR.items()}
            if g["motion"] > g["scene"]:
                st.success(f"The LSTM's advantage is {g['motion'] - g['scene']:.1f} points larger on "
                           "motion-heavy actions, so modelling the order of frames matters most when "
                           "appearance alone can't separate the classes.")
            else:
                st.warning("The LSTM's advantage is not larger on motion-heavy actions in this run. "
                           "Check the per-class results: the classes may still differ by objects or "
                           "setting (a punching bag, a pull-up bar), which averaging can pick up.")

# ---------------- Test video explorer ----------------
with tab_test:
    if not R:
        st.info("Run `python train.py` first.")
    else:
        L = R["models"]["lstm"]
        df = pd.DataFrame({"path": R["test_paths"], "true": [classes[i] for i in R["test_labels"]],
                           "pred": [classes[i] for i in L["pred"]]})
        df["name"] = df["path"].map(lambda p: Path(p).name)
        df["correct"] = df["true"] == df["pred"]
        f1c, f2c = st.columns(2)
        cls = f1c.selectbox("Action", ["All"] + classes)
        res = f2c.radio("Show", ["All", "Correct only", "Mistakes only"], horizontal=True)
        sub = df if cls == "All" else df[df["true"] == cls]
        sub = sub[sub["correct"]] if res == "Correct only" else sub[~sub["correct"]] if res == "Mistakes only" else sub
        if sub.empty:
            st.success("No videos match this filter.")
        else:
            st.caption(f"{len(sub)} videos · {sub['correct'].mean() * 100:.0f}% correct in this selection")
            name = st.selectbox("Video", sub["name"].tolist())
            row = df[df["name"] == name].iloc[0]
            idx = int(row.name)
            show_frames(row["path"])
            a, b = st.columns([1, 2])
            a.markdown(f"**True action:** {row['true']}")
            a.markdown(f"**Predicted:** {row['pred']}  {'✅' if row['correct'] else '❌'}")
            a.markdown(f"**Confidence:** {max(L['probs'][idx]) * 100:.1f}%")
            b.plotly_chart(prob_chart(L["probs"][idx], classes, highlight=row["pred"]),
                           use_container_width=True)

# ---------------- Try your own video ----------------
with tab_try:
    model_path = C.OUT_ROOT / exp / "lstm_model.keras"
    if not R or not model_path.exists():
        st.info("Train the model first (`python train.py`).")
    else:
        st.markdown(f"Using the **{EXP_LABEL.get(exp, exp)}** model. Upload a short clip of one of: "
                    + ", ".join(classes) + ".")
        up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm"])
        if up:
            tmp = Path(tempfile.gettempdir()) / f"ar_upload{Path(up.name).suffix}"
            tmp.write_bytes(up.getvalue())
            left, right = st.columns([1, 1])
            with left:
                st.video(up.getvalue())
            with st.spinner("Extracting frames and running CNN + BiLSTM..."):
                frames = load_video_frames(tmp)
                probs = None if frames is None else load_lstm(exp, mtime(model_path)).predict(
                    frames_to_features(frames, load_cnn())[None, ...], verbose=0)[0]
            if probs is None:
                st.error("Could not read this video. Try an .mp4 file.")
            else:
                top = int(np.argmax(probs))
                with right:
                    st.metric("Predicted action", classes[top])
                    st.caption(f"Confidence: {probs[top] * 100:.1f}% · top 5 shown below")
                    st.plotly_chart(prob_chart(probs, classes, highlight=classes[top], top_k=5),
                                    use_container_width=True)
                st.markdown("**Frames the model looked at**")
                show_frames(tmp)
