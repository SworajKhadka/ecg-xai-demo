"""
Streamlit demo - Explainable AI-Based Detection and Severity Grading of
Cardiovascular Abnormalities using 12-lead ECG signals.

Run:  streamlit run app.py
Needs: models/ (resnet.pt, inception.pt, bilstm.pt, meta.json) from the Kaggle notebook,
       and ecgxai.py in the same folder.
"""
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy.signal import resample_poly

import ecgxai as E

st.set_page_config(page_title="Explainable ECG Analysis", page_icon="🫀", layout="wide")
NAVY, RED, MUTED = "#14213D", "#C8323A", "#4A5568"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
SAMPLE_DIR = os.path.join(HERE, "samples")


@st.cache_resource
def get_ensemble():
    models, meta = E.load_models(MODEL_DIR)
    return E.Ensemble(models, meta), meta


def read_fs(hea_path):
    with open(hea_path) as f:
        first = f.readline().split()
    return float(first[2]) if len(first) > 2 else 100.0


def load_record(base):
    sig = E.read_wfdb(base)
    fs = read_fs(base + ".hea")
    if abs(fs - 500) < 1:                       # 500 Hz record -> 100 Hz
        sig = resample_poly(sig, 1, 5, axis=0).astype(np.float32)
    elif abs(fs - 100) > 1:
        raise ValueError(f"Sampling rate {fs} Hz is not supported (use 100 or 500 Hz PTB-XL records).")
    if sig.shape[1] != 12:
        raise ValueError("Record must have 12 leads.")
    return E.fix_length(sig)


def analyse(sig, ens, meta):
    filt = E.bandpass(sig)                                                     # (1000, 12) in mV
    xn = E.normalise(filt, np.array(meta["norm_mean"], np.float32), np.array(meta["norm_std"], np.float32))
    P, S = ens.base_probs(xn[None])
    prob, grades, grade_p, conf = ens.combine(P, S)
    return filt, xn, P[0], prob[0], int(grades[0]), grade_p[0], float(conf[0])


def plot_ecg(sig, heat=None, per_lead=False, title=""):
    t = np.arange(E.SIG_LEN) / E.FS
    fig, axs = plt.subplots(6, 2, figsize=(14, 7), sharex=True)
    fig.patch.set_facecolor("#FBFAF8")
    for j in range(12):
        ax = axs[j % 6, j // 6]
        ax.set_facecolor("#FBFAF8")
        y = sig[:, j]
        lo, hi = y.min() - 0.1, y.max() + 0.1
        if heat is not None:
            h = heat[j] if per_lead else heat
            ax.imshow(h[None, :], aspect="auto", cmap="Reds", alpha=0.6, extent=[0, 10, lo, hi], vmin=0, vmax=1)
        ax.plot(t, y, color=NAVY, lw=0.8)
        ax.set_ylim(lo, hi); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(-0.01, 0.5, E.LEADS[j], transform=ax.transAxes, ha="right", va="center", color=RED, fontweight="bold")
    for ax in axs[-1]:
        ax.set_xlabel("seconds", color=MUTED)
    if title:
        fig.suptitle(title, color=NAVY)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- page
st.title("🫀 Explainable ECG Analysis")
st.caption("Detection and severity grading of cardiovascular abnormalities from a 12-lead ECG, "
           "with a three-model deep learning ensemble and explanations. Research decision-support prototype, not a medical device.")

if not os.path.exists(os.path.join(MODEL_DIR, "meta.json")):
    st.error("No trained models found. Unzip ecg_models.zip from Kaggle so that this folder contains models/meta.json.")
    st.stop()
ens, meta = get_ensemble()

SAMPLE_INFO = {}
if os.path.exists(os.path.join(SAMPLE_DIR, "samples.json")):
    SAMPLE_INFO = json.load(open(os.path.join(SAMPLE_DIR, "samples.json")))

with st.sidebar:
    st.header("1. Choose an ECG")
    samples = sorted({f[:-4] for f in os.listdir(SAMPLE_DIR) if f.endswith(".hea")}) if os.path.isdir(SAMPLE_DIR) else []
    mode = st.radio("Source", ["Sample record", "Upload WFDB files"], index=0 if samples else 1)
    base, truth = None, None
    if mode == "Sample record" and samples:
        def _fmt(stem):
            i = SAMPLE_INFO.get(stem)
            return f"{stem}  ·  {'/'.join(i['labels'])}, grade {i['severity']}" if i else stem
        choice = st.selectbox("PTB-XL record (label = cardiologist's)", samples, format_func=_fmt)
        base = os.path.join(SAMPLE_DIR, choice)
        truth = SAMPLE_INFO.get(choice)
    else:
        up = st.file_uploader("Upload the .hea and .dat files of one record", type=["hea", "dat"], accept_multiple_files=True)
        heas = [f for f in up if f.name.endswith(".hea")] if up else []
        dats = [f for f in up if f.name.endswith(".dat")] if up else []
        if heas and dats:
            tmp = tempfile.mkdtemp()
            stem = heas[0].name[:-4]
            for f in heas + dats:
                open(os.path.join(tmp, stem + f.name[-4:]), "wb").write(f.getbuffer())
            base = os.path.join(tmp, stem)
    st.header("2. Explanation")
    method = st.radio("Method", ["Grad-CAM (M1 ResNet)", "Grad-CAM (M2 InceptionTime)", "Integrated Gradients (ensemble)"])

if base is None:
    st.info("Choose a sample record or upload a record's .hea and .dat files in the sidebar.")
    st.stop()

try:
    sig = load_record(base)
except Exception as ex:
    st.error(f"Could not read the record: {ex}")
    st.stop()

with st.spinner("Running the three models and the meta-learner..."):
    filt, xn, P, prob, grade, grade_p, conf = analyse(sig, ens, meta)

thr = np.array(meta["thresholds"])
positive = [c for k, c in enumerate(E.CLASSES) if prob[k] >= thr[k]]
top = E.CLASSES[int(np.argmax(prob))]

c1, c2, c3 = st.columns(3)
c1.caption("Diagnosis")
c1.markdown("#### " + (", ".join(E.CLASS_NAMES[c] for c in positive) if positive else E.CLASS_NAMES[top]))
c2.metric("Severity grade", E.GRADES[grade])
c3.metric("Confidence", f"{conf:.0%}", "Flag for clinician review" if conf < meta["confidence_cut"] else "OK",
          delta_color="inverse" if conf < meta["confidence_cut"] else "normal")

if truth:
    pred_set = set(positive) if positive else {top}
    match = "✅ matches" if pred_set == set(truth["labels"]) else ("🟡 partly matches" if pred_set & set(truth["labels"]) else "❌ differs")
    st.info(f"**Cardiologist label (PTB-XL ground truth):** {', '.join(E.CLASS_NAMES[c] for c in truth['labels'])} · "
            f"severity grade {truth['severity']} · fold {truth['fold']}"
            f"{' (test set: never seen in training)' if truth['fold'] == 10 else ''} — prediction {match}.")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Diagnosis probabilities (stacked ensemble)")
    fig, ax = plt.subplots(figsize=(6, 2.8))
    names = [E.CLASS_NAMES[c] for c in E.CLASSES]
    ax.barh(names, prob, color=[RED if prob[k] >= thr[k] else "#9FB2C8" for k in range(5)])
    ax.scatter(thr, names, marker="|", s=300, color=NAVY, label="decision threshold")
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.legend(loc="lower right", fontsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(); st.pyplot(fig)
with right:
    st.subheader("Each model's opinion")
    df = pd.DataFrame(P, index=[E.CLASS_NAMES[c] for c in E.CLASSES],
                      columns=[E.MODEL_LABEL[n] for n in meta["model_order"]])
    df["Ensemble"] = prob
    st.dataframe(df.style.format("{:.2f}").background_gradient(cmap="Reds", vmin=0, vmax=1), width="stretch")
    st.caption("Severity probabilities: " + " · ".join(f"G{g}: {p:.2f}" for g, p in enumerate(grade_p)))

st.subheader("Explanation: where the evidence is")
cls_name = st.selectbox("Explain which class?", [E.CLASS_NAMES[c] for c in E.CLASSES],
                        index=E.CLASSES.index(positive[0] if positive else top))
k = [E.CLASS_NAMES[c] for c in E.CLASSES].index(cls_name)
x = torch.as_tensor(xn[None]).transpose(1, 2)
with st.spinner("Computing explanation..."):
    if method.startswith("Grad-CAM (M1"):
        heat, per_lead = E.grad_cam(ens.models["resnet"], x, k), False
    elif method.startswith("Grad-CAM (M2"):
        heat, per_lead = E.grad_cam(ens.models["inception"], x, k), False
    else:
        ig = np.abs(E.integrated_gradients(ens.models, ens.names, ens.w, ens.b, x, k))
        heat, per_lead = ig / (ig.max() + 1e-8), True
st.pyplot(plot_ecg(filt, heat, per_lead, f"{method} for '{cls_name}' (red = more important)"))
if E.CLASSES[k] == "NORM":
    st.caption("For a normal ECG the evidence is spread across every beat (there is no single abnormal spot). "
               "Pick an abnormal record to see the heatmap focus on specific beats and leads.")

st.subheader("SHAP: which model drove the decision")
phi = ens.shap_values(P[None])[0, k]
fig, ax = plt.subplots(figsize=(7, 2.2))
ax.barh([E.MODEL_LABEL[n] for n in meta["model_order"]], phi, color=[NAVY if v >= 0 else "#9FB2C8" for v in phi])
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel(f"contribution to '{cls_name}' (log-odds); ensemble probability {prob[k]:.2f}")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(); st.pyplot(fig)
st.caption("Exact SHAP values for the linear meta-learner (identical to shap.LinearExplainer): "
           "contribution = weight × (model probability − its average probability on the validation fold).")

res_path = os.path.join(HERE, "results.json")
if os.path.exists(res_path):
    with st.expander("Model performance on PTB-XL fold 10 (test set)"):
        res = json.load(open(res_path))
        st.dataframe(pd.DataFrame(res["ablation"]).set_index("Method").style.format("{:.4f}"), width="stretch")
        st.dataframe(pd.DataFrame(res["severity"]).set_index("index").style.format("{:.4f}"), width="stretch")
        lo, hi = res["stacking_auroc_ci95"]
        st.caption(f"Stacking macro AUROC 95% bootstrap CI: {lo:.4f} – {hi:.4f} · test ECGs: {res['n_test']}")
