"""
ecgxai.py - shared code for the Explainable ECG project.
Used by BOTH the Kaggle training notebook and the Streamlit demo app,
so the demo behaves exactly like the trained system.

Contents: signal reading, pre-processing, labels + draft severity rules,
the three models (1D ResNet, InceptionTime, CNN-BiLSTM + attention),
sliding-window prediction, the stacking ensemble, Grad-CAM,
Integrated Gradients and SHAP (exact for the linear meta-learner).
"""
import ast
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GRADES = ["0 - Normal", "1 - Abnormal, not acute", "2 - Subacute", "3 - Acute"]
FS = 100          # Hz (PTB-XL 100 Hz version)
SIG_LEN = 1000    # 10 s
WIN = 250         # 2.5 s training crops
STRIDE = 125      # sliding-window step at test time

# ---------------------------------------------------------------- reading
def read_wfdb(base_path):
    """Read a WFDB record (format 16, as in PTB-XL) -> (T, 12) float32 in mV."""
    with open(base_path + ".hea") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    n_sig, n_samp = int(lines[0].split()[1]), int(lines[0].split()[3])
    gains, baselines = [], []
    for l in lines[1:1 + n_sig]:
        p = l.split()
        if p[1] != "16":
            raise ValueError("Only WFDB format 16 is supported")
        g = p[2].split("/")[0]
        if "(" in g:
            g, b = g.split("(")
            b = float(b.rstrip(")"))
        else:
            b = float(p[4]) if len(p) > 4 else 0.0
        gains.append(float(g)); baselines.append(b)
    raw = np.fromfile(base_path + ".dat", dtype="<i2")[: n_samp * n_sig].reshape(n_samp, n_sig)
    sig = (raw.astype(np.float32) - np.array(baselines, np.float32)) / np.array(gains, np.float32)
    return sig

# ---------------------------------------------------------- pre-processing
_B, _A = butter(3, [0.5, 40.0], btype="bandpass", fs=FS)

def bandpass(x):
    """Zero-phase 0.5-40 Hz Butterworth band-pass along time. x: (..., T, 12)."""
    return filtfilt(_B, _A, x, axis=-2).astype(np.float32)

def normalise(x, mean, std):
    return ((x - mean) / std).astype(np.float32)

def fix_length(sig, n=SIG_LEN):
    """Crop or zero-pad a (T, 12) signal to n samples."""
    if sig.shape[0] >= n:
        return sig[:n]
    return np.pad(sig, ((0, n - sig.shape[0]), (0, 0)))

# ----------------------------------------------------- labels and severity
def load_scp_map(scp_csv_path):
    import pandas as pd
    s = pd.read_csv(scp_csv_path, index_col=0)
    s = s[s.diagnostic == 1]
    return s["diagnostic_class"].to_dict()

def superclasses(scp_codes, scp_map):
    codes = ast.literal_eval(scp_codes) if isinstance(scp_codes, str) else scp_codes
    return sorted({scp_map[c] for c in codes if c in scp_map})

# Draft severity mapping (to be reviewed by a clinician), following the
# 4-level triage idea of van de Leur et al. [12]. An ECG takes its HIGHEST grade.
SEV3 = {"INJAS", "INJAL", "INJIN", "INJLA", "INJIL", "2AVB", "3AVB"}
SEV2 = {"CLBBB", "CRBBB", "WPW", "LNGQT", "AFIB", "AFLT", "SVTAC", "PSVT",
        "ISC_", "ISCAL", "ISCIN", "ISCIL", "ISCAS", "ISCLA", "ISCAN"}
MI_CODES = {"IMI", "ASMI", "ILMI", "AMI", "ALMI", "LMI", "IPLMI", "IPMI", "PMI"}
STADIUM_GRADE = {"Stadium I": 3, "Stadium I-II": 3, "Stadium II": 2,
                 "Stadium II-III": 2, "Stadium III": 1}

def severity_grade(scp_codes, infarction_stadium=None):
    codes = ast.literal_eval(scp_codes) if isinstance(scp_codes, str) else scp_codes
    codes = set(codes)
    grade = 0
    for c in codes:
        if c in ("NORM", "SR"):
            g = 0
        elif c in SEV3:
            g = 3
        elif c in MI_CODES:
            g = STADIUM_GRADE.get(infarction_stadium, 2)  # unknown stage -> 2 (conservative)
        elif c in SEV2:
            g = 2
        else:
            g = 1  # any other abnormal statement
        grade = max(grade, g)
    return grade

# ------------------------------------------------------------------ models
class Heads(nn.Module):
    """Two outputs: 5 diagnostic logits + CORAL ordinal severity (3 cumulative logits)."""
    def __init__(self, d, n_cls=len(CLASSES), n_grades=4, p=0.3):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.diag = nn.Linear(d, n_cls)
        self.sev_w = nn.Linear(d, 1, bias=False)              # shared weights (CORAL)
        self.sev_b = nn.Parameter(torch.zeros(n_grades - 1))  # one bias per threshold
    def forward(self, h):
        h = self.drop(h)
        return self.diag(h), self.sev_w(h) + self.sev_b

def gpool(z):  # global average + max pooling over time
    return torch.cat([z.mean(-1), z.amax(-1)], 1)

class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False); self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False); self.b2 = nn.BatchNorm1d(cout)
        self.sc = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + self.sc(x))

class ResNet1D(nn.Module):
    """M1: residual 1D CNN - local waveform shape."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 64, 15, 1, 7, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        self.layers = nn.Sequential(ResBlock(64, 64, 1), ResBlock(64, 64, 1),
                                    ResBlock(64, 128, 2), ResBlock(128, 128, 1),
                                    ResBlock(128, 128, 2), ResBlock(128, 128, 1))
        self.heads = Heads(256)
    def features(self, x):
        return self.layers(self.stem(x))          # last conv feature maps (for Grad-CAM)
    def forward(self, x):
        return self.heads(gpool(self.features(x)))

class InceptionModule(nn.Module):
    def __init__(self, cin, nf=32, ks=(39, 19, 9)):
        super().__init__()
        self.bott = nn.Conv1d(cin, nf, 1, bias=False) if cin > 1 else nn.Identity()
        self.convs = nn.ModuleList([nn.Conv1d(nf, nf, k, padding=k // 2, bias=False) for k in ks])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * 4)
    def forward(self, x):
        b = self.bott(x)
        y = torch.cat([c(b) for c in self.convs] + [self.pool(x)], 1)
        return F.relu(self.bn(y))

class InceptionTime(nn.Module):
    """M2: InceptionTime - patterns at several time scales."""
    def __init__(self, depth=6, nf=32):
        super().__init__()
        self.mods = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        cin, res_in = 12, 12
        for i in range(depth):
            self.mods.append(InceptionModule(cin, nf)); cin = nf * 4
            if i % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(res_in, cin, 1, bias=False), nn.BatchNorm1d(cin)))
                res_in = cin
        self.heads = Heads(cin * 2)
    def features(self, x):
        res = x; j = 0
        for i, m in enumerate(self.mods):
            x = m(x)
            if i % 3 == 2:
                x = F.relu(x + self.shortcuts[j](res)); res = x; j += 1
        return x
    def forward(self, x):
        return self.heads(gpool(self.features(x)))

class CNNBiLSTM(nn.Module):
    """M3: CNN front-end + 2-layer BiLSTM + temporal attention - beat-to-beat rhythm."""
    def __init__(self, ch=64, hid=64):
        super().__init__()
        def blk(cin, cout):
            return nn.Sequential(nn.Conv1d(cin, cout, 7, 1, 3, bias=False), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2))
        self.cnn = nn.Sequential(blk(12, ch), blk(ch, ch))
        self.lstm = nn.LSTM(ch, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.att = nn.Linear(2 * hid, 1)
        self.heads = Heads(2 * hid)
        self.last_attention = None
    def features(self, x):
        return self.cnn(x)
    def forward(self, x):
        z, _ = self.lstm(self.cnn(x).transpose(1, 2))      # (B, T', 2h)
        a = torch.softmax(self.att(z).squeeze(-1), dim=1)  # attention over time
        self.last_attention = a.detach()
        return self.heads((z * a.unsqueeze(-1)).sum(1))

MODEL_ZOO = {"resnet": ResNet1D, "inception": InceptionTime, "bilstm": CNNBiLSTM}
MODEL_LABEL = {"resnet": "M1 1D ResNet", "inception": "M2 InceptionTime", "bilstm": "M3 CNN-BiLSTM"}

# ------------------------------------------------------- losses and decoding
def coral_targets(grades, k=3):
    g = torch.as_tensor(grades).view(-1, 1)
    return (g > torch.arange(k, device=g.device).view(1, -1)).float()

def coral_loss(logits, grades):
    return F.binary_cross_entropy_with_logits(logits, coral_targets(grades).to(logits.device))

def grade_from_cum(cum_probs):
    """cum_probs: (..., 3) = P(grade > k). Returns integer grades and full 4-grade probabilities."""
    cum = np.clip(cum_probs, 0, 1)
    cum = np.minimum.accumulate(cum, axis=-1)  # enforce rank consistency
    p = np.concatenate([1 - cum[..., :1], cum[..., :-1] - cum[..., 1:], cum[..., -1:]], -1)
    return (cum > 0.5).sum(-1), p

# ----------------------------------------------------- windowed prediction
@torch.no_grad()
def predict_windows(model, X, device="cpu", batch=256):
    """X: (N, 1000, 12) normalised. Averages sigmoid outputs over 2.5 s windows (stride 1.25 s).
    Returns diag probs (N, 5) and cumulative severity probs (N, 3)."""
    model.eval()
    starts = list(range(0, SIG_LEN - WIN + 1, STRIDE))
    D, S = [], []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch]).transpose(1, 2).to(device)   # (B, 12, T)
        d_acc = 0; s_acc = 0
        for s in starts:
            d, sv = model(xb[:, :, s:s + WIN])
            d_acc = d_acc + torch.sigmoid(d); s_acc = s_acc + torch.sigmoid(sv)
        D.append((d_acc / len(starts)).cpu().numpy()); S.append((s_acc / len(starts)).cpu().numpy())
    return np.concatenate(D), np.concatenate(S)

# ------------------------------------------------------ stacking ensemble
class Ensemble:
    """Three base models + per-class logistic-regression meta-learner (weights from meta.json)."""
    def __init__(self, models, meta):
        self.models = models                     # dict name -> nn.Module (eval mode)
        self.names = meta["model_order"]
        self.w = np.array(meta["meta_coef"], np.float32)        # (5, 3)
        self.b = np.array(meta["meta_intercept"], np.float32)   # (5,)
        self.bg = np.array(meta["meta_background_mean"], np.float32)  # (5, 3) mean val probs
        self.thr = np.array(meta["thresholds"], np.float32)
        self.conf_cut = meta["confidence_cut"]

    def base_probs(self, Xn):
        out = {n: predict_windows(self.models[n], Xn) for n in self.names}
        P = np.stack([out[n][0] for n in self.names], -1)   # (N, 5, 3)
        S = np.stack([out[n][1] for n in self.names], -1)   # (N, 3, 3)
        return P, S

    def combine(self, P, S):
        logit = (P * self.w[None]).sum(-1) + self.b[None]    # (N, 5)
        prob = 1 / (1 + np.exp(-logit))
        grades, grade_p = grade_from_cum(S.mean(-1))
        disagreement = P.std(-1).mean(-1)                    # mean std across models
        confidence = np.clip(1 - 2 * disagreement, 0, 1)
        return prob, grades, grade_p, confidence

    def shap_values(self, P):
        """Exact SHAP for a linear meta-learner (what shap.LinearExplainer returns):
        phi_i = w_i * (p_i - E[p_i]), in log-odds units. Returns (N, 5, 3)."""
        return self.w[None] * (P - self.bg[None])

# --------------------------------------------------------- explanations
def grad_cam(model, x, cls):
    """1D Grad-CAM. x: (1, 12, T) tensor. Returns importance over time, length T, in [0, 1]."""
    model.eval()
    x = x.clone().requires_grad_(False)
    feats = model.features(x)
    feats.retain_grad()
    h = gpool(feats)
    logit = model.heads(h)[0][0, cls]
    model.zero_grad(); logit.backward()
    w = feats.grad.mean(-1, keepdim=True)                # average gradient per feature map
    cam = F.relu((w * feats).sum(1))[0]                  # weighted sum, ReLU
    cam = F.interpolate(cam[None, None], size=x.shape[-1], mode="linear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / (cam.max() + 1e-8)

def ensemble_logit_torch(models, names, w, b, x, cls):
    """Differentiable ensemble output (full 10 s input) for Integrated Gradients."""
    ps = [torch.sigmoid(models[n](x)[0][:, cls]) for n in names]
    return sum(float(w[cls, i]) * ps[i] for i in range(len(names))) + float(b[cls])

def integrated_gradients(models, names, w, b, x, cls, steps=32):
    """IG from a flat-line (zero) baseline to x. Returns (12, T) attributions."""
    for m in models.values():
        m.eval()
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (base + a * (x - base)).requires_grad_(True)
        out = ensemble_logit_torch(models, names, w, b, xi, cls).sum()
        g, = torch.autograd.grad(out, xi)
        total += g
    return ((x - base) * total / steps)[0].detach().cpu().numpy()

# ----------------------------------------------------------------- loading
def load_models(folder, device="cpu"):
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    models = {}
    for n in meta["model_order"]:
        m = MODEL_ZOO[n]()
        m.load_state_dict(torch.load(os.path.join(folder, f"{n}.pt"), map_location=device))
        models[n] = m.to(device).eval()
    return models, meta
