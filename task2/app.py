# app.py — MNIST Digit Classifier (PyTorch + robust preprocessing)
from pathlib import Path
import os, sys
import numpy as np
from collections import deque
from PIL import Image, ImageOps, ImageFilter
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms

# ---------- Always render something first ----------
st.title("MNIST Digit Classifier (PyTorch)")
st.caption("If you can read this, the app rendered.")
st.write("Python:", sys.executable)
st.write("CWD:", os.getcwd())

# ---------- Locate weights ----------
HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE / "mnist_cnn.pt",
    HERE / "models" / "mnist_cnn.pt",
    HERE.parent / "task2" / "mnist_cnn.pt",
    HERE.parent / "task2" / "models" / "mnist_cnn.pt",
]
WEIGHTS = next((p for p in CANDIDATES if p.exists()), None)
st.write("Searching weights:", [str(p) for p in CANDIDATES])
st.write("Found weights:", str(WEIGHTS) if WEIGHTS else "None")

# ---------- Model ----------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*7*7,128)
        self.fc2 = nn.Linear(128,10)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = self.dropout(x); x = torch.flatten(x,1)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        return self.fc2(x)

@st.cache_resource
def load_model():
    if WEIGHTS is None:
        raise FileNotFoundError("Weights file not found. Put 'mnist_cnn.pt' in task2/, task2/models/, or repo root.")
    m = CNN()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    m.eval()
    return m

# Load model now (so failures show on page)
try:
    model = load_model()
    st.success(f"Model loaded: {WEIGHTS.name}")
except Exception as e:
    st.error("Model failed to load.")
    st.exception(e)
    st.stop()

# ---------- Preprocessing helpers (SciPy-free) ----------
def _otsu_threshold(arr: np.ndarray) -> int:
    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = (omega * (1 - omega))
    denom[denom == 0] = 1e-9
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2[np.isnan(sigma_b2)] = 0
    return int(np.argmax(sigma_b2))

def _largest_component_mask(bw: np.ndarray) -> np.ndarray:
    # bw: uint8 {0,1}
    h, w = bw.shape
    visited = np.zeros_like(bw, dtype=bool)
    best_area, best_coords = 0, None
    for y in range(h):
        for x in range(w):
            if bw[y, x] == 1 and not visited[y, x]:
                q = deque([(y, x)])
                visited[y, x] = True
                coords = [(y, x)]
                while q:
                    cy, cx = q.popleft()
                    for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                        if 0 <= ny < h and 0 <= nx < w and bw[ny, nx]==1 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                            coords.append((ny, nx))
                # reject ultra-thin “line” components by aspect ratio
                ys, xs = zip(*coords)
                y0, y1 = min(ys), max(ys)
                x0, x1 = min(xs), max(xs)
                hbb, wbb = (y1 - y0 + 1), (x1 - x0 + 1)
                aspect = max(hbb, wbb) / (min(hbb, wbb) + 1e-6)
                if len(coords) > best_area and aspect < 6.0:
                    best_area, best_coords = len(coords), coords
    mask = np.zeros_like(bw, dtype=np.uint8)
    if best_coords:
        ys, xs = zip(*best_coords)
        mask[np.array(ys), np.array(xs)] = 1
    return mask

def _center_pad_resize(img: Image.Image, size=28) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)  # black background
    canvas.paste(img, ((side - w)//2, (side - h)//2))
    return canvas.resize((size, size), Image.BILINEAR)

def _prepare_once(img: Image.Image, invert: bool, thr_bias: int, use_components: bool) -> Image.Image:
    """Make a MNIST-like 28x28 tile.
    KEY FIX: when not inverted, digits are darker than paper → use (arr < thr).
              when inverted, digits are bright → use (arr > thr).
    """
    # 1) grayscale + contrast
    g = img.convert("L")
    g = ImageOps.autocontrast(g)

    # 2) optional invert
    if invert:
        g = ImageOps.invert(g)

    # 3) light denoise
    g = g.filter(ImageFilter.MedianFilter(3))

    # 4) threshold (Otsu + bias)
    arr = np.array(g)
    thr = int(np.clip(_otsu_threshold(arr) + int(thr_bias), 0, 255))

    # --- POLARITY-AWARE FOREGROUND SELECTION (the actual fix) ---
    if invert:
        # strokes are bright after inversion → keep pixels ABOVE threshold
        bw = (arr > thr).astype(np.uint8)
    else:
        # original photo: strokes are dark → keep pixels BELOW threshold
        bw = (arr < thr).astype(np.uint8)

    # 5) optionally keep only the largest component (drop ruled lines/background)
    if use_components:
        mask = _largest_component_mask(bw)
        if mask.sum() > 0:
            bw = mask

    # 6) thicken strokes a bit (no SciPy)
    thick = Image.fromarray((bw * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))

    # 7) tight crop → center-pad → resize 28x28
    nz = np.argwhere(np.array(thick) > 0)  # list of (y, x)
    if nz.size == 0:
        return _center_pad_resize(g)  # fallback if nothing detected

    (y0, x0) = nz.min(axis=0)
    (y1, x1) = nz.max(axis=0)
    crop = thick.crop((x0, y0, x1 + 1, y1 + 1))

    return _center_pad_resize(crop)


def preprocess_candidates(img: Image.Image, thr_bias: int, use_components: bool):
    # Return both polarities so the model (or user) can choose
    cand_noninv = _prepare_once(img, invert=False, thr_bias=thr_bias, use_components=use_components)
    cand_inv    = _prepare_once(img, invert=True,  thr_bias=thr_bias, use_components=use_components)
    return cand_noninv, cand_inv

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- UI controls ----------
thr_bias = st.slider("Threshold bias (add to Otsu)", -40, 40, 0, 1)
use_components = st.checkbox("Use largest-component cleanup", value=True)
manual_pick = st.radio("Polarity selection", ["Auto (by model)", "Non-inverted", "Inverted"], index=0)

# ---------- Inference ----------
file = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png","jpg","jpeg"])
if file:
    raw = Image.open(file)
    st.image(raw, caption="Input", width=160)

    c1, c2 = preprocess_candidates(raw, thr_bias=thr_bias, use_components=use_components)
    col1, col2 = st.columns(2)
    with col1: st.image(c1, caption="Non-inverted", width=160)
    with col2: st.image(c2, caption="Inverted", width=160)

    # Decide which candidate to use
    if manual_pick == "Non-inverted":
        chosen = c1
    elif manual_pick == "Inverted":
        chosen = c2
    else:
        with torch.no_grad():
            p1 = torch.softmax(model(tfm(c1).unsqueeze(0)), dim=1)[0]
            p2 = torch.softmax(model(tfm(c2).unsqueeze(0)), dim=1)[0]
        chosen = c1 if float(p1.max()) >= float(p2.max()) else c2

    st.image(chosen, caption="Used for prediction", width=160)

    with torch.no_grad():
        logits = model(tfm(chosen).unsqueeze(0))
        prob = torch.softmax(logits, dim=1)[0]
        pred = int(prob.argmax().item())

    st.subheader(f"Prediction: {pred}")
    st.write({str(i): float(prob[i]) for i in range(10)})
