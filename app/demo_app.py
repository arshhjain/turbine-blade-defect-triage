"""
Demo page for the wind turbine blade defect + severity model.

Uses the SAME checkpoint and preprocessing as inference/predict_single.py.
It only adds an image picker (with true label and split) and charts of the
model's outputs. No training happens here.

Run (offline):
    python -m streamlit run app/demo_app.py
"""
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
import json
import random
from html import escape
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.predict_single import load_model                    # same loader as the CLI
from datasets.multitask_dataset import CLASSES
from datasets.augmentations import get_val_transforms               # resize 224 + ImageNet normalisation

CKPT = PROJECT_ROOT / "outputs" / "proposed_multitask_cbam" / "checkpoints" / "best.pth"
MANIFEST = PROJECT_ROOT / "data" / "splits" / "leak_free_split_manifest.json"
DEMO_IMAGES = ["4.jpg", "3.jpg", "10.jpg", "754.jpg"]               # test split; see RESULTS_VERIFIED.md
LOW_CONF = 0.60

# Illustrative routing table: a fixed rule applied AFTER the model, not something the model learned.
from app import dispatch_board as board
REPAIR_TEAM = board.TEAM_NAME
VIEW_INSPECT, VIEW_DISPATCH = "🔍  Inspect one image", "📋  Dispatch board (simulation)"
SEV_NAME = {1: "Low", 2: "Moderate", 3: "Elevated", 4: "High", 5: "Critical"}   # same words as predict_single.py
SEV_COLOR = {1: "#3A9D5D", 2: "#9BBF3B", 3: "#F2A541", 4: "#E07B39", 5: "#C8553D"}
TEAL, DARK, AMBER, MUTED = "#1F7A8C", "#0F2A36", "#F2A541", "#B8C7CC"


def nice(name):
    return name.replace("_", " ").capitalize()


# ----------------------------------------------------------------- model + data
@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(CKPT, device)
    return model, device, config.get("image_size", 224)


@st.cache_data
def get_manifest():
    m = json.load(open(MANIFEST, encoding="utf-8"))
    rows = {}
    for split in ("train", "val", "test"):
        for r in m[split]:
            rows[r["image_id"]] = {"split": split, "true_class": r["dominant_class"],
                                   "true_severity": r["severity_level"], "path": r["image_path"],
                                   "defects": r.get("defects", [])}
    return rows


def predict(rgb, model, device, image_size):
    x = get_val_transforms(image_size=image_size)(image=rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    probs = F.softmax(out["type_logits"], dim=-1)[0].cpu().numpy()            # 6 class probabilities
    ord_probs = torch.sigmoid(out["ordinal_logits"])[0].cpu().numpy()        # P(severity > k), k = 1..4
    level = 1 + int((ord_probs > 0.5).sum())                                 # ordinal decoding
    scalar = float(out["severity_scalar"][0].item())                         # 1 + 4*sigmoid(z)
    return probs, ord_probs, level, scalar


def draw_boxes(rgb, defects):
    """Ground-truth boxes from the dataset annotation (NOT model output)."""
    img = rgb.copy()
    t = max(2, img.shape[0] // 256)
    for d in defects:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (242, 165, 65), t)
        cv2.putText(img, nice(d["class"]), (x1 + 4, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3 * t, (242, 165, 65), max(1, t - 1))
    return img


# ----------------------------------------------------------------- styling
CSS = f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px;}}
#MainMenu, footer {{visibility: hidden;}}
.hero {{background: linear-gradient(120deg, {DARK} 0%, #16404F 60%, {TEAL} 100%); color: #fff;
        border-radius: 14px; padding: 22px 28px; margin-bottom: 18px;}}
.hero h1 {{color: #fff; font-size: 1.75rem; margin: 0 0 4px 0; padding: 0; font-weight: 700;}}
.hero p {{color: #D5E4E8; margin: 0 0 14px 0; font-size: 0.98rem;}}
.chips {{display: flex; flex-wrap: wrap; gap: 8px;}}
.chip {{background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.22); color: #fff;
        border-radius: 999px; padding: 3px 12px; font-size: 0.82rem;}}
.chip b {{color: {AMBER};}}
.card {{background: #fff; border: 1px solid #DCE5E8; border-radius: 14px; padding: 18px 20px;
        margin-bottom: 14px; box-shadow: 0 1px 3px rgba(15,42,54,0.05);}}
.label {{text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; color: #5B7480; font-weight: 600;}}
.big {{font-size: 2rem; font-weight: 700; color: {DARK}; line-height: 1.15; margin: 2px 0 6px 0;}}
.sub {{color: #5B7480; font-size: .9rem;}}
.bar {{height: 10px; background: #E6EEF0; border-radius: 99px; overflow: hidden; margin: 8px 0 4px 0;}}
.bar > div {{height: 100%; border-radius: 99px;}}
.sev {{display: flex; gap: 6px; margin: 10px 0 6px 0;}}
.sev div {{flex: 1; height: 30px; border-radius: 7px; display: flex; align-items: center;
           justify-content: center; font-weight: 700; font-size: .85rem;}}
.pill {{display: inline-block; border-radius: 999px; padding: 3px 11px; font-size: .8rem; font-weight: 600;
        margin: 0 6px 6px 0;}}
.ok {{background: #E3F3E8; color: #24683C;}}
.bad {{background: #FBE4DF; color: #9B3A26;}}
.warn {{background: #FDF1DC; color: #8A5A0B;}}
.info {{background: #E2EFF2; color: #1B5E6C;}}
.route {{display: flex; align-items: center; gap: 14px;}}
.route .icon {{font-size: 1.7rem;}}
.small {{font-size: .78rem; color: #7B909A;}}
</style>
"""

st.set_page_config(page_title="Blade defect triage", page_icon="🌬️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

model, device, image_size = get_model()
manifest = get_manifest()
test_ids = sorted([k for k, v in manifest.items() if v["split"] == "test"], key=lambda s: int(s.split(".")[0]))
all_ids = sorted(manifest, key=lambda s: int(s.split(".")[0]))

st.markdown(f"""
<div class="hero">
  <h1>Wind turbine blade: defect type + severity</h1>
  <p>Triage of blades already reported as damaged: the type picks the repair team, the severity sets the priority.</p>
  <div class="chips">
    <span class="chip">ResNet50 + CBAM, 3 heads</span>
    <span class="chip">Test accuracy <b>92.5%</b></span>
    <span class="chip">Severity MAE <b>0.35</b> levels</span>
    <span class="chip">159 held-out test images</span>
    <span class="chip">Running on <b>{device.type.upper()}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- sidebar
def pick_random_test():
    st.session_state["test_pick"] = random.choice(test_ids)


with st.sidebar:
    st.markdown("### Choose an image")
    source = st.radio("Source", ["Demo images (test split)", "Any test-split image", "Any dataset image", "Upload my own"],
                      label_visibility="collapsed")
    upload = None
    if source == "Demo images (test split)":
        image_id = st.selectbox("Image", DEMO_IMAGES,
                                format_func=lambda k: f"{k}  ({nice(manifest[k]['true_class'])})")
    elif source == "Any test-split image":
        image_id = st.selectbox("Test image (never seen in training)", test_ids, key="test_pick",
                                format_func=lambda k: f"{k}  ({nice(manifest[k]['true_class'])})")
        st.button("🎲  Random test image", on_click=pick_random_test, width="stretch")
    elif source == "Any dataset image":
        image_id = st.selectbox("Image", all_ids, format_func=lambda k: f"{k}  [{manifest[k]['split']}]")
    else:
        image_id = None
        upload = st.file_uploader("JPG or PNG", type=["jpg", "jpeg", "png"])

    show_boxes = False
    if image_id is not None:
        show_boxes = st.toggle("Show annotated boxes", value=False,
                               help="Boxes drawn by the dataset's human annotators. The model does NOT draw boxes; "
                                    "it only outputs a type and a severity for the whole image.")
    st.divider()
    st.markdown(f"<div class='small'>Checkpoint<br><code>{CKPT.relative_to(PROJECT_ROOT).as_posix()}</code><br><br>"
                "Same weights and preprocessing as <code>inference/predict_single.py</code>.</div>",
                unsafe_allow_html=True)

view = st.segmented_control("View", [VIEW_INSPECT, VIEW_DISPATCH], default=VIEW_INSPECT, key="view",
                            label_visibility="collapsed") or VIEW_INSPECT

# ----------------------------------------------------------------- load image
rgb = None
if image_id is not None:
    meta = manifest[image_id]
    bgr = cv2.imread(str(PROJECT_ROOT / meta["path"]))
else:
    meta = None
    if upload is None:
        st.info("Upload a JPG or PNG in the sidebar.")
        if view == VIEW_INSPECT:
            st.stop()
        bgr = None
    else:
        bgr = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read that file as an image.")
            st.stop()
if bgr is not None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    probs, ord_probs, level, scalar = predict(rgb, model, device, image_size)
    pred_idx = int(probs.argmax())
    pred = CLASSES[pred_idx]
    conf = float(probs[pred_idx])


# ----------------------------------------------------------------- dispatch board (simulation)
def predict_image_id(iid):
    img = cv2.cvtColor(cv2.imread(str(PROJECT_ROOT / manifest[iid]["path"])), cv2.COLOR_BGR2RGB)
    p, _, lv, _ = predict(img, model, device, image_size)
    return img, CLASSES[int(p.argmax())], float(p.max()), lv


if view == VIEW_DISPATCH:
    ss = st.session_state
    ss.setdefault("tickets", [])
    ss.setdefault("seq", 0)

    def add(items):
        """items: list of (image_id, rgb, pred, conf, level). Records old queue positions for the animation."""
        prev = board.snapshot(ss.tickets)
        new = []
        for iid, img, p, c, lv in items:
            ss.seq += 1
            ss.tickets.append(board.make_ticket(ss.seq, iid, img, p, c, lv, LOW_CONF))
            new.append(ss.seq)
        ss.anim = {"prev": prev, "new": new}
        last = ss.tickets[-1]
        dest = "Human review" if last["lane"] == board.HUMAN else REPAIR_TEAM[last["pred"]]
        ss.last_msg = (f"{last['image']}: {nice(last['pred'])} {last['conf']*100:.0f}%, level {last['level']} → {dest}"
                       + (f"  (and {len(items)-1} more)" if len(items) > 1 else ""))

    st.markdown("<div class='card' style='padding:10px 16px;'><span class='pill warn'>Simulation</span>"
                "<span class='sub'>Routing (type → team), queue order (severity, then arrival) and the "
                f"&lt; {int(LOW_CONF*100)}% human-review rule are fixed rules applied to the model's output. "
                "Nothing on this board is learned.</span></div>", unsafe_allow_html=True)

    on_board = {t["image"] for t in ss.tickets}
    b1, b2, b3, info = st.columns([1.25, 1.25, 0.7, 2.3])
    current_label = image_id or "uploaded image"
    if b1.button(f"➤  Dispatch {current_label}", type="primary", width="stretch", disabled=rgb is None):
        if image_id is not None and image_id in on_board:
            st.toast(f"{image_id} is already on the board.")
        else:
            tag = image_id or f"upload {sum(1 for t in ss.tickets if t['image'].startswith('upload')) + 1}"
            add([(tag, rgb, pred, conf, level)])
    if b2.button("🎲  Dispatch 5 random", width="stretch"):
        pool = [i for i in test_ids if i not in on_board]
        picks = random.sample(pool, min(5, len(pool)))
        if picks:
            with st.spinner("Running the model on 5 test images..."):
                add([(i, *predict_image_id(i)) for i in picks])
    if b3.button("Reset", width="stretch"):
        ss.tickets, ss.seq, ss.anim, ss.last_msg = [], 0, None, ""
    if rgb is not None:
        info.markdown(f"<div class='sub' style='padding-top:6px;'>Current image: <b>{escape(current_label)}</b> → "
                      f"{escape(nice(pred))} {conf*100:.0f}%, level {level}</div>", unsafe_allow_html=True)

    if ss.get("last_msg"):
        st.caption(f"Last dispatch: {ss.last_msg}")
    board.render(ss.tickets, ss.get("anim"), SEV_COLOR, nice, LOW_CONF)
    ss.anim = None          # play the animation once, not on every rerun
    st.stop()

# ----------------------------------------------------------------- layout
left, right = st.columns([1, 1.1], gap="large")

with left:
    shown = draw_boxes(rgb, meta["defects"]) if (show_boxes and meta) else rgb
    st.image(shown, width="stretch",
             caption=f"{image_id or 'Uploaded image'}  ·  {rgb.shape[1]}×{rgb.shape[0]}, "
                     f"resized to {image_size}×{image_size} for the model"
                     + ("  ·  orange boxes = human annotation, not model output" if (show_boxes and meta) else ""))

    if meta is not None:
        split_pill = {"test": ("ok", "TEST split · never seen in training"),
                      "val": ("warn", "VALIDATION split · used to pick the best epoch"),
                      "train": ("bad", "TRAIN split · the model learned from this image")}[meta["split"]]
        ok_type = pred == meta["true_class"]
        diff = abs(level - meta["true_severity"])
        type_pill = ("ok", "Type correct") if ok_type else ("bad", "Type WRONG")
        sev_pill = ("ok", "Severity exact") if diff == 0 else (("warn" if diff == 1 else "bad"), f"Severity off by {diff}")
        largest = max((d["area_ratio"] for d in meta["defects"]), default=0.0)
        st.markdown(f"""
<div class="card">
  <div class="label">Ground truth</div>
  <div style="margin:6px 0 8px 0;font-size:1.05rem;"><b>{escape(nice(meta['true_class']))}</b>, severity {meta['true_severity']}
    <span class="small">(my proxy rule: type base level, ±1 by box size; largest box = {largest*100:.1f}% of image)</span></div>
  <span class="pill {split_pill[0]}">{split_pill[1]}</span>
  <span class="pill {type_pill[0]}">{type_pill[1]}</span>
  <span class="pill {sev_pill[0]}">{sev_pill[1]}</span>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="card"><span class="pill warn">Uploaded image · no true label</span>
<div class="sub" style="margin-top:6px;">The model <b>assumes a defect is present</b>: a healthy blade will still be given
one of the 6 defect types. Photos unlike the Figshare drone images may give meaningless answers.</div></div>""",
                    unsafe_allow_html=True)

with right:
    conf_color = TEAL if conf >= LOW_CONF else "#C8553D"
    st.markdown(f"""
<div class="card">
  <div class="label">Predicted defect type</div>
  <div class="big">{escape(nice(pred))}</div>
  <div class="sub">Confidence <b style="color:{conf_color}">{conf*100:.1f}%</b></div>
  <div class="bar"><div style="width:{conf*100:.1f}%;background:{conf_color};"></div></div>
  {"<span class='pill bad'>Low confidence (&lt; 60%): send to a human inspector</span>" if conf < LOW_CONF else ""}
</div>""", unsafe_allow_html=True)

    cells = "".join(
        f"<div style='background:{SEV_COLOR[k] if k <= level else '#E6EEF0'};"
        f"color:{'#fff' if k <= level else '#8FA3AB'};'>{k}</div>" for k in range(1, 6))
    st.markdown(f"""
<div class="card">
  <div class="label">Severity</div>
  <div class="big" style="color:{SEV_COLOR[level]}">Level {level} <span style="font-size:1.1rem;font-weight:600;">· {SEV_NAME[level]}</span></div>
  <div class="sev">{cells}</div>
  <div class="sub">Ordinal head: 1 + {int((ord_probs > 0.5).sum())} "yes" answers = {level}. Continuous head: {scalar:.2f}</div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="card">
  <div class="label">Route to</div>
  <div class="route"><span class="icon">🛠️</span>
    <div><div style="font-size:1.15rem;font-weight:700;">{escape(REPAIR_TEAM[pred])} team</div>
    <div class="sub">Priority: severity {level} of 5</div></div></div>
  <div class="small" style="margin-top:8px;">Fixed lookup rule applied after the model, not learned.</div>
</div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------- details
st.markdown("#### Under the hood")
tab1, tab2, tab3 = st.tabs(["Type probabilities", "Severity questions", "About this model"])

with tab1:
    df = pd.DataFrame({"type": [nice(c) for c in CLASSES], "p": probs.astype(float)})
    df["predicted"] = df["type"] == nice(pred)
    base = alt.Chart(df).encode(
        y=alt.Y("type:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=13, labelLimit=200)),
        x=alt.X("p:Q", title="softmax probability", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], grid=False)))
    bars = base.mark_bar(cornerRadiusEnd=5, height=22).encode(
        color=alt.condition("datum.predicted", alt.value(TEAL), alt.value(MUTED)))
    text = base.mark_text(align="left", dx=5, fontSize=12, color=DARK).encode(text=alt.Text("p:Q", format=".1%"))
    st.altair_chart((bars + text).properties(height=240), width="stretch")
    st.caption("The 6 probabilities sum to 100%. The model must pick one of these 6 types: there is no 'healthy' class.")

with tab2:
    od = pd.DataFrame({"question": [f"Is severity > {k}?" for k in range(1, 5)], "p": ord_probs.astype(float)})
    od["yes"] = od["p"] > 0.5
    base = alt.Chart(od).encode(
        y=alt.Y("question:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13)),
        x=alt.X("p:Q", title="P(yes)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], grid=False)))
    bars = base.mark_bar(cornerRadiusEnd=5, height=22).encode(
        color=alt.condition("datum.yes", alt.value(TEAL), alt.value(MUTED)))
    text = base.mark_text(align="left", dx=5, fontSize=12, color=DARK).encode(text=alt.Text("p:Q", format=".1%"))
    rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color="#C8553D", strokeDash=[5, 4]).encode(x="x:Q")
    st.altair_chart((bars + text + rule).properties(height=190), width="stretch")
    st.caption(f"Each bar is one sigmoid output. Above the dashed 50% line counts as 'yes'. "
               f"Level = 1 + number of yes = 1 + {int(od['yes'].sum())} = {level}.")

with tab3:
    a, b = st.columns(2)
    a.markdown("""
**Pipeline**
- Image resized to 3 × 224 × 224, ImageNet normalisation
- ResNet50 (ImageNet weights, all layers fine-tuned)
- CBAM attention on the 7 × 7 × 2048 feature map
- Avg + max pool → 4096 → FC 512 (LayerNorm, ReLU, dropout 0.3)
- Heads: type (focal loss), ordinal severity (4 × BCE), continuous severity (Smooth L1)
""")
    b.markdown("""
**Limitations**
- Closed set: always outputs a defect type, even for a healthy blade
- Severity labels are my proxy rule; their size part depends on camera distance
- One seed, 159 test images: differences of 1–3 images are noise
- One label per image, though an image can hold several defects
""")
    st.caption("All numbers: RESULTS_VERIFIED.md. Data: public Figshare WT blade defect dataset.")
