import os, time, tempfile, zipfile
from collections import defaultdict, deque
from datetime import datetime

import streamlit as st
import cv2
import pandas as pd
import plotly.graph_objects as go
import gdown

# ===================== CONFIG =====================
MODEL_PATH = "models/yolov8_ppe.pt"
MODEL_URL = "https://drive.google.com/uc?id=1qLB4ZjijrpNdHcphQftVudm8y4SOZDoL"

SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
DETECT_EVERY = 5
PERSIST_FRAMES = 5

CSV_HEADER = [
    "timestamp","worker_id","class_name",
    "confidence","x1","y1","x2","y2","snapshot"
]

CLASS_COLORS = {
    "Hardhat": (0,255,0),
    "Mask": (0,200,0),
    "Safety Vest": (0,180,0),
    "Gloves": (0,160,0),
    "Goggles": (0,140,0),
    "Shoes": (0,120,0),
    "NO-Hardhat": (255,60,60),
    "NO-Mask": (255,60,60),
    "NO-Safety Vest": (255,60,60)
}

# ===================== MODEL DOWNLOAD =====================
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with st.spinner("⬇️ Downloading PPE model (first run only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

MODEL_PATH = download_model()

# ===================== PAGE =====================
st.set_page_config(
    page_title="PPE AI Safety Monitoring",
    page_icon="🦺",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
    color: white;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
}
.badge {
    padding: 6px 12px;
    border-radius: 12px;
    font-weight: bold;
}
.ok { background: #1f8f3a; }
.warn { background: #b33a3a; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🦺 AI-Powered PPE Compliance Monitoring")
st.caption("Real-time multi-class PPE detection with smooth video playback")

# ===================== IMPORT =====================
from detect import PPEDetector

# ===================== SESSION STATE =====================
defaults = {
    "running": False,
    "paused": False,
    "frames": 0,
    "violations": 0,
    "rows": [],
    "worker_counter": 0,
    "persist": defaultdict(int),
    "seen": set(),
    "class_count": defaultdict(int),
    "last_detections": []
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ===================== SIDEBAR =====================
st.sidebar.markdown("## ⚙️ Control Panel")

confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Start"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.session_state.running = True

if c2.button("⏸ Pause"):
    st.session_state.paused = True

if c3.button("▶ Resume"):
    st.session_state.paused = False

uploaded_video = st.sidebar.file_uploader("📤 Upload Video", ["mp4","avi"])

st.sidebar.markdown("---")
st.sidebar.info(
    "🟢 Green = PPE Compliant\n\n"
    "🔴 Red = PPE Violation\n\n"
    "⚡ Optimized for Cloud"
)

# ===================== KPI =====================
k1, k2, k3, k4 = st.columns(4)

k1.markdown(f"<div class='card'>📽 Frames<br><h2>{st.session_state.frames}</h2></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='card'>🚨 Violations<br><h2>{st.session_state.violations}</h2></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='card'>👷 Workers<br><h2>{st.session_state.worker_counter}</h2></div>", unsafe_allow_html=True)

status = "PAUSED" if st.session_state.paused else "RUNNING" if st.session_state.running else "IDLE"
badge_class = "ok" if status == "RUNNING" else "warn"
k4.markdown(
    f"<div class='card'>Status<br><span class='badge {badge_class}'>{status}</span></div>",
    unsafe_allow_html=True
)

# ===================== LAYOUT =====================
video_col, charts_col = st.columns([2.2, 1])
log_ph = st.empty()

fps_hist, time_hist = deque(maxlen=120), deque(maxlen=120)

# ===================== DRAW BOXES =====================
def draw_boxes(img, detections):
    for d in detections:
        x1,y1,x2,y2 = map(int, d["bbox"].values())
        cls = d["class_name"]
        conf = d["confidence"]
        color = CLASS_COLORS.get(cls, (255,255,0))

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        cv2.putText(
            img,f"{cls} {conf:.2f}",
            (x1, max(20, y1-6)),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2
        )
    return img

# ===================== DETECTION =====================
if st.session_state.running:

    if not uploaded_video:
        st.error("❌ Please upload a video")
        st.stop()

    detector = PPEDetector(MODEL_PATH, confidence)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded_video.read())
    cap = cv2.VideoCapture(tmp.name)

    fps_target = cap.get(cv2.CAP_PROP_FPS) or 25
    prev = time.time()

    while cap.isOpened() and st.session_state.running:
        if st.session_state.paused:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if st.session_state.frames % DETECT_EVERY == 0:
            annotated, detections = detector.detect(rgb)
            st.session_state.last_detections = detections
        else:
            annotated = rgb.copy()
            detections = st.session_state.last_detections

        annotated = draw_boxes(annotated, detections)

        for d in detections:
            if d["class_name"] in VIOLATION_CLASSES:
                key = f"{d['class_name']}_{round(d['bbox']['x1'],1)}"
                st.session_state.persist[key] += 1

                if st.session_state.persist[key] >= PERSIST_FRAMES and key not in st.session_state.seen:
                    st.session_state.worker_counter += 1
                    worker_id = f"W_{st.session_state.worker_counter}"

                    snap = f"{SNAP_DIR}/{int(time.time()*1000)}.jpg"
                    cv2.imwrite(snap, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                    st.session_state.rows.append([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        worker_id,
                        d["class_name"],
                        d["confidence"],
                        *d["bbox"].values(),
                        snap
                    ])

                    st.session_state.violations += 1
                    st.session_state.class_count[d["class_name"]] += 1
                    st.session_state.seen.add(key)

                    st.toast(f"🚨 {d['class_name']} detected", icon="⚠️")

        video_col.image(annotated, channels="RGB", use_container_width=True)

        fps = 1 / max(time.time() - prev, 1e-6)
        prev = time.time()
        fps_hist.append(fps)
        time_hist.append(datetime.now().strftime("%H:%M:%S"))

        fig = go.Figure(go.Scatter(
            x=list(time_hist),
            y=list(fps_hist),
            mode="lines"
        ))
        fig.update_layout(
            title="FPS Performance",
            height=260,
            margin=dict(l=10,r=10,t=40,b=10)
        )
        charts_col.plotly_chart(fig, use_container_width=True)

        log_ph.dataframe(
            pd.DataFrame(st.session_state.rows, columns=CSV_HEADER).tail(12),
            use_container_width=True
        )

        time.sleep(1 / fps_target)

    cap.release()
    st.session_state.running = False
    st.success("✅ Video processing completed")

# ===================== DOWNLOAD =====================
if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows, columns=CSV_HEADER)
    st.download_button("📥 Download CSV Report", df.to_csv(index=False), "ppe_violations.csv")

    zip_path = "snapshots.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for p in df["snapshot"]:
            if os.path.exists(p):
                z.write(p)

    with open(zip_path, "rb") as f:
        st.download_button("🗂 Download Snapshots", f, "snapshots.zip")
