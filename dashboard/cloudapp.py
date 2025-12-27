import os
import time
import csv
import tempfile
from collections import deque, OrderedDict
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import gdown
import streamlit as st

MODEL_PATH = "models/yolov8_ppe.pt"
MODEL_URL = "https://drive.google.com/uc?id=1qLB4ZjijrpNdHcphQftVudm8y4SOZDoL"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with st.spinner("⬇️ Downloading PPE model (first run only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

MODEL_PATH = download_model()

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="PPE AI Monitoring",
    page_icon="🦺",
    layout="wide"
)

# ===================== CSS (UI BOOST) =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
    color: white;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}
.metric-card h1 {
    font-size: 36px;
}
[data-testid="stSidebar"] {
    background: #0b1e2d;
}
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ===================== TITLE =====================
st.markdown("## 🦺 AI-Powered PPE Compliance Monitoring Dashboard")
st.caption("Real-time detection • Worker tracking • Cloud analytics")

# ===================== IMPORT DETECTOR =====================
from detect import PPEDetector

# ===================== CONSTANTS =====================
LOG_CSV = "violations.csv"
SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
CSV_HEADER = [
    "timestamp","worker_id","class_name",
    "confidence","x1","y1","x2","y2","snapshot"
]

# ===================== SESSION STATE =====================
for k, v in {
    "running": False,
    "paused": False,
    "rows": [],
    "frames": 0,
    "violations": 0
}.items():
    st.session_state.setdefault(k, v)

# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ Controls")

video_mode = st.sidebar.radio(
    "Video Source",
    ["Upload Video", "Webcam (Local only)"]
)

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

col1, col2 = st.sidebar.columns(2)
if col1.button("▶ Start"):
    st.session_state.running = True
    st.session_state.paused = False
    st.session_state.rows = []
    st.session_state.frames = 0
    st.session_state.violations = 0

if col2.button("⏹ Stop"):
    st.session_state.running = False
    st.session_state.paused = False

if st.sidebar.button("⏸ Pause"):
    st.session_state.paused = True

if st.sidebar.button("▶ Resume"):
    st.session_state.paused = False

uploaded_video = None
if video_mode == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload MP4 / AVI", type=["mp4","avi"])

# ===================== KPI CARDS =====================
def kpi(title, value, icon):
    st.markdown(f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <h1>{value}</h1>
    </div>
    """, unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
with k1: kpi("Frames", st.session_state.frames, "🎥")
with k2: kpi("Violations", st.session_state.violations, "🚨")
with k3: kpi("Status", "RUNNING" if st.session_state.running else "IDLE", "⚡")

# ===================== MAIN LAYOUT =====================
left, right = st.columns([2.3, 1])

video_ph = left.empty()
chart_ph = right.empty()
log_ph = st.empty()

fps_hist = deque(maxlen=120)
time_hist = deque(maxlen=120)

# ===================== TRACKER =====================
class SimpleTracker:
    def __init__(self):
        self.id = 1
        self.tracks = {}

    def update(self, boxes):
        out = {}
        for b in boxes:
            out[f"P_{self.id}"] = b
            self.id += 1
        return out

tracker = SimpleTracker()

# ===================== DETECTION LOOP =====================
if st.session_state.running:

    detector = PPEDetector(conf=confidence)

    if video_mode == "Webcam (Local only)":
        cap = cv2.VideoCapture(0)
    else:
        if not uploaded_video:
            st.warning("Upload a video first")
            st.stop()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded_video.read())
        cap = cv2.VideoCapture(tmp.name)

    prev = time.time()

    while cap.isOpened() and st.session_state.running:

        if st.session_state.paused:
            time.sleep(0.3)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        annotated, detections = detector.detect(rgb)

        frame_violations = 0

        for d in detections:
            if d["class_name"] in VIOLATION_CLASSES:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                snap = f"{SNAP_DIR}/{int(time.time()*1000)}.jpg"
                cv2.imwrite(snap, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                st.session_state.rows.append([
                    ts, "P_1", d["class_name"], round(d["confidence"],3),
                    *d["bbox"].values(), snap
                ])

                frame_violations += 1
                st.session_state.violations += 1

        # FPS
        now = time.time()
        fps = 1 / max(now-prev, 1e-6)
        prev = now
        fps_hist.append(fps)
        time_hist.append(datetime.now().strftime("%H:%M:%S"))

        # UI updates
        video_ph.image(annotated, channels="RGB", use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(time_hist), y=list(fps_hist), mode="lines"))
        fig.update_layout(
            title="FPS Trend",
            height=300,
            margin=dict(l=10,r=10,t=40,b=10)
        )
        chart_ph.plotly_chart(fig, use_container_width=True)

        log_ph.dataframe(
            pd.DataFrame(
                st.session_state.rows,
                columns=CSV_HEADER
            ).tail(50),
            use_container_width=True
        )

        time.sleep(0.01)

    cap.release()
    st.session_state.running = False
    st.success("Detection stopped")

# ===================== DOWNLOAD =====================
if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows, columns=CSV_HEADER)
    st.download_button(
        "📥 Download Session CSV",
        df.to_csv(index=False),
        "ppe_violations.csv"
    )
