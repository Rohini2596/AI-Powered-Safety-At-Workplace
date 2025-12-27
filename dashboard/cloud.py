import os
import time
import tempfile
from collections import deque
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gdown

# ===================== MODEL DOWNLOAD =====================
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

# ===================== UI STYLE =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
    color: white;
}
.stSidebar .sidebar-content {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 12px;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🦺 AI-Powered PPE Compliance Monitoring Dashboard")
st.caption("Cloud-safe PPE detection using YOLOv8")

# ===================== IMPORT DETECTOR =====================
from detect import PPEDetector

# ===================== CONSTANTS =====================
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
    "violations": 0
}.items():
    st.session_state.setdefault(k, v)

# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ Controls")

video_mode = st.sidebar.radio(
    "Video Source",
    ["Upload Video"]
)

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

if st.sidebar.button("▶ Start"):
    st.session_state.running = True
    st.session_state.rows = []
    st.session_state.violations = 0

if st.sidebar.button("⏹ Stop"):
    st.session_state.running = False

uploaded_video = st.sidebar.file_uploader(
    "Upload MP4 / AVI", type=["mp4", "avi"]
)

# ===================== KPI =====================
k1, k2 = st.columns(2)
k1.metric("Violations", st.session_state.violations)
k2.metric("Status", "RUNNING" if st.session_state.running else "IDLE")

video_ph, chart_ph = st.columns([2, 1])
log_ph = st.empty()

fps_hist = deque(maxlen=120)
time_hist = deque(maxlen=120)

# ===================== DETECTION =====================
if st.session_state.running:

    if not uploaded_video:
        st.error("❌ Please upload a video (Webcam is not supported on Streamlit Cloud)")
        st.stop()

    detector = PPEDetector(model_path=MODEL_PATH, conf=confidence)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded_video.read())
    cap = cv2.VideoCapture(tmp.name)

    prev = time.time()

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        annotated, detections = detector.detect(rgb)

        # Save violations
        for d in detections:
            if d["class_name"] in VIOLATION_CLASSES:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                snap = f"{SNAP_DIR}/{int(time.time()*1000)}.jpg"
                cv2.imwrite(snap, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                st.session_state.rows.append([
                    ts, "P_1", d["class_name"], d["confidence"],
                    *d["bbox"].values(), snap
                ])
                st.session_state.violations += 1

        # Update FPS chart
        now = time.time()
        fps = 1 / max(now - prev, 1e-6)
        prev = now
        fps_hist.append(fps)
        time_hist.append(datetime.now().strftime("%H:%M:%S"))

        video_ph.image(annotated, channels="RGB", use_container_width=True)

        fig = go.Figure()
        fig.add_scatter(x=list(time_hist), y=list(fps_hist))
        fig.update_layout(height=300, title="FPS Trend")
        chart_ph.plotly_chart(fig, use_container_width=True)

        # Update violation table
        log_ph.dataframe(
            pd.DataFrame(st.session_state.rows, columns=CSV_HEADER).tail(20),
            use_container_width=True
        )

    cap.release()
    st.session_state.running = False
    st.success("✅ Detection completed")

# ===================== DOWNLOAD =====================
if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows, columns=CSV_HEADER)
    st.download_button(
        "📥 Download Violations CSV",
        df.to_csv(index=False),
        "ppe_violations.csv"
    )
