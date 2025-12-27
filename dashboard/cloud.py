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
        with st.spinner("⬇️ Downloading PPE model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

MODEL_PATH = download_model()

# ===================== IMPORT DETECTOR =====================
from detect import PPEDetector

# ===================== CONFIG =====================
st.set_page_config(page_title="🦺 PPE Monitoring", layout="wide")
st.markdown("<h1 style='text-align:center'>🦺 PPE Compliance Dashboard</h1>", unsafe_allow_html=True)

# ===================== CONSTANTS =====================
SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
CSV_HEADER = ["timestamp","worker_id","class_name","confidence","x1","y1","x2","y2","snapshot"]

# ===================== SESSION STATE =====================
if "running" not in st.session_state: st.session_state.running = False
if "paused" not in st.session_state: st.session_state.paused = False
if "rows" not in st.session_state: st.session_state.rows = []

# ===================== CONTROLS =====================
col1, col2, col3 = st.columns([1,1,1])
video_file = col1.file_uploader("Upload Video (mp4/avi)", type=["mp4","avi"])
confidence = col2.slider("Confidence Threshold", 0.1, 1.0, 0.5)
if col3.button("▶ Start"):
    st.session_state.running = True
    st.session_state.paused = False
    st.session_state.rows = []
if col3.button("⏹ Stop"):
    st.session_state.running = False
    st.session_state.paused = False

# ===================== KPI CARDS =====================
kpi_cols = st.columns(3)
fps_hist = deque(maxlen=120)
viol_hist = deque(maxlen=120)
time_hist = deque(maxlen=120)

# ===================== DETECTION LOOP =====================
if st.session_state.running:
    if not video_file:
        st.error("Upload a video first!")
        st.stop()

    detector = PPEDetector(model_path=MODEL_PATH, conf=confidence)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_file.read())
    cap = cv2.VideoCapture(tmp.name)
    prev_time = time.time()

    video_ph, chart_ph, table_ph = st.columns([3,2,3])
    frame_count = 0
    total_violations = 0

    while cap.isOpened() and st.session_state.running:
        if st.session_state.paused:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated, detections = detector.detect(rgb)

        # ---------------- Violations ----------------
        frame_violations = 0
        for d in detections:
            if d["class_name"] in VIOLATION_CLASSES:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snap_path = f"{SNAP_DIR}/snap_{ts}.jpg"
                cv2.imwrite(snap_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                st.session_state.rows.append([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "P_1",
                    d["class_name"],
                    d["confidence"],
                    *d["bbox"].values(),
                    snap_path
                ])
                frame_violations += 1

        total_violations += frame_violations

        # ---------------- FPS ----------------
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps_hist.append(fps)
        viol_hist.append(total_violations)
        time_hist.append(datetime.now().strftime("%H:%M:%S"))

        # ---------------- UI Updates ----------------
        video_ph.image(annotated, channels="RGB", use_column_width=True)
        kpi_cols[0].metric("Frames Processed", frame_count)
        kpi_cols[1].metric("Total Violations", total_violations)
        kpi_cols[2].metric("FPS", f"{fps:.2f}")

        # ---------------- Trend Chart ----------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(time_hist), y=list(fps_hist), mode='lines+markers', name='FPS'))
        fig.add_trace(go.Scatter(x=list(time_hist), y=list(viol_hist), mode='lines+markers', name='Violations'))
        fig.update_layout(height=300, title="FPS & Violations Trend")
        chart_ph.plotly_chart(fig, use_container_width=True)

        # ---------------- Violations Table ----------------
        table_ph.dataframe(pd.DataFrame(st.session_state.rows, columns=CSV_HEADER).tail(20), use_container_width=True)

        time.sleep(0.01)

    cap.release()
    st.session_state.running = False
    st.success("Detection completed!")

# ===================== Download CSV =====================
if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows, columns=CSV_HEADER)
    st.download_button("📥 Download Violations CSV", df.to_csv(index=False), "ppe_violations.csv")
