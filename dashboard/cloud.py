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

from detect import PPEDetector

# ---------------- Config ----------------
st.set_page_config(page_title="🦺 PPE Monitoring Dashboard", layout="wide")
st.title("🦺 PPE Compliance Monitoring Dashboard")
st.markdown("Live PPE detection with FPS trend, violation table, and snapshots")

# ---------------- Constants ----------------
SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
CSV_HEADER = ["timestamp", "worker_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "snapshot"]

# ---------------- Session State ----------------
if "running" not in st.session_state:
    st.session_state.running = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "rows" not in st.session_state:
    st.session_state.rows = []
if "download_ready" not in st.session_state:
    st.session_state.download_ready = False

# ---------------- Controls ----------------
controls = st.columns(4)
if controls[0].button("▶ Start"):
    st.session_state.running = True
    st.session_state.paused = False
    st.session_state.rows = []
    st.session_state.download_ready = False

if controls[1].button("⏸ Pause"):
    st.session_state.paused = True

if controls[2].button("▶ Resume"):
    st.session_state.paused = False

if controls[3].button("⏹ Stop"):
    st.session_state.running = False
    st.session_state.paused = False

uploaded_video = st.file_uploader("Upload MP4 / AVI", type=["mp4","avi"])

# ---------------- Layout ----------------
left_col, right_col = st.columns([2.5, 1])
video_ph = left_col.empty()
kpi_cols = left_col.columns(3)
kpi_frames, kpi_viol, kpi_status = kpi_cols

fps_chart_ph = right_col.empty()
viol_chart_ph = right_col.empty()
logs_ph = st.empty()

# ---------------- Stats ----------------
fps_hist = deque(maxlen=120)
time_hist = deque(maxlen=120)
total_frames = 0
total_violations = 0

# ---------------- Detector ----------------
MODEL_PATH = "models/yolov8_ppe.pt"
detector = PPEDetector(model_path=MODEL_PATH, conf=0.5)

# ---------------- Detection Loop ----------------
if st.session_state.running:

    if not uploaded_video:
        st.warning("Please upload a video file first.")
        st.stop()

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(uploaded_video.read())
    cap = cv2.VideoCapture(tmp_file.name)
    prev_time = time.time()

    while cap.isOpened() and st.session_state.running:

        if st.session_state.paused:
            time.sleep(0.2)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated, detections = detector.detect(rgb)

        frame_violations = 0
        for d in detections:
            if d["class_name"] in VIOLATION_CLASSES:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                snap = f"{SNAP_DIR}/{int(time.time()*1000)}.jpg"
                cv2.imwrite(snap, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                st.session_state.rows.append([
                    ts, "P_1", d["class_name"], d["confidence"],
                    *d["bbox"].values(), snap
                ])
                frame_violations += 1
                total_violations += 1

        # FPS
        now = time.time()
        fps = 1 / max(1e-6, now - prev_time)
        prev_time = now
        fps_hist.append(fps)
        time_hist.append(datetime.now().strftime("%H:%M:%S"))

        # Update KPIs
        kpi_frames.metric("Frames", total_frames)
        kpi_viol.metric("Violations", total_violations)
        kpi_status.metric("Status", "RUNNING" if st.session_state.running else "IDLE")

        # Display video
        video_ph.image(annotated, channels="RGB", use_container_width=True)

        # Update charts
        if len(fps_hist) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(time_hist), y=list(fps_hist), mode="lines+markers", name="FPS"))
            fig.update_layout(title="FPS Trend", height=250, xaxis_title="Time", yaxis_title="FPS")
            fps_chart_ph.plotly_chart(fig, use_container_width=True)

            df = pd.DataFrame(st.session_state.rows, columns=CSV_HEADER)
            dist = df['class_name'].value_counts().reset_index()
            dist.columns = ['violation_type','count']
            fig2 = go.Figure(go.Bar(x=dist['violation_type'], y=dist['count'], marker_color='red'))
            fig2.update_layout(title="Violation Counts", xaxis_title="Type", yaxis_title="Count")
            viol_chart_ph.plotly_chart(fig2, use_container_width=True)

            # Logs
            logs_ph.dataframe(df.tail(20), use_container_width=True)

            # CSV download
            if not st.session_state.download_ready:
                st.session_state.download_ready = True
                st.download_button(
                    "📥 Download CSV",
                    df.to_csv(index=False),
                    file_name=f"ppe_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )

        time.sleep(0.01)

    cap.release()
    st.session_state.running = False
    st.success("✅ Detection stopped")

else:
    st.info("Upload a video and click ▶ Start to begin detection.")
