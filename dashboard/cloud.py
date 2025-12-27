import os
import time
import tempfile
from collections import deque
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import cv2

from detect import PPEDetector

# ----------------- Config -----------------
st.set_page_config(page_title="PPE Monitoring Dashboard", layout="wide")
st.title("🦺 PPE Compliance Monitoring Dashboard")
st.markdown("Live PPE detection | FPS & Violation trends | Logs")

# ----------------- Constants -----------------
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
CSV_HEADER = ["timestamp", "worker_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "snapshot"]
LOG_CSV = "violations.csv"
SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

# ----------------- Session state -----------------
for key, default in {
    "session_rows": [],
    "fps_hist": deque(maxlen=200),
    "viol_hist": deque(maxlen=200),
    "time_hist": deque(maxlen=200),
    "total_violations": 0,
    "total_frames": 0
}.items():
    st.session_state.setdefault(key, default)

# ----------------- Upload video -----------------
uploaded_file = st.file_uploader("Upload MP4 / AVI", type=["mp4", "avi"])
confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ----------------- Placeholders -----------------
kpi_cols = st.columns(3)
total_frames_ph = kpi_cols[0].empty()
total_viol_ph = kpi_cols[1].empty()
active_viol_ph = kpi_cols[2].empty()

chart_ph = st.empty()
table_ph = st.empty()

# ----------------- Detection Loop -----------------
if uploaded_file:
    detector = PPEDetector(model_path="models/yolov8_ppe.pt", conf=confidence)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(uploaded_file.read())
    tmp_file.close()

    cap = cv2.VideoCapture(tmp_file.name)
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated, detections = detector.detect(rgb)

        frame_violations = 0
        rows_to_append = []

        for d in detections:
            cls = d["class_name"]
            conf = d["confidence"]
            if cls in VIOLATION_CLASSES:
                frame_violations += 1
                st.session_state.total_violations += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snap_path = f"{SNAP_DIR}/snap_{ts}.jpg"
                cv2.imwrite(snap_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                row = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "P_1",
                    cls,
                    conf,
                    d["bbox"]["x1"], d["bbox"]["y1"],
                    d["bbox"]["x2"], d["bbox"]["y2"],
                    snap_path
                ]
                rows_to_append.append(row)
                st.session_state.session_rows.append(row)

        # FPS
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        # Update histories
        st.session_state.fps_hist.append(fps)
        st.session_state.viol_hist.append(frame_violations)
        st.session_state.time_hist.append(datetime.now().strftime("%H:%M:%S"))

        # ----------------- Update KPIs -----------------
        total_frames_ph.metric("Total Frames", st.session_state.total_frames)
        total_viol_ph.metric("Total Violations", st.session_state.total_violations)
        active_viol_ph.metric("Violations This Frame", frame_violations)

        # ----------------- Update Charts -----------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(st.session_state.time_hist),
            y=list(st.session_state.fps_hist),
            mode="lines+markers", name="FPS"
        ))
        fig.add_trace(go.Scatter(
            x=list(st.session_state.time_hist),
            y=list(st.session_state.viol_hist),
            mode="lines+markers", name="Violations"
        ))
        fig.update_layout(title="FPS & Violations Trend", height=300, xaxis_title="Time")
        chart_ph.plotly_chart(fig, use_container_width=True)

        # ----------------- Update Table -----------------
        if st.session_state.session_rows:
            df_table = pd.DataFrame(st.session_state.session_rows, columns=CSV_HEADER)
            table_ph.dataframe(df_table.tail(50), use_container_width=True)

    cap.release()
    st.success("✅ Video processing complete")

# ----------------- CSV Download -----------------
if st.session_state.session_rows:
    df_export = pd.DataFrame(st.session_state.session_rows, columns=CSV_HEADER)
    st.download_button(
        "📥 Download Session Violations CSV",
        df_export.to_csv(index=False).encode("utf-8"),
        file_name=f"ppe_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
