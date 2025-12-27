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

from ultralytics import YOLO

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="PPE AI Monitoring",
    page_icon="🦺",
    layout="wide"
)

# ===================== MODEL DOWNLOAD =====================
MODEL_PATH = "models/yolov8_ppe.pt"
MODEL_URL = "https://drive.google.com/uc?id=1qLB4ZjijrpNdHcphQftVudm8y4SOZDoL"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with st.spinner("⬇️ Downloading PPE detection model (first run only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

# ===================== CONSTANTS =====================
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

# ===================== UI STYLES =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
    color: white;
}
.block {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# ===================== TITLE =====================
st.markdown("## 🦺 AI-Powered PPE Compliance Monitoring")
st.caption("Live video • Dynamic analytics • Cloud-ready")

# ===================== CONTROLS =====================
top1, top2 = st.columns([3, 1])

with top1:
    uploaded_video = st.file_uploader(
        "📁 Upload Site / CCTV Video",
        type=["mp4", "avi"]
    )

with top2:
    start_btn = st.button("▶ Start Detection", use_container_width=True)

confidence = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.5)

st.markdown("---")

# ===================== DASHBOARD LAYOUT =====================
left, right = st.columns([2.2, 1])

video_ph = left.empty()
kpi_ph = right.empty()
fps_chart_ph = right.empty()

st.markdown("### 🚨 Live Violations")
table_ph = st.empty()

# ===================== DETECTION =====================
if start_btn:

    if not uploaded_video:
        st.error("❌ Please upload a video first")
        st.stop()

    # Temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded_video.read())
    cap = cv2.VideoCapture(tmp.name)

    fps_hist = deque(maxlen=60)
    frame_hist = deque(maxlen=60)
    violations = []

    prev_time = time.time()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model.predict(rgb, conf=confidence, verbose=False)
        annotated = results[0].plot()
        boxes = results[0].boxes

        # ---------------- Violations ----------------
        for box in boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name in VIOLATION_CLASSES:
                violations.append([
                    datetime.now().strftime("%H:%M:%S"),
                    cls_name,
                    round(float(box.conf[0]), 2)
                ])

        # ---------------- FPS ----------------
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        fps_hist.append(fps)
        frame_hist.append(frame_id)

        # ================= UI UPDATES =================

        # 🎥 Video
        video_ph.image(
            annotated,
            channels="RGB",
            use_container_width=True
        )

        # 🚨 KPI
        kpi_ph.markdown(
            f"""
            <div class="block">
                <h3>🚨 Total Violations</h3>
                <h1>{len(violations)}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 📊 FPS Chart
        fig = go.Figure()
        fig.add_scatter(
            x=list(frame_hist),
            y=list(fps_hist),
            mode="lines",
            line=dict(width=3)
        )
        fig.update_layout(
            height=260,
            template="plotly_dark",
            title="📊 Live FPS Trend",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Frame",
            yaxis_title="FPS"
        )
        fps_chart_ph.plotly_chart(fig, use_container_width=True)

        # 📋 Table
        table_ph.dataframe(
            pd.DataFrame(
                violations,
                columns=["Time", "Violation Type", "Confidence"]
            ).tail(15),
            use_container_width=True
        )

    cap.release()
    st.success("✅ Detection completed")

    # ===================== DOWNLOAD =====================
    if violations:
        df = pd.DataFrame(
            violations,
            columns=["Time", "Violation Type", "Confidence"]
        )
        st.download_button(
            "📥 Download Violations CSV",
            df.to_csv(index=False),
            "ppe_violations.csv"
        )
