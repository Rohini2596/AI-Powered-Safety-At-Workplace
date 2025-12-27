import os, time, tempfile, threading
from collections import deque
from datetime import datetime

import streamlit as st
import cv2
import pandas as pd
import plotly.graph_objects as go
import gdown

# ================= CONFIG =================
MODEL_PATH = "models/yolov8_ppe.pt"
MODEL_URL = "https://drive.google.com/uc?id=1qLB4ZjijrpNdHcphQftVudm8y4SOZDoL"

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
DETECT_EVERY = 5

# ================= MODEL =================
@st.cache_resource
def download_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

MODEL_PATH = download_model()

from detect import PPEDetector

# ================= STATE =================
if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.fps_hist = []
    st.session_state.time_hist = []
    st.session_state.processing_done = False

# ================= UI =================
st.set_page_config("PPE Monitoring", "🦺", layout="wide")
st.title("🦺 PPE Compliance Monitoring (Media-Player Mode)")
st.caption("Video playback without frame rendering • Background detection")

uploaded_video = st.file_uploader("Upload video", ["mp4", "avi"])

start = st.button("▶ Start Analysis")

video_col, chart_col = st.columns([2, 1])

# ================= BACKGROUND PROCESS =================
def process_video(video_path):
    detector = PPEDetector(MODEL_PATH, 0.5)
    cap = cv2.VideoCapture(video_path)

    prev = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_id % DETECT_EVERY == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, detections = detector.detect(rgb)

            for d in detections:
                if d["class_name"] in VIOLATION_CLASSES:
                    st.session_state.results.append([
                        datetime.now().strftime("%H:%M:%S"),
                        d["class_name"],
                        d["confidence"]
                    ])

        now = time.time()
        fps = 1 / max(now - prev, 1e-6)
        prev = now

        st.session_state.fps_hist.append(fps)
        st.session_state.time_hist.append(frame_id)

    cap.release()
    st.session_state.processing_done = True

# ================= MAIN =================
if uploaded_video and start:

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded_video.read())
    video_path = tmp.name

    # ▶ Play video normally
    video_col.video(video_path)

    # 🧠 Run detection in background
    thread = threading.Thread(target=process_video, args=(video_path,))
    thread.start()

    # 📊 Live chart
    while not st.session_state.processing_done:
        if st.session_state.fps_hist:
            fig = go.Figure(go.Scatter(
                x=st.session_state.time_hist,
                y=st.session_state.fps_hist,
                mode="lines"
            ))
            fig.update_layout(
                title="Processing FPS (Background)",
                height=300
            )
            chart_col.plotly_chart(fig, use_container_width=True)

        time.sleep(0.5)

    st.success("✅ Analysis completed")

# ================= LOG =================
if st.session_state.results:
    st.markdown("### 🚨 Violations Detected")
    st.dataframe(
        pd.DataFrame(
            st.session_state.results,
            columns=["Time", "Class", "Confidence"]
        ),
        use_container_width=True
    )
