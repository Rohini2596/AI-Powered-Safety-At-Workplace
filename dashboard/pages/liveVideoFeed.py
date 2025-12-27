import streamlit as st
import cv2
import time
import numpy as np

# FIXED IMPORT (no dashboard.)
from detect import PPEDetector


def run_live_feed():
    st.title("🟦 Real-Time PPE Detection Dashboard")
    st.markdown("### YOLOv8 PPE Monitoring — Streamlit")

    # Sidebar Controls
    st.sidebar.header("⚙️ Settings")

    video_source = st.sidebar.selectbox(
        "Select Source",
        ["Webcam (0)", "External Cam (1)", "RTSP Stream", "Video File"]
    )

    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    run_detection = st.sidebar.checkbox("Start Detection")

    # Select source
    if video_source == "Webcam (0)":
        source = 0
    elif video_source == "External Cam (1)":
        source = 1
    elif video_source == "RTSP Stream":
        source = st.sidebar.text_input("RTSP URL", "rtsp://username:password@ip:port/stream")
    else:
        source = st.sidebar.file_uploader("Upload video", type=["mp4", "avi"])

    detector = PPEDetector(conf=confidence)

    frame_placeholder = st.empty()
    fps_placeholder = st.empty()

    if run_detection:

        # Uploaded file
        if video_source == "Video File":
            if source:
                temp_path = "uploaded_video.mp4"
                with open(temp_path, "wb") as f:
                    f.write(source.read())
                cap = cv2.VideoCapture(temp_path)
            else:
                st.warning("Upload a video first.")
                st.stop()

        # Webcam / RTSP
        elif isinstance(source, int) or "rtsp" in str(source):
            cap = cv2.VideoCapture(source)
        else:
            st.warning("Invalid video source.")
            st.stop()

        prev_time = time.time()

        # Video Loop
        while cap.isOpened() and run_detection:
            ret, frame = cap.read()
            if not ret:
                st.warning("Stream ended or failed.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detection
            annotated, results = detector.detect(frame)

            # FPS
            curr = time.time()
            fps = 1 / (curr - prev_time)
            prev_time = curr

            frame_placeholder.image(annotated, channels="RGB")
            fps_placeholder.markdown(f"### ⚡ FPS: `{fps:.2f}`")

        cap.release()
