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

# ---------------- Mode ----------------
MODE = os.getenv("MODE", "development")  # set MODE=production in Streamlit Cloud
st.write(f"Running in {MODE} mode")

# ---------------- Optional system libs ----------------
try:
    import psutil
except Exception:
    psutil = None

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
    nvmlInit()
    have_nvml = True
except Exception:
    have_nvml = False

# ---------------- Model loading ----------------
import gdown  # for downloading model if missing
from detect import PPEDetector

MODEL_PATH = "models/yolov8_ppe.pt"
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    os.makedirs("models", exist_ok=True)
    MODEL_URL = "YOUR_MODEL_DOWNLOAD_LINK"  # replace with your Google Drive/HuggingFace link
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded.")

# ---------------- Config ----------------
st.set_page_config(page_title="PPE Monitoring Dashboard", layout="wide")
st.title("🟦 PPE Monitoring Dashboard")
st.markdown("Live PPE detection, violations, snapshots, charts — now with Pause / Resume / Stop controls")

# ---------------- Sidebar Controls ----------------
st.sidebar.header("Settings")
video_source = st.sidebar.selectbox("Video Source", ["Upload Video File"])
confidence_slider = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.01)
uploaded_file = st.sidebar.file_uploader("Upload video (mp4/avi)", type=["mp4", "avi"])

st.sidebar.markdown("### Controls")
if "running" not in st.session_state:
    st.session_state.running = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "session_rows" not in st.session_state:
    st.session_state.session_rows = []
if "download_button_created" not in st.session_state:
    st.session_state.download_button_created = False
if "download_csv_bytes" not in st.session_state:
    st.session_state.download_csv_bytes = None
if "download_file_name" not in st.session_state:
    st.session_state.download_file_name = None

# Buttons: Start, Pause, Resume, Stop
col_a, col_b = st.sidebar.columns(2)
if col_a.button("Start"):
    if uploaded_file is None:
        st.warning("Please upload a video file to start.")
        st.stop()
    st.session_state.running = True
    st.session_state.paused = False
    st.session_state.session_rows = []
    st.session_state.download_button_created = False
if col_a.button("Pause") and st.session_state.running:
    st.session_state.paused = True
if col_b.button("Resume") and st.session_state.running:
    st.session_state.paused = False
if col_b.button("Stop"):
    st.session_state.running = False
    st.session_state.paused = False
    st.session_state.download_button_created = False

# ---------------- Files & constants ----------------
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
CSV_HEADER = ["timestamp", "worker_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "snapshot_path"]

SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

# ---------------- Simple Tracker ----------------
class SimpleTracker:
    def __init__(self, max_lost=30, dist_thresh=75):
        self.next_id = 1
        self.tracks = OrderedDict()
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(self, person_bboxes):
        updated = {}
        if len(self.tracks) == 0:
            for bbox in person_bboxes:
                cid = f"P_{self.next_id}"
                self.tracks[cid] = {"centroid": self._centroid(bbox), "lost": 0, "bbox": bbox}
                updated[cid] = bbox
                self.next_id += 1
            return updated

        detections = [self._centroid(b) for b in person_bboxes]
        track_ids = list(self.tracks.keys())
        track_cents = [self.tracks[t]["centroid"] for t in track_ids]

        used_dets = set()
        for i, tcent in enumerate(track_cents):
            best_j, best_dist = None, None
            for j, dcent in enumerate(detections):
                if j in used_dets:
                    continue
                dist = np.linalg.norm(np.array(tcent) - np.array(dcent))
                if best_dist is None or dist < best_dist:
                    best_j, best_dist = j, dist
            tid = track_ids[i]
            if best_j is not None and best_dist <= self.dist_thresh:
                self.tracks[tid]["centroid"] = detections[best_j]
                self.tracks[tid]["bbox"] = person_bboxes[best_j]
                self.tracks[tid]["lost"] = 0
                updated[tid] = person_bboxes[best_j]
                used_dets.add(best_j)
            else:
                self.tracks[tid]["lost"] += 1

        for j, bbox in enumerate(person_bboxes):
            if j in used_dets:
                continue
            cid = f"P_{self.next_id}"
            self.tracks[cid] = {"centroid": detections[j], "lost": 0, "bbox": bbox}
            updated[cid] = bbox
            self.next_id += 1

        to_remove = [tid for tid, v in self.tracks.items() if v["lost"] > self.max_lost]
        for tid in to_remove:
            self.tracks.pop(tid, None)

        return updated

tracker = SimpleTracker(max_lost=30, dist_thresh=80)

# ---------------- Detection Loop ----------------
if st.session_state.running:
    detector = PPEDetector(model_path=MODEL_PATH, conf=confidence_slider)

    # Save uploaded video to temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]).name
    with open(tmp_file, "wb") as f:
        f.write(uploaded_file.read())
    cap = cv2.VideoCapture(tmp_file)

    prev_time = time.time()
    frame_idx = 0
    total_frames = 0
    total_violations = 0
    fps_history = deque(maxlen=240)
    conf_history = deque(maxlen=240)
    time_history = deque(maxlen=240)

    video_ph = st.empty()
    fps_ph = st.empty()
    kpi_cols = st.columns(3)
    kpi1_ph, kpi2_ph, kpi3_ph = kpi_cols

    try:
        while cap.isOpened() and st.session_state.running:
            if st.session_state.paused:
                fps_ph.markdown("### ⏸ Paused")
                time.sleep(0.2)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                st.info("Stream ended or cannot read frame.")
                break

            total_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                annotated_frame, detections = detector.detect(rgb)
            except Exception as e:
                st.error(f"Detector error: {e}")
                annotated_frame = rgb.copy()
                detections = []

            # Tracker & violations
            person_bboxes = [(int(d["bbox"]["x1"]), int(d["bbox"]["y1"]),
                              int(d["bbox"]["x2"]), int(d["bbox"]["y2"]))
                             for d in detections if d.get("class_name", "").lower() == "person"]
            tracks = tracker.update(person_bboxes)
            track_centroids = {tid: ((b[0]+b[2])//2, (b[1]+b[3])//2) for tid, b in tracks.items()}

            rows_to_append = []
            frame_violations = 0
            max_conf_frame = 0.0

            for d in detections:
                cls = d.get("class_name", "")
                conf = float(d.get("confidence", 0.0))
                max_conf_frame = max(max_conf_frame, conf)

                if cls in VIOLATION_CLASSES:
                    bx1, by1 = int(d["bbox"]["x1"]), int(d["bbox"]["y1"])
                    bx2, by2 = int(d["bbox"]["x2"]), int(d["bbox"]["y2"])
                    bcx, bcy = (bx1+bx2)//2, (by1+by2)//2

                    nearest_id, nearest_dist = "", None
                    for tid, (tcx, tcy) in track_centroids.items():
                        dist = np.linalg.norm(np.array((tcx, tcy)) - np.array((bcx, bcy)))
                        if nearest_dist is None or dist < nearest_dist:
                            nearest_dist = dist
                            nearest_id = tid
                    worker_id = nearest_id if (nearest_dist is not None and nearest_dist < 150) else ""

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    snap = f"{SNAP_DIR}/snap_{ts}.jpg"
                    try:
                        cv2.imwrite(snap, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                    except Exception:
                        snap = ""

                    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), worker_id, cls, round(conf,3),
                           bx1, by1, bx2, by2, snap]
                    rows_to_append.append(row)
                    st.session_state.session_rows.append(row)
                    frame_violations += 1

            total_violations += frame_violations

            # KPI updates
            kpi1_ph.metric("Total Frames", total_frames)
            kpi2_ph.metric("Total Violations", total_violations)
            kpi3_ph.metric("Active Alerts (this frame)", frame_violations)

            # show annotated frame
            video_ph.image(annotated_frame, channels="RGB", width='stretch')
            fps_ph.markdown(f"### ⚡ FPS: {1.0 / max(1e-6, time.time() - prev_time):.2f}")
            prev_time = time.time()

            time.sleep(0.01)

    finally:
        cap.release()
        st.session_state.running = False
        st.session_state.paused = False
        st.success("Detection stopped.")

else:
    st.info("Upload a video and use the sidebar controls to Start / Pause / Resume / Stop.")
