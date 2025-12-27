
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

# optional system libs
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

# Detector interface: detect.PPEDetector.detect(rgb)->(annotated_rgb, detections_list)
from detect import PPEDetector

# plotly imports
try:
    import plotly.express as px
except Exception:
    px = None
import plotly.graph_objects as go

# ---------------- Config ----------------
st.set_page_config(page_title="PPE Monitoring Dashboard", layout="wide")
st.title("🟦 PPE Monitoring Dashboard")
st.markdown("Live PPE detection, violations, snapshots, charts — now with Pause / Resume / Stop controls")

# ---------------- Sidebar Controls ----------------
st.sidebar.header("Settings")

video_source = st.sidebar.selectbox("Video Source", ["Webcam (0)", "Video File"])
confidence_slider = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.01)
uploaded_file = st.sidebar.file_uploader("Upload video (mp4/avi)", type=["mp4", "avi"])
use_dshow = st.sidebar.checkbox("Use DirectShow (Windows webcam fix)", value=True)

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
LOG_CSV = "violations.csv"
SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
CSV_HEADER = ["timestamp", "worker_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "snapshot_path"]

# ensure CSV header exists
if not os.path.isfile(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

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

# ---------------- CSV helpers ----------------
def clean_csv(path):
    if not os.path.isfile(path):
        return
    cleaned_rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == len(CSV_HEADER):
                cleaned_rows.append(row)
            else:
                print("Skipping bad row:", row)
    if len(cleaned_rows) == 0 or cleaned_rows[0] != CSV_HEADER:
        cleaned = [CSV_HEADER] + [r for r in cleaned_rows if r != CSV_HEADER]
    else:
        cleaned = cleaned_rows
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cleaned)

def load_filtered_df(filter_type=None, filter_worker="", filter_from=None, filter_to=None):
    clean_csv(LOG_CSV)
    if not os.path.isfile(LOG_CSV):
        return pd.DataFrame(columns=CSV_HEADER)
    try:
        df = pd.read_csv(LOG_CSV, parse_dates=["timestamp"], engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(LOG_CSV, parse_dates=["timestamp"])
    if filter_type:
        df = df[df["class_name"].isin(filter_type)]
    if filter_worker:
        df = df[df["worker_id"].astype(str).str.contains(filter_worker, na=False)]
    if filter_from is not None:
        df = df[df["timestamp"].dt.date >= filter_from]
    if filter_to is not None:
        df = df[df["timestamp"].dt.date <= filter_to]
    return df

# ---------------- Layout & Placeholders ----------------
st.markdown("---")
col_video, col_stats = st.columns([2.2, 1])

video_ph = col_video.empty()
fps_ph = col_video.empty()
kpi_cols = col_video.columns(3)
kpi1_ph = kpi_cols[0].empty()
kpi2_ph = kpi_cols[1].empty()
kpi3_ph = kpi_cols[2].empty()

col_stats.subheader("Detection Statistics")
viol_chart_ph = col_stats.empty()
fps_chart_ph = col_stats.empty()
conf_chart_ph = col_stats.empty()

st.markdown("Violation Logs (filters + export)")
fc1, fc2, fc3, fc4 = st.columns(4)
filter_type = fc1.multiselect("Violation Type", options=sorted(list(VIOLATION_CLASSES)), default=sorted(list(VIOLATION_CLASSES)))
filter_worker = fc2.text_input("Worker ID (e.g., P_1)")
filter_from = fc3.date_input("From", value=None)
filter_to = fc4.date_input("To", value=None)
logs_ph = st.empty()

# ---------------- stats & counters ----------------
fps_history = deque(maxlen=240)
conf_history = deque(maxlen=240)
time_history = deque(maxlen=240)
total_frames = 0
total_violations = 0
active_alerts = 0

if psutil:
    try:
        st.sidebar.metric("CPU %", f"{psutil.cpu_percent()}%")
        st.sidebar.metric("Memory %", f"{psutil.virtual_memory().percent}%")
    except Exception:
        pass
if have_nvml:
    try:
        handle = nvmlDeviceGetHandleByIndex(0)
        util = nvmlDeviceGetUtilizationRates(handle)
        st.sidebar.metric("GPU %", f"{util.gpu}%")
    except Exception:
        pass

# ---------------- Detection Loop ----------------
if st.session_state.running:
    detector = PPEDetector(conf=confidence_slider)

    # prepare video source
    if video_source == "Webcam (0)":
        cap_src = 0
    else:
        if uploaded_file is None:
            st.warning("Please upload a video file to start.")
            st.session_state.running = False
            st.stop()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]).name
        with open(tmp_file, "wb") as f:
            f.write(uploaded_file.read())
        cap_src = tmp_file

    # Open capture with DirectShow if requested (Windows)
    if isinstance(cap_src, int) and cap_src == 0 and use_dshow:
        cap = cv2.VideoCapture(cap_src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cap_src)
    else:
        cap = cv2.VideoCapture(cap_src)

    prev_time = time.time()
    frame_idx = 0

    # Reset download button state for this session
    st.session_state.download_button_created = False

    try:
        while cap.isOpened() and st.session_state.running:
            if st.session_state.paused:
                fps_ph.markdown("### ⏸ Paused")
                time.sleep(0.2)
                continue

            ret, frame = False, None
            for _ in range(3):
                ret, frame = cap.read()
                if ret:
                    break
            if not ret or frame is None:
                st.info("Stream ended or cannot read frame.")
                break

            frame_idx += 1
            total_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                annotated_frame, detections = detector.detect(rgb)
            except Exception as e:
                print("Detector error:", e)
                annotated_frame = rgb.copy()
                detections = []

            # collect person boxes for tracker
            person_bboxes = [
                (int(d["bbox"]["x1"]), int(d["bbox"]["y1"]), int(d["bbox"]["x2"]), int(d["bbox"]["y2"]))
                for d in detections if d.get("class_name", "").lower() == "person"
            ]
            tracks = tracker.update(person_bboxes)
            track_centroids = {tid: ((b[0]+b[2])//2, (b[1]+b[3])//2) for tid, b in tracks.items()}

            # violations
            frame_violations = 0
            max_conf_frame = 0.0
            rows_to_append = []

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

            # append to global CSV
            if rows_to_append:
                try:
                    with open(LOG_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows_to_append)
                    total_violations += len(rows_to_append)
                except Exception as e:
                    print("CSV append error:", e)

            active_alerts = frame_violations

            # fps & histories
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            fps_history.append(fps)
            conf_history.append(max_conf_frame if max_conf_frame > 0 else 0.0)
            time_history.append(datetime.now().strftime("%H:%M:%S"))

            # KPI updates
            kpi1_ph.metric("Total Frames", total_frames)
            kpi2_ph.metric("Total Violations", total_violations)
            kpi3_ph.metric("Active Alerts (this frame)", active_alerts)

            # show annotated frame
            video_ph.image(annotated_frame, channels="RGB", width='stretch')
            fps_ph.markdown(f"### ⚡ FPS: `{fps:.2f}`")

            # charts
            if frame_idx % 3 == 0:
                try:
                    df_all = pd.DataFrame(st.session_state.session_rows, columns=CSV_HEADER)
                    if not df_all.empty:
                        dist = df_all['class_name'].value_counts().reset_index()
                        dist.columns = ['violation_type','count']
                        if px:
                            fig = px.pie(dist, names='violation_type', values='count', title="Violation Distribution")
                        else:
                            fig = go.Figure(go.Pie(labels=dist['violation_type'], values=dist['count']))
                            fig.update_layout(title="Violation Distribution")
                        viol_chart_ph.plotly_chart(fig, width='stretch', key=f"viol_chart_{frame_idx}")
                except Exception as e:
                    print("Violation chart error:", e)

                try:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=list(time_history), y=list(fps_history),
                                              mode='lines+markers', name='FPS'))
                    fig2.add_trace(go.Scatter(x=list(time_history), y=list(conf_history),
                                              mode='lines+markers', name='Max Conf'))
                    fig2.update_layout(height=300, title="FPS & Confidence over time", xaxis_title="Time")
                    fps_chart_ph.plotly_chart(fig2, width='stretch', key=f"fps_chart_{frame_idx}")
                except Exception as e:
                    print("Line chart error:", e)

            # logs & single-session export
            try:
                if st.session_state.session_rows:
                    df_session = pd.DataFrame(st.session_state.session_rows, columns=CSV_HEADER)
                    logs_ph.dataframe(df_session.tail(200), width='stretch')

                    # prepare download CSV bytes
                    st.session_state.download_csv_bytes = df_session.to_csv(index=False).encode("utf-8")
                    st.session_state.download_file_name = f"violations_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                    # create download button only once
                    if not st.session_state.download_button_created:
                        st.session_state.download_button_created = True
                        st.download_button(
                            "📥 Download Current Session Logs",
                            st.session_state.download_csv_bytes,
                            file_name=st.session_state.download_file_name,
                            key="download_session"
                        )
            except Exception as e:
                print("Session logs update error:", e)

            time.sleep(0.001)

    finally:
        cap.release()
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.download_button_created = False
        st.success("Detection stopped.")

else:
    st.info("Use the sidebar controls to Start / Pause / Resume / Stop.")
