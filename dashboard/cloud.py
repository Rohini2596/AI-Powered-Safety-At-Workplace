import time
import av
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from collections import deque
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ===================== CONFIG =====================
st.set_page_config(
    page_title="PPE Real-Time Monitoring",
    page_icon="🦺",
    layout="wide"
)

MODEL_PATH = "models/yolov8_ppe.pt"

VIOLATION_CLASSES = {
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest"
}

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ===================== SESSION STATE =====================
if "fps_hist" not in st.session_state:
    st.session_state.fps_hist = deque(maxlen=100)
    st.session_state.time_hist = deque(maxlen=100)
    st.session_state.violations = 0
    st.session_state.last_update = time.time()

# ===================== UI =====================
st.markdown("## 🦺 Real-Time PPE Compliance Monitoring")
st.caption("Streamlit + WebRTC • True real-time detection • Accurate FPS")

m1, m2 = st.columns(2)
fps_metric = m1.metric("FPS", "0.0")
vio_metric = m2.metric("Violations", "0")

video_col, chart_col = st.columns([2, 1])
chart_placeholder = chart_col.empty()

# ===================== VIDEO PROCESSOR =====================
class PPEProcessor(VideoTransformerBase):
    def __init__(self):
        self.prev_time = time.time()

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=0.5, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 255, 0)
            if label in VIOLATION_CLASSES:
                color = (0, 0, 255)
                st.session_state.violations += 1

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # FPS calculation
        now = time.time()
        fps = 1 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        st.session_state.fps_hist.append(fps)
        st.session_state.time_hist.append(time.strftime("%H:%M:%S"))

        return img

# ===================== WEBRTC =====================
with video_col:
    webrtc_ctx = webrtc_streamer(
        key="ppe-realtime",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PPEProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

# ===================== LIVE UI UPDATE =====================
if webrtc_ctx.state.playing and st.session_state.fps_hist:
    avg_fps = sum(st.session_state.fps_hist) / len(st.session_state.fps_hist)
    fps_metric.metric("FPS", f"{avg_fps:.2f}")
    vio_metric.metric("Violations", st.session_state.violations)

    fig = go.Figure()
    fig.add_scatter(
        x=list(st.session_state.time_hist),
        y=list(st.session_state.fps_hist),
        mode="lines"
    )
    fig.update_layout(
        title="Live FPS Trend",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)
