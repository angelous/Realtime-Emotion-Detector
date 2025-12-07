import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import av
from ultralytics import YOLO
import numpy as np
import time

MODEL_PATH = 'model_yolo_v8/weights/best.pt'

st.title("Emotion Detection")

model = YOLO(MODEL_PATH)

if "processor" not in st.session_state:
    st.session_state.processor = None

# st_autorefresh(interval=500, key="data_refresh")

# video transformer
class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        self.current_label = "None"
        self.warning = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # YOLO detection
        results = model.predict(img, verbose=False)

        # ambil label pertama
        if len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes[0].cls[0])
            label = model.names[cls_id]

            self.current_label = label
            self.warning = (label.lower() == "sleepy")
        else:
            self.current_label = "None"
            self.warning = False

        # render annotated frame
        annotated = results[0].plot()

        return annotated

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="yolo-drowsy",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


label_placeholder = st.empty()
warning_placeholder = st.empty()

while True:
    if webrtc_ctx.video_processor:
        label = webrtc_ctx.video_processor.current_label
        warn = webrtc_ctx.video_processor.warning

        label_placeholder.markdown(f"### 🧠 **{label}**")

        if warn:
            warning_placeholder.markdown(
                "<span style='color:red; font-size:24px;'>⚠️ WARNING: You look sleepy!</span>",
                unsafe_allow_html=True
            )
        else:
            warning_placeholder.markdown("")
    
    time.sleep(0.1)