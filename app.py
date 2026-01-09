import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
import time
from collections import deque
import cv2
import base64

MODEL_PATH = 'model_yolo_v8/weights/best.pt'
ALERT_SOUND_PATH = "alert.wav"

st.title("Sleepiness and Drowsiness Detection")

model = YOLO(MODEL_PATH)

if "processor" not in st.session_state:
    st.session_state.processor = None

if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0

st_autorefresh(interval=500, key="data_refresh")

def play_audio_alert():
    """Play alert sound using HTML audio element"""
    try:
        with open(ALERT_SOUND_PATH, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Alert sound file not found. Please add 'alert.wav' to your project directory.")


# video transformer
class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        self.sleepy_queue = deque(maxlen=20)
        self.current_label = "None"
        self.warning = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(img, conf=0.4, verbose=False)
        boxes = results[0].boxes

        self.current_label = ""
        self.warning = False

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                label = model.names[cls_id]
                conf = float(boxes.conf[i])

                if label.lower() == "sleepy" and conf >= 0.4:
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        img,
                        "SLEEPY",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2
                    )

                    self.current_label = "sleepy"
                    self.warning = True

                    break

        return img

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="yolo-drowsy",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    video_processor_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


label_placeholder = st.empty()
warning_placeholder = st.empty()
audio_placeholder = st.empty()

while True:
    if webrtc_ctx.video_processor:
        label = webrtc_ctx.video_processor.current_label
        warn = webrtc_ctx.video_processor.warning

        if warn:
            label_placeholder.markdown("### 💤 **SLEEPY**")
            warning_placeholder.markdown(
                "<span style='color:red; font-size:24px;'>⚠️ WARNING: You look sleepy!</span>",
                unsafe_allow_html=True
            )

            current_time = time.time()
            if current_time - st.session_state.last_alert_time >= 2.0:
                with audio_placeholder.container():
                    play_audio_alert()
                st.session_state.last_alert_time = current_time
        else:
            label_placeholder.markdown("")
            warning_placeholder.markdown("")

    time.sleep(0.1)