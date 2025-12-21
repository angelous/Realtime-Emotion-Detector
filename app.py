import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
import time
from collections import deque
import cv2

MODEL_PATH = 'model_yolo_v8/weights/best.pt'
ALERT_SOUND_PATH = "alert.wav" # Belum ada fungsi untuk ini

st.title("Sleepiness and Drowsiness Detection")

model = YOLO(MODEL_PATH)

if "processor" not in st.session_state:
    st.session_state.processor = None

st_autorefresh(interval=500, key="data_refresh")

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
            {
                "urls": "turn:relay1.expressturn.com:3480?transport=udp",
                "username": "000000002080719928",
                "credential": "+SHJnrTpaYjzqu9zIi04haY0qnw="
            }
        ]
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

        if warn:
            label_placeholder.markdown("### 💤 **SLEEPY**")
            warning_placeholder.markdown(
                "<span style='color:red; font-size:24px;'>⚠️ WARNING: You look sleepy!</span>",
                unsafe_allow_html=True
            )
        else:
            label_placeholder.markdown("")
            warning_placeholder.markdown("")

    time.sleep(0.1)