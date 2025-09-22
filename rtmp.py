from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import threading
import time
import asyncio
import torch
import numpy as np
import yt_dlp
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===================== CONFIG =====================
# YouTube Live URL
YT_URL = "https://www.youtube.com/watch?v=tuCzMVja0wI&pp=ygURY2N0diBzdHJlYW0gamFsYW4%3D"  # Your YouTube live URL
# RTMP stream URL from OBS (replace with your OBS RTMP endpoint)
RTMP_URL = "rtmp://localhost/live/testcctv"  # For local RTMP server, or e.g., rtmp://a.rtmp.youtube.com/live2/your_stream_key
YOLO_MODEL_PATH = "trained/beste.pt"  # Your trained YOLOv11 weights
TARGET_CLASSES = ["bus", "car", "motorcycle", "truck"]

# FPS Configuration
TARGET_FPS = 10  # Desired FPS for processing

# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Init DeepSORT
tracker = DeepSort(max_age=30)

# Vehicle count state
vehicle_count = {cls: 0 for cls in TARGET_CLASSES}
total_count = 0

# Create a fallback image
fallback_image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(fallback_image, "Stream Unavailable", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
ret, fallback_buffer = cv2.imencode('.jpg', fallback_image)
fallback_frame = fallback_buffer.tobytes()

# OpenCV capture object
def get_stream_url():
    logger.info("Fetching YouTube stream URL")
    try:
        ydl_opts = {'format': 'best'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(YT_URL, download=False)
            return info["url"]
    except Exception as e:
        logger.error(f"Failed to fetch YouTube stream URL: {e}")
        return None

# Initialize stream (try OBS RTMP first, fall back to YouTube)
def initialize_stream():
    logger.info(f"Attempting to open OBS RTMP stream: {RTMP_URL}")
    cap = cv2.VideoCapture(RTMP_URL)
    if not cap.isOpened():
        logger.warning(f"Failed to open OBS RTMP stream {RTMP_URL}, falling back to YouTube stream")
        stream_url = get_stream_url()
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logger.error(f"Failed to open YouTube stream {stream_url}")
            else:
                logger.info("YouTube stream opened successfully")
        else:
            logger.error("No valid stream available")
    else:
        logger.info("OBS RTMP stream opened successfully")
    return cap

cap = initialize_stream()

# Lock for thread-safe updates
count_lock = threading.Lock()

# ==================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "location_name": "Orinunggu Kambu",
        "cctv_connected": cap.isOpened()
    })

@app.get("/api/vehicle-count")
async def get_vehicle_count():
    with count_lock:
        total = sum(vehicle_count.values())
        return JSONResponse({
            "data": vehicle_count,
            "total": total
        })

# ----------------- Detection + Tracking -----------------
def run_detection():
    global cap, total_count
    frame_interval = 1.0 / TARGET_FPS

    while True:
        start_time = time.time()

        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            logger.warning("Failed to read valid frame from stream, attempting to reconnect")
            cap.release()
            cap = initialize_stream()
            latest_frame[0] = fallback_frame
            time.sleep(2)  # Increased delay to avoid rapid reconnection
            continue

        logger.debug("Frame read successfully, processing with YOLO")
        # YOLOv11 inference
        try:
            results = model.predict(frame, imgsz=640, conf=0.1, iou=0.5, verbose=False)[0]
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            latest_frame[0] = fallback_frame
            time.sleep(frame_interval)
            continue

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(((x1, y1, x2-x1, y2-y1), conf, cls_name))

        # DeepSORT tracking
        try:
            tracks = tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            logger.error(f"DeepSORT tracking failed: {e}")
            latest_frame[0] = fallback_frame
            time.sleep(frame_interval)
            continue

        with count_lock:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                cls_name = track.det_class
                if cls_name in TARGET_CLASSES:
                    if not hasattr(track, "counted"):
                        vehicle_count[cls_name] += 1
                        total_count += 1
                        track.counted = True

        # Draw detections
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            cls_name = track.det_class
            track_id = track.track_id
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {track_id}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            annotated_frame = buffer.tobytes()
            latest_frame[0] = annotated_frame
            logger.debug("Frame encoded and stored successfully")
        else:
            logger.warning("Failed to encode frame to JPEG")
            latest_frame[0] = fallback_frame

        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed_time)
        time.sleep(sleep_time)

# Storage for latest annotated frame
latest_frame = [None]

def generate_video():
    while True:
        if latest_frame[0] is None:
            logger.debug("No frame available, yielding fallback")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fallback_frame + b'\r\n')
            time.sleep(0.033)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame[0] + b'\r\n')
        time.sleep(0.033)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace; boundary=frame")

# Start detection thread
t = threading.Thread(target=run_detection, daemon=True)
t.start()

# ==================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)