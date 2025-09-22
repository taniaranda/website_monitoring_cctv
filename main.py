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
RTMP_URL = "rtmp://localhost/live/testcctv"
TARGET_FPS = 30
YOLO_MODEL_PATH = "model/best.pt"
TARGET_CLASSES = ["bus", "car", "motorcycle", "truck"]



# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Init DeepSORT
tracker = DeepSort(max_age=30)

# Vehicle count state
vehicle_count = {cls: 0 for cls in TARGET_CLASSES}
total_count = 0

# ====== LINE CONFIG ======
line_coords = np.array([[1034, 432], [1485, 729]])  # updated coordinates
track_last_positions = {}  # remember last positions of tracks

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check_line_cross(p1, p2, l1, l2):
    return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)
# ==========================

# Create a fallback image
fallback_image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(fallback_image, "Stream Unavailable", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
ret, fallback_buffer = cv2.imencode('.jpg', fallback_image)
fallback_frame = fallback_buffer.tobytes()

# OpenCV capture object
def initialize_stream():
    logger.info(f"Attempting to open OBS RTMP stream: {RTMP_URL}")
    cap = cv2.VideoCapture(RTMP_URL)
    if not cap.isOpened():
        logger.error(f"Failed to open OBS RTMP stream {RTMP_URL}")
    else:
        logger.info("OBS RTMP stream opened successfully")
    return cap

cap = initialize_stream()

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
            time.sleep(2)
            continue
# YOLOV11
        try:
            results = model.predict(frame, imgsz=640, conf=0.5, iou=0.5, verbose=False)[0]
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
# DEEPSORT 
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
                    # center of the box
                    x1, y1, x2, y2 = track.to_ltrb()
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)

                    prev_pos = track_last_positions.get(track.track_id)
                    if prev_pos is not None:
                        crossed = check_line_cross(prev_pos, (cx, cy),
                                                   tuple(line_coords[0]), tuple(line_coords[1]))
                        if crossed and not hasattr(track, "counted"):
                            vehicle_count[cls_name] += 1
                            total_count += 1
                            track.counted = True

                    track_last_positions[track.track_id] = (cx, cy)

        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            cls_name = track.det_class
            track_id = track.track_id
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {track_id}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # draw the red line
        cv2.line(frame, tuple(line_coords[0]), tuple(line_coords[1]), (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            annotated_frame = buffer.tobytes()
            latest_frame[0] = annotated_frame
        else:
            latest_frame[0] = fallback_frame

        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed_time)
        time.sleep(sleep_time)

latest_frame = [None]

def generate_video():
    while True:
        if latest_frame[0] is None:
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

t = threading.Thread(target=run_detection, daemon=True)
t.start()

# ==================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)