import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------------
# Threaded RTSP Stream Class
# ---------------------------
class RTSPStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# ---------------------------
# Setup
# ---------------------------
RTSP_URL = "rtsp://admin:admin@180.245.207.84/V_ENC_000"  # your CCTV RTSP
YOLO_MODEL_PATH = "model/best.pt"
TARGET_CLASSES = ["bus", "car", "motorcycle", "truck"]
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# DeepSORT tracker
tracker = DeepSort(max_age=30)

# Initialize RTSP stream
cap = RTSPStream(RTSP_URL)

# Vehicle count state
vehicle_count = {cls: 0 for cls in TARGET_CLASSES}
total_count = 0

# Line crossing config
line_coords = np.array([(734, 514), (1627, 795)])
track_last_positions = {}

# Helper functions
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check_line_cross(p1, p2, l1, l2):
    return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)


# ---------------------------
# Main loop
# ---------------------------
while True:
    start_time = time.time()

    frame = cap.read()
    if frame is None:
        continue

    # --- YOLOv11 detection ---
    results = model.predict(frame, imgsz=480, conf=0.3, iou=0.45, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name in TARGET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(((x1, y1, x2 - x1, y2 - y1), conf, cls_name))

    # --- DeepSORT tracking ---
    tracks = tracker.update_tracks(detections, frame=frame)

    # Counting logic
    for track in tracks:
        if not track.is_confirmed():
            continue
        cls_name = track.det_class
        if cls_name in TARGET_CLASSES:
            # center of the box
            x1, y1, x2, y2 = track.to_ltrb()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            prev_pos = track_last_positions.get(track.track_id)
            if prev_pos is not None:
                crossed = check_line_cross(prev_pos, (cx, cy),
                                           tuple(line_coords[0]), tuple(line_coords[1]))
                if crossed and not hasattr(track, "counted"):
                    vehicle_count[cls_name] += 1
                    total_count += 1
                    track.counted = True

            track_last_positions[track.track_id] = (cx, cy)

    # Draw tracked bounding boxes (with ID, class, conf)
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = track.to_ltrb()
        cls_name = track.det_class
        track_id = track.track_id
        conf = getattr(track, "det_conf", None)
        if conf is None:
            conf = 0.0  # default if DeepSORT doesn’t give a conf

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id} {cls_name} {conf:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    # Draw the red line
    cv2.line(frame, tuple(line_coords[0]), tuple(line_coords[1]), (0, 0, 255), 3)

    # Show vehicle counts
    y_offset = 40
    for cls, count in vehicle_count.items():
        cv2.putText(frame, f"{cls}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 30
    cv2.putText(frame, f"Total: {total_count}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Resize for display only
    display_width = 640
    aspect_ratio = frame.shape[1] / frame.shape[0]
    display_height = int(display_width / aspect_ratio)
    resized_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    # Display
    cv2.imshow("YOLOv11 + DeepSORT Vehicle Counting", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Limit FPS
    elapsed = time.time() - start_time
    fps = 1.0 / elapsed if elapsed > 0 else 0
    print(f"[INFO] Current FPS: {fps:.2f}")

cap.stop()
cv2.destroyAllWindows()
