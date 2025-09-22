import cv2
import threading
import queue
import os
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
RTMP_URL = "rtmp://localhost/live/testcctv"
MODEL_PATH = r"trained/bestestfix.pt"
TARGET_CLASSES = ["bus", "car", "motorcycle", "truck"]
WINDOW_NAME = "Vehicle Tracker - Orinunggu Kambu"
INFERENCE_SIZE = (320, 320)
CONF_THRESHOLD = 0.25
TARGET_FPS = 30  # For reference only; not enforced
INFERENCE_INTERVAL = 3

# Line crossing configuration
line_coords = np.array([[1034, 432], [1485, 729]])  # updated coordinates
track_last_positions = {}  # remember last positions of tracks
vehicle_count = {cls: 0 for cls in TARGET_CLASSES}
total_count = 0

# Thread-safe variables
latest_results = None
results_lock = threading.Lock()
count_lock = threading.Lock()

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file {MODEL_PATH} not found.")
    exit(1)

# Load YOLO model
logger.info(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Initialize DeepSort
deepsort = DeepSort(max_age=30, nn_budget=100)

# Line crossing detection
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check_line_cross(p1, p2, l1, l2):
    return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)

# Threaded VideoCapture for low-latency streaming
class StreamVideoCapture:
    def __init__(self, url, queue_size=3):
        self.url = url
        self.queue_size = queue_size
        self.Q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.stream = None
        self.reconnect()

    def reconnect(self):
        if self.stream:
            self.stream.release()
        logger.info(f"Attempting to open RTMP stream: {self.url}")
        self.stream = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.stream.isOpened():
            logger.error(f"Failed to open RTMP stream {self.url}")
            return False
        return True

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                if not self.reconnect():
                    time.sleep(1)
                    continue
            ret, frame = self.stream.read()
            if not ret or frame is None or frame.size == 0:
                logger.warning("Failed to read frame, reconnecting")
                continue
            try:
                self.Q.put_nowait(frame)
            except queue.Full:
                try:
                    self.Q.get_nowait()  # Drop oldest frame
                except queue.Empty:
                    pass
                self.Q.put_nowait(frame)

    def read(self):
        try:
            return True, self.Q.get(timeout=0.1)
        except queue.Empty:
            return False, None

    def stop(self):
        self.stopped = True
        if self.stream:
            self.stream.release()

# Set low-latency FFMPEG options
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Suppress MSMF backend issues on Windows

# Initialize stream
cap = StreamVideoCapture(RTMP_URL).start()

# Get stream resolution (wait for first frame)
frame_width, frame_height = 0, 0
while frame_width == 0:
    ret, frame = cap.read()
    if ret:
        frame_height, frame_width = frame.shape[:2]
    time.sleep(0.1)
logger.info(f"Stream resolution: {frame_width}x{frame_height}")

# Inference thread
def inference_thread(frame_queue):
    global latest_results
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
            results = model.predict(frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, verbose=False)
            with results_lock:
                latest_results = results[0]
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")

# Initialize queue and thread
frame_queue = queue.Queue(maxsize=5)
inference_thread = threading.Thread(target=inference_thread, args=(frame_queue,))
inference_thread.daemon = True
inference_thread.start()

# Create fallback image
fallback_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
cv2.putText(fallback_image, "Stream Unavailable", (50, frame_height//2), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Main loop
fps_start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cv2.imshow(WINDOW_NAME, fallback_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5)  # Brief wait before next read attempt
        continue

    frame_count += 1

    # Send frame to inference thread
    if frame_count % INFERENCE_INTERVAL == 0:
        resized_frame = cv2.resize(frame, INFERENCE_SIZE)
        try:
            frame_queue.put_nowait(resized_frame)
        except queue.Full:
            while not frame_queue.empty():
                frame_queue.get()
            frame_queue.put_nowait(resized_frame)

    # Process detections
    annotated_frame = frame.copy()
    with results_lock:
        if latest_results is not None:
            orig_h, orig_w = frame.shape[:2]
            inf_h, inf_w = INFERENCE_SIZE
            scale_x = orig_w / inf_w
            scale_y = orig_h / inf_h

            detections = []
            detection_confs = []
            for box in latest_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0])
                cls_name = latest_results.names[cls]
                if cls_name in TARGET_CLASSES:
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_name))
                    detection_confs.append(conf)

            # Update DeepSort
            try:
                tracks = deepsort.update_tracks(detections, frame=frame)
            except Exception as e:
                logger.error(f"DeepSort tracking failed: {e}")
                continue

            # Process tracks for counting
            with count_lock:
                for i, track in enumerate(tracks):
                    if not track.is_confirmed():
                        continue
                    cls_name = track.det_class
                    if cls_name in TARGET_CLASSES:
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

                        # Draw bounding box and label
                        conf = detection_confs[i] if i < len(detection_confs) else 0.0
                        label = f"ID {track.track_id} {cls_name} {conf:.2f}"
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), 
                                    (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw counting line
    cv2.line(annotated_frame, tuple(line_coords[0]), tuple(line_coords[1]), (0, 0, 255), 3)

    # Display vehicle counts
    y_pos = 30
    cv2.putText(annotated_frame, f"Total: {total_count}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for cls, count in vehicle_count.items():
        y_pos += 30
        cv2.putText(annotated_frame, f"{cls.capitalize()}: {count}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Calculate and display FPS
    elapsed_time = time.time() - fps_start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    if frame_count % 10 == 0:
        logger.info(f"FPS: {fps:.2f}")
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (frame_width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display frame
    cv2.imshow(WINDOW_NAME, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.stop()
cv2.destroyAllWindows()
logger.info("Vehicle tracking process completed.")