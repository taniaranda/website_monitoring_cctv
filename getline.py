import cv2

# Change this to your RTMP stream or a test video file
VIDEO_SOURCE = "rtmp://localhost/live/testcctv"

def get_line_coordinates(video_source=VIDEO_SOURCE):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("❌ Failed to open video stream. Check your RTMP or video path.")
        return

    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Clicked at: ({x}, {y})")

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback)

    print("👉 Click TWO points on the video window to define your red line.")
    print("👉 Press 'q' when done.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Draw selected points
        for p in points:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)

        # Draw line if two points are chosen
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(points) == 2:
        print(f"\n✅ Use these coordinates in your code:\nline_coords = np.array([{points[0]}, {points[1]}])")
    else:
        print("⚠️ You didn’t pick two points!")

if __name__ == "__main__":
    get_line_coordinates()
