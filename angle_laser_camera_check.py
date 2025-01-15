import cv2
import numpy as np
from ultralytics import YOLO
from rplidar import RPLidar
import threading

CAMERA_INDEX = 1  # Default camera index (adjust if needed)
CAMERA_FOV = 68  # Camera field of view in degrees
PORT_NAME = 'COM3'  # LIDAR port
# MAX_DISTANCE = 3000  # Maximum distance measurable by the LIDAR in mm

angles = []
distances = []
frame = None
stop_threads = False

def calculate_angle(bbox, frame_width):
    """
    Calculate the angle of the detected person relative to the center of the camera frame.
    :param bbox: Bounding box of the detected person (x_min, y_min, x_max, y_max).
    :param frame_width: Width of the camera frame.
    :return: Angle in degrees.
    """
    x_center = (bbox[0] + bbox[2]) / 2  # Center of the bounding box
    normalized_x = (x_center - frame_width / 2) / (frame_width / 2)  # Normalize to [-1, 1]
    angle = normalized_x * (CAMERA_FOV / 2)  # Map to FOV
    return angle

def lidar_thread():
    global angles, distances, stop_threads
    lidar = RPLidar(PORT_NAME)

    print("Starting LIDAR thread...")

    try:
        for scan in lidar.iter_scans():
            if stop_threads:
                break
            angles.clear()
            distances.clear()
            for (_, angle, distance) in scan:
                if distance > 0:
                    angles.append(round(angle, 2))
                    distances.append(distance)
    except Exception as e:
        print(f"LIDAR error: {e}")
    finally:
        lidar.stop()
        lidar.disconnect()
        print("LIDAR thread stopped.")

def camera_thread():
    global frame, angles, distances, stop_threads

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        stop_threads = True
        return

    model = YOLO('yolov8n.pt')  # Load YOLOv8 model

    print("Starting person detection... Press 'q' to stop.")

    try:
        while not stop_threads:
            ret, frame = cap.read()
            if not ret:
                print("Error: Camera frame could not be read.")
                break

            results = model.predict(frame)  # Run YOLOv8 on the current frame

            for box in results[0].boxes:  # Extract bounding boxes from results
                x_min, y_min, x_max, y_max = box.xyxy[0]  # Get bounding box coordinates
                cls = int(box.cls)  # Get class ID
                if cls == 0:  # Class 0 corresponds to 'person'
                    angle = calculate_angle((x_min, y_min, x_max, y_max), frame.shape[1])
                    angle_range_min = angle - 5
                    angle_range_max = angle + 5

                    # Find matching LIDAR angle and extract distance
                    lidar_angle = next((a for a in angles if angle_range_min <= a <= angle_range_max), None)
                    if lidar_angle is not None:
                        index = angles.index(lidar_angle)
                        distance = distances[index]
                        print(f"Detected person at angle: {angle:.2f} degrees, range: [{angle_range_min:.2f}, {angle_range_max:.2f}] degrees, distance: {distance:.2f} mm")
                    else:
                        print(f"Detected person at angle: {angle:.2f} degrees, range: [{angle_range_min:.2f}, {angle_range_max:.2f}] degrees, no matching LIDAR data")

            # Display the frame
            cv2.imshow('Camera', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_threads = True

    except KeyboardInterrupt:
        print("Stopping person detection...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera thread stopped.")

def main():
    global stop_threads

    lidar_t = threading.Thread(target=lidar_thread)
    camera_t = threading.Thread(target=camera_thread)

    lidar_t.start()
    camera_t.start()

    try:
        lidar_t.join()
        camera_t.join()
    except KeyboardInterrupt:
        print("Stopping all threads...")
        stop_threads = True
        lidar_t.join()
        camera_t.join()

if __name__ == "__main__":
    main()
