import cv2
import numpy as np
from ultralytics import YOLO
from rplidar import RPLidar
import threading
import time
import pygame

CAMERA_INDEX = 1  # Default camera index (adjust if needed)
CAMERA_FOV = 68  # Camera field of view in degrees
PORT_NAME = 'COM3'  # LIDAR port
MAX_DISTANCE_MM = 2000  # Maximum distance threshold in mm (2 meters)

angles = []
distances = []
frame = None
stop_threads = False

# Load models
segmentation_model = YOLO("runs/segment/train3/weights/best.pt")
object_detection_model = YOLO("yolov8n.pt")

# Log file
log_file = "detection_log.txt"

# Initialize logging
def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")

def calculate_angle(bbox, frame_width):
    """
    Calculate the angle of the detected object relative to the center of the camera frame.
    :param bbox: Bounding box of the detected object (x_min, y_min, x_max, y_max).
    :param frame_width: Width of the camera frame.
    :return: Angle in degrees.
    """
    x_center = (bbox[0] + bbox[2]) / 2  # Center of the bounding box
    normalized_x = (x_center - frame_width / 2) / (frame_width / 2)  # Normalize to [-1, 1]
    angle = normalized_x * (CAMERA_FOV / 2)  # Map to FOV
    return angle

def play_alarm():
    """Play a sound alarm."""
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.wav")  # Replace 'alarm.wav' with the path to your alarm sound file
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def lidar_thread():
    global angles, distances, stop_threads
    lidar = RPLidar(PORT_NAME)

    log_message("Starting LIDAR thread...")

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
        log_message(f"LIDAR error: {e}")
    finally:
        lidar.stop()
        lidar.disconnect()
        log_message("LIDAR thread stopped.")

def camera_thread():
    global frame, angles, distances, stop_threads

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        log_message("Error: Camera could not be opened.")
        stop_threads = True
        return

    log_message("Starting detection...")

    try:
        while not stop_threads:
            ret, frame = cap.read()
            if not ret:
                log_message("Error: Camera frame could not be read.")
                break

            # Step 1: Segment the bike path
            segmentation_results = segmentation_model.predict(source=frame, task="segment", conf=0.6, save=False)
            segmentation_result = segmentation_results[0]

            # Extract and process mask
            mask = segmentation_result.masks.data[0].cpu().numpy()
            binary_mask = (mask > 0.5).astype(np.uint8)
            binary_mask = cv2.resize(binary_mask, (segmentation_result.orig_shape[1], segmentation_result.orig_shape[0]))
            masked_image = cv2.bitwise_and(frame, frame, mask=binary_mask)

            # Step 2: Detect objects in the masked region
            detection_results = object_detection_model.predict(source=masked_image, conf=0.5, save=False)

            for result in detection_results:
                for box in result.boxes:
                    if box.cls in [0, 1]:  # Class 0 = person, Class 1 = bike
                        class_name = "person" if box.cls == 0 else "bike"
                        x_min, y_min, x_max, y_max = box.xyxy[0]
                        angle = calculate_angle((x_min, y_min, x_max, y_max), frame.shape[1])
                        angle_range_min = angle - 5
                        angle_range_max = angle + 5

                        # Match LIDAR data
                        lidar_angle = next((a for a in angles if angle_range_min <= a <= angle_range_max), None)
                        if lidar_angle is not None:
                            index = angles.index(lidar_angle)
                            distance = distances[index]
                            if distance < MAX_DISTANCE_MM:
                                log_message(f"Alarm: {class_name.capitalize()} detected at {distance / 1000:.2f} meters.")
                                play_alarm()
                            else:
                                log_message(f"{class_name.capitalize()} detected at {distance / 1000:.2f} meters.")
                        else:
                            log_message(f"{class_name.capitalize()} detected, no matching LIDAR data.")

    except KeyboardInterrupt:
        log_message("Stopping detection...")

    finally:
        cap.release()
        log_message("Camera thread stopped.")

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
        log_message("Stopping all threads...")
        stop_threads = True
        lidar_t.join()
        camera_t.join()

if __name__ == "__main__":
    main()
