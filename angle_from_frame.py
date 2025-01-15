import cv2
import numpy as np
from ultralytics import YOLO

CAMERA_INDEX = 1  # Default camera index (adjust if needed)
CAMERA_FOV = 68  # Camera field of view in degrees

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

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    model = YOLO('yolov8n.pt')  # Load YOLOv8 model

    print("Starting person detection... Press 'q' to stop.")

    try:
        while True:
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
                    angle_range_min = angle - 10
                    angle_range_max = angle + 10
                    print(f"Detected person at angle: {angle:.2f} degrees, range: [{angle_range_min:.2f}, {angle_range_max:.2f}] degrees")

            # Display the frame
            cv2.imshow('Camera', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping person detection...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Program stopped.")

if __name__ == "__main__":
    main()
