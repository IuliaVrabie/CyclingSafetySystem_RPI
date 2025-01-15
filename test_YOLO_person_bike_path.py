from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the segmentation model for bike path
segmentation_model = YOLO("runs/segment/train3/weights/best.pt")

# Load the object detection model (supports both person and bike detection)
object_detection_model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 model

# Step 1: Segment the bike path
results = segmentation_model.predict(source="searchwindows/frame_0870.jpg", task="segment", conf=0.6, save=False)

# Get the first result (assuming single image inference)
segmentation_result = results[0]

# Extract the mask for the segmented path
mask = segmentation_result.masks.data[0].cpu().numpy()  # Convert the mask to a NumPy array

# Step 2: Create a binary mask
binary_mask = (mask > 0.5).astype(np.uint8)  # Threshold the mask to create a binary mask
binary_mask = cv2.resize(binary_mask, (segmentation_result.orig_shape[1], segmentation_result.orig_shape[0]))  # Resize mask to original image size

# Step 3: Apply the mask to isolate the ROI
original_image = cv2.imread("searchwindows/frame_0870.jpg")
masked_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)

# Step 4: Run object detection on the ROI
# Convert the masked image to RGB (YOLO expects RGB format)
masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
detection_results = object_detection_model.predict(source=masked_image_rgb, conf=0.5, save=False)

# Filter for person and bike detections (assuming YOLOv8 class IDs: 0=person, 1=bicycle)
filtered_detections = []
for result in detection_results:
    for box in result.boxes:
        if box.cls in [0, 1]:  # Keep only person (0) and bicycle (1)
            filtered_detections.append(box)

# Visualize Results
plt.figure(figsize=(12, 6))

# Plot the segmentation result
plt.subplot(1, 2, 1)
segmentation_overlay = segmentation_result.plot()
plt.imshow(cv2.cvtColor(segmentation_overlay, cv2.COLOR_BGR2RGB))
plt.title("Segmentation (Bike Path)")
plt.axis("off")

# Plot the detection results
plt.subplot(1, 2, 2)
detection_overlay = detection_results[0].plot()  # Overlay all detections
plt.imshow(cv2.cvtColor(detection_overlay, cv2.COLOR_BGR2RGB))
plt.title("Detection (Filtered for Person and Bike)")
plt.axis("off")

plt.show()

# Step 5: Output detection details
for box in filtered_detections:
    print(f"Class: {box.cls}, Confidence: {box.conf}, Bounding Box: {box.xyxy}")
