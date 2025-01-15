from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load the trained model
model = YOLO("runs/segment/train3/weights/best.pt")

# Run inference on a test image
results = model.predict(source="searchwindows/frame_1520.jpg", task="segment", conf=0.6, save=False)

# Plot the result using Matplotlib
for result in results:
    # Get the annotated image (with predictions overlaid)
    img_with_predictions = result.plot()

    # Display the image using Matplotlib
    plt.imshow(cv2.cvtColor(img_with_predictions, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
