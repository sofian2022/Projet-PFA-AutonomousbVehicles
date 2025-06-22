import cv2
import numpy as np
import time

# Custom class list and corresponding original MobileNet SSD class IDs
CLASSES = ["aeroplane", "bicycle", "bus", "car", "cat", "cow", "dog", "horse", "motorbike", "person", "train"]
VALID_CLASS_IDS = [0, 1, 6, 7, 8, 9, 10, 11, 14, 15, 19]  # IDs from original MobileNet SSD
CUSTOM_CLASS_MAPPING = {
    0: "aeroplane",
    1: "bicycle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "cow",
    10: "dog",
    11: "horse",
    14: "motorbike",
    15: "person",
    19: "train"
}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained MobileNet SSD model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Start webcam video stream
print("[INFO] starting webcam...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

# Process frames from webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to width=400 for better performance
    frame = cv2.resize(frame, (400, int(frame.shape[0] * 400 / frame.shape[1])))

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Forward pass through the network
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            if idx not in VALID_CLASS_IDS:
                continue

            custom_label = CUSTOM_CLASS_MAPPING[idx]
            class_color_index = VALID_CLASS_IDS.index(idx)

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(custom_label, confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_color_index], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_color_index], 2)

    # Display the frame
    cv2.imshow("Webcam Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
