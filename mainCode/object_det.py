import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ---------- HC-SR04 Sensor Setup ----------
TRIG = 25
ECHO = 26

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def measure_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time() + 0.04
    pulse_start = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if time.time() > timeout:
            return -1

    timeout = time.time() + 0.04
    pulse_end = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if time.time() > timeout:
            return -1

    duration = pulse_end - pulse_start
    return round(duration * 17150, 2)

# ---------- MobileNet SSD Class Setup ----------
CLASSES = ["aeroplane", "bicycle", "bus", "car", "cat", "cow", "dog", "horse", "motorbike", "person", "train"]
VALID_CLASS_IDS = [0, 1, 6, 7, 8, 9, 10, 11, 14, 15, 19]
CUSTOM_CLASS_MAPPING = {
    0: "aeroplane", 1: "bicycle", 6: "bus", 7: "car",
    8: "cat", 9: "cow", 10: "dog", 11: "horse",
    14: "motorbike", 15: "person", 19: "train"
}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# ---------- Load MobileNet SSD Model ----------
print("[INFO] Loading MobileNet SSD model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# ---------- Initialize Pi Camera ----------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

# ---------- Main Loop ----------
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.resize(frame, (400, int(frame.shape[0] * 400 / frame.shape[1])))
        (h_f, w_f) = frame.shape[:2]

        # Measure distance
        distance = measure_distance()

        # Object detection (preprocessing and forward pass)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        detected_objects = []

        # Display distance
        if distance != -1:
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, h_f - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Distance: N/A", (10, h_f - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Parse detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                if idx not in VALID_CLASS_IDS:
                    continue

                label = CUSTOM_CLASS_MAPPING[idx]
                box = detections[0, 0, i, 3:7] * np.array([w_f, h_f, w_f, h_f])
                (startX, startY, endX, endY) = box.astype("int")
                color = COLORS[VALID_CLASS_IDS.index(idx)]

                detected_objects.append(label)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, f"{label}: {confidence * 100:.1f}%", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Print to console
        if detected_objects:
            print(f"[DETECTED] Objects: {', '.join(detected_objects)} | Distance: {distance} cm")
        else:
            print("No object detected.")

        # Display the live frame
        cv2.imshow("Pi Camera - Object Detection + Distance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("[INFO] Program exited and GPIO cleaned up.")
