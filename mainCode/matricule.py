import cv2
from ultralytics import YOLO
import easyocr
import torch
import time

# === Load YOLOv8 model ===
model = YOLO("./license_plate_detector.pt")  # Replace with your model path

# === Initialize EasyOCR ===
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# === Start Pi camera (or USB cam) ===
cap = cv2.VideoCapture(0)  # Use 0 for default camera

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("📸 Camera initialized. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            continue

        # Detect license plates
        results = model.predict(frame, conf=0.25, iou=0.45)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]

                if plate_img.size == 0:
                    continue

                # Preprocess plate image
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (3, 3))

                # OCR
                results = reader.readtext(morph)
                plate_text = ""
                for _, text, conf in results:
                    if conf > 0.15:
                        plate_text += text + " "
                plate_text = plate_text.strip()

                if plate_text:
                    print(f"✅ Detected Plate: {plate_text}")

        time.sleep(1)  # Adjust delay to control frame rate

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
