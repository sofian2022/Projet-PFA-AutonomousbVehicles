import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import torch

class LicensePlateRecognizer:
    def __init__(self, model_path='D:\Autonomous_Vehicule\CarLicencePlate\licensePlateRecog_mongo_emailAlert__Project\license_plate_detector.pt', ocr_langs=['en']):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(ocr_langs, gpu=torch.cuda.is_available())
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

    def preprocess_license_plate(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return morph

    def recognize_plate(self, plate_img):
        processed_img = self.preprocess_license_plate(plate_img)
        results = self.reader.readtext(processed_img)
        plate_text = ""
        for _, text, conf in results:
            if conf > 0.5:
                plate_text += text + " "
        return plate_text.strip()

    def detect_and_recognize(self, image, class_filter=None):
        processed_img = image.copy()
        results = self.model.predict(image, conf=self.conf_threshold, iou=self.iou_threshold)
        plates_info = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                if class_filter is not None and box.cls[0].item() != class_filter:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = image[y1:y2, x1:x2]
                if plate_img.size == 0:
                    continue
                plate_text = self.recognize_plate(plate_img)

                # Draw rectangle and plate number
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_img, plate_text, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                plates_info.append({
                    'bbox': (x1, y1, x2, y2),
                    'text': plate_text,
                    'confidence': float(box.conf)
                })

        return processed_img, plates_info


def run_webcam(model_path='./license_plate_detector.pt'):
    lp_recognizer = LicensePlateRecognizer(model_path=model_path)

    cap = cv2.VideoCapture(0)  # Use the default camera (index 0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("[INFO] Starting webcam license plate recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, plates_info = lp_recognizer.detect_and_recognize(frame)

        # Show the processed frame
        cv2.imshow("License Plate Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam(model_path="D:\Autonomous_Vehicule\CarLicencePlate\licensePlateRecog_mongo_emailAlert__Project\license_plate_detector.pt")  # Update path if needed
