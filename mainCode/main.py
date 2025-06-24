import cv2
import numpy as np
import time
import threading
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# ------------------- GPIO & Motor Setup ------------------- #
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# HC-SR04 Distance Sensor
TRIG = 25
ECHO = 26
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Motor Pins
ENA = 17
IN1 = 27
IN2 = 22
ENB = 18
IN3 = 23
IN4 = 24
GPIO.setup([ENA, IN1, IN2, ENB, IN3, IN4], GPIO.OUT)

# LED Pin
LED_PIN = 2
GPIO.setup(LED_PIN, GPIO.OUT)

# Motor PWM
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# Motor control functions
def move_forward(speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def stop_motors():
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

def led_on():
    GPIO.output(LED_PIN, GPIO.HIGH)

def led_off():
    GPIO.output(LED_PIN, GPIO.LOW)

# ------------------- Camera Stream Class ------------------- #
class CameraStream:
    def __init__(self, resolution=(640, 480)):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": resolution})
        self.picam2.configure(config)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

    def start(self):
        self.picam2.start()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            frame = self.picam2.capture_array()
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.picam2.close()

# ------------------- Distance Measurement ------------------- #
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

# ------------------- MobileNet SSD Setup ------------------- #
CLASSES = ["aeroplane", "bicycle", "bus", "car", "cat", "cow", "dog", "horse", "motorbike", "person", "train"]
VALID_CLASS_IDS = [0, 1, 6, 7, 8, 9, 10, 11, 14, 15, 19]
CUSTOM_CLASS_MAPPING = {
    0: "aeroplane", 1: "bicycle", 6: "bus", 7: "car", 8: "cat", 9: "cow", 10: "dog", 11: "horse",
    14: "motorbike", 15: "person", 19: "train"
}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# ------------------- Traffic Sign Recognition Setup ------------------- #
MODEL_PATH = "model/best_float32.tflite"
INPUT_SIZE = 320
CONFIDENCE_THRESHOLD = 0.25
class_names = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(frame):
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.ones((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8) * 114
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    tensor = padded.astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)

def postprocess(outputs, conf_thresh=0.25):
    preds = np.squeeze(outputs).T
    scores = preds[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    valid = confidences >= conf_thresh
    results = []
    for i in range(len(class_ids)):
        if valid[i] and class_ids[i] < len(class_names):
            label = class_names[class_ids[i]]
            results.append((label, float(confidences[i])))
    return results

def detect_signs(frame):
    input_tensor = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return postprocess(output_data, CONFIDENCE_THRESHOLD)



# ------------------- Main ------------------- #

def main():
    motors_running = True
    current_speed = 30
    move_forward(current_speed)
    led_off()

    camera = CameraStream()
    camera.start()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            display_frame = cv2.resize(frame_rgb, (400, int(frame_rgb.shape[0] * 400 / frame_rgb.shape[1])))
            h_f, w_f = display_frame.shape[:2]

            distance = measure_distance()
            dist_text = f"Distance: {distance:.2f} cm" if distance != -1 else "Distance: N/A"
            cv2.putText(display_frame, dist_text, (10, h_f - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Emergency stop if distance <= 30cm and car is moving
            if distance != -1 and distance <= 30 and current_speed > 0:
                print(f"🚧 Obstacle too close ({distance} cm)! Reducing speed to 0.")
                stop_motors()
                led_on()
                time.sleep(1)
                current_speed = 0
                motors_running = False

            blob = cv2.dnn.blobFromImage(cv2.resize(display_frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            detected_objects = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    if idx not in VALID_CLASS_IDS:
                        continue
                    label = CUSTOM_CLASS_MAPPING[idx]
                    box = detections[0, 0, i, 3:7] * np.array([w_f, h_f, w_f, h_f])
                    startX, startY, endX, endY = box.astype("int")
                    color = COLORS[VALID_CLASS_IDS.index(idx)]
                    detected_objects.append(label)
                    cv2.rectangle(display_frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(display_frame, f"{label}: {confidence*100:.1f}%", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            predictions = detect_signs(frame)
            stop_detected = False
            new_speed = None

            if predictions:
                print("\n🚦 Traffic Signs Detected:")
            else:
                print("🚫 No valid traffic signs detected! Forcing stop.")
                stop_motors()
                led_on()
                motors_running = False
                current_speed = 0

            for (label, score) in predictions:
                print(f"  ➤ {label} (confidence: {score:.2f})")
                if label in ["Stop", "Red Light"] and score >= 0.8:
                    stop_detected = True
                elif score >= 0.8:
                    if label == "Speed Limit 20":
                        new_speed = 40
                    elif label == "Speed Limit 60":
                        new_speed = 60
                    elif label == "Speed Limit 90":
                        new_speed = 90
                    elif label == "Speed Limit 120":
                        new_speed = 100
                elif label == "Green Light":
                        new_speed = 60
                y_offset = 20 + 20 * predictions.index((label, score))
                cv2.putText(display_frame, f"{label} ({score:.2f})", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if stop_detected and motors_running:
                print("🛑 Stop or Red Light detected!")
                stop_motors()
                led_on()
                motors_running = False
            elif not stop_detected and not motors_running and new_speed is not None:
                print("✅ Resuming movement.")
                move_forward(current_speed)
                led_off()
                motors_running = True
            elif new_speed is not None and motors_running and new_speed != current_speed:
                print(f"⚙ Adjusting speed to {new_speed}%")
                move_forward(new_speed)
                current_speed = new_speed

            if detected_objects:
                print(f"Objects: {', '.join(detected_objects)} | Distance: {distance} cm")

            cv2.imshow("Detection + Traffic Signs", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        stop_motors()
        led_off()
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("✅ Program ended safely.")




if __name__ == "__main__":
    main()
