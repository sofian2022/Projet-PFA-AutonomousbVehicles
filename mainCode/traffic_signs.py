import time
import numpy as np
import cv2
import threading
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# === GPIO CONFIGURATION ===
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor 1
ENA = 17
IN1 = 27
IN2 = 22

# Motor 2
ENB = 18
IN3 = 23
IN4 = 24

# LED
LED_PIN = 2

# Setup GPIO
GPIO.setup([ENA, IN1, IN2, ENB, IN3, IN4, LED_PIN], GPIO.OUT)

# PWM setup
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# === MOTOR & LED FUNCTIONS ===
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

# === MODEL CONFIG ===
MODEL_PATH = "model/best_float32.tflite"
INPUT_SIZE = 320
CONFIDENCE_THRESHOLD = 0.25

class_names = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]

# Initial motor state
motors_running = True
current_speed = 30
move_forward(current_speed)
led_off()

# === LOAD TFLITE MODEL ===
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite model loaded")

# === CAMERA STREAM THREAD ===
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

# === PREPROCESS FRAME ===
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

# === POSTPROCESS DETECTIONS ===
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

# === DETECTION ===
def detect(frame):
    input_tensor = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return postprocess(output_data, CONFIDENCE_THRESHOLD)

# === MAIN LOOP ===
def main():
    global motors_running, current_speed
    camera = CameraStream()
    camera.start()
    print("📷 Detection started (Press 'q' to quit)")

    try:
        frame_count = 0
        while True:
            frame = camera.read()
            if frame is None:
                continue

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR if frame.shape[2] == 4 else cv2.COLOR_RGB2BGR)

            start = time.time()
            predictions = detect(frame)
            elapsed = time.time() - start

            frame_count += 1
            print(f"\n📸 Frame {frame_count} | {len(predictions)} detections | {elapsed:.2f}s")

            detected_stop = False
            new_speed = None

            for i, (label, score) in enumerate(predictions, 1):
                print(f"  {i}. {label} - Confidence: {score:.2f}")

                if label == "Stop" and score >= 0.8:
                    detected_stop = True
                if label == "Red Light" and score >= 0.8:
                    detected_stop = True
                elif score >= 0.8:
                    if label == "Speed Limit 20":
                        new_speed = 20
                    if label == "Green Light":
                        new_speed = 20
                    elif label == "Speed Limit 60":
                        new_speed = 60
                    elif label == "Speed Limit 90":
                        new_speed = 90

            # STOP sign action
            if detected_stop and motors_running:
                print("🛑 Stop sign detected! Stopping motors and turning on LED.")
                stop_motors()
                led_on()
                motors_running = False

            elif not detected_stop and not motors_running:
                print("✅ Stop sign cleared. Resuming motors and turning off LED.")
                move_forward(current_speed)
                led_off()
                motors_running = True

            # SPEED LIMIT control
            elif new_speed is not None and motors_running:
                if new_speed != current_speed:
                    print(f"⚙️ Speed sign detected. Adjusting speed to {new_speed}%")
                    move_forward(new_speed)
                    current_speed = new_speed

            # Show frame
            cv2.imshow("YOLOv8 Detection", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")

    finally:
        print("🧹 Cleaning up...")
        camera.stop()
        stop_motors()
        led_off()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
