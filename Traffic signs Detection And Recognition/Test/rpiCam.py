import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# Initialize GPIO for motor control and LED
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor 1 pins
ENA = 17  # PWM pin for Motor 1 speed control
IN1 = 27  # Direction control pin 1
IN2 = 22  # Direction control pin 2

# Motor 2 pins
ENB = 18  # PWM pin for Motor 2 speed control
IN3 = 23  # Direction control pin 1
IN4 = 24  # Direction control pin 2

# LED pin
LED_PIN = 2

# Set up pins as output
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)

# Create PWM instances for speed control
pwm_a = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
pwm_b = GPIO.PWM(ENB, 1000)

# Start PWM with 0% duty cycle (stopped)
pwm_a.start(0)
pwm_b.start(0)

def move_forward(speed):
    """Move both motors forward at specified speed (0-100%)"""
    # Set motor directions
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    
    # Set motor speeds
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def stop_motors():
    """Stop both motors"""
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

def led_on():
    """Turn LED on"""
    GPIO.output(LED_PIN, GPIO.HIGH)

def led_off():
    """Turn LED off"""
    GPIO.output(LED_PIN, GPIO.LOW)

# Load the model
model = load_model('../model/modele_panneaux.h5')

IMG_HEIGHT, IMG_WIDTH = 30, 30

# Class dictionary
classes = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry', 
    18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 22:'Bumpy road', 
    23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians',
    28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only',
    36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
    41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# Initialize camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("Press Ctrl+C to exit.")

# Initial state
motors_running = True
move_forward(30)  # Start with motors moving at low speed
led_off()         # Start with LED off

try:
    while True:
        frame = picam2.capture_array()
        img = Image.fromarray(frame).resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        
        if confidence < 0.8:
            label = "No prediction (low confidence)"
        else:
            label = classes[predicted_class]

        print(f"Prediction: {label} (Confidence: {confidence:.2f}) - Motors: {'ON' if motors_running else 'OFF'}")
        
        # Check if "Stop" sign is detected with high confidence
        if predicted_class == 14 and confidence >= 0.8:  # Class 14 is 'Stop'
            if motors_running:
                print("Stop sign detected! Stopping motors and turning on LED...")
                stop_motors()
                led_on()
                motors_running = False
        else:
            if not motors_running:
                print("No stop sign detected. Resuming motors and turning off LED...")
                move_forward(30)
                led_off()
                motors_running = True
            
        time.sleep(0.5)  # Small delay between detections

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    print("Cleaning up...")
    stop_motors()
    led_off()
    picam2.stop()
    GPIO.cleanup()