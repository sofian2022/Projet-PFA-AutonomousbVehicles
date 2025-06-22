import RPi.GPIO as GPIO
import time

SERVO_PIN = 12
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

def set_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

try:
    for angle in [0, 90, 180]:
        print(f"Rotating to {angle}°")
        set_angle(angle)
        time.sleep(1)

finally:
    pwm.stop()
    GPIO.cleanup()
