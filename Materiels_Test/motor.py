import RPi.GPIO as GPIO
import time

IN1 = 17
IN2 = 27
ENA = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

pwm = GPIO.PWM(ENA, 1000)
pwm.start(0)

try:
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    
    for speed in range(0, 101, 20):
        pwm.ChangeDutyCycle(speed)
        print(f"Speed: {speed}%")
        time.sleep(1)

finally:
    pwm.stop()
    GPIO.cleanup()
