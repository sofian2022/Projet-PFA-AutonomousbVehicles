from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.start()
time.sleep(2)  # Let the camera warm up
picam2.capture_file("test_image.jpg")
print("Image captured successfully.")
