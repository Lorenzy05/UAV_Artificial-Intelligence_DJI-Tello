import cv2
from djitellopy import Tello
import time

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()


time.sleep(5)
file = "File_Paper-Road/test.jpg"
cv2.imwrite(file, frame_read.frame)

tello.land()
