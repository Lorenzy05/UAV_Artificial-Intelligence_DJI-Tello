import cv2
import time
import joblib
import numpy as np
from djitellopy import Tello
import matplotlib.pyplot as plt


loaded_svm_model = joblib.load('svm_model.pkl')

def Gray_Binary(file_path, shape, thresh):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_resize, img_binary


Drone = Tello()

Drone.connect()
Drone.streamon()
frame_read = Drone.get_frame_read()

Drone.takeoff()

print(Drone.get_battery())

Drone.send_rc_control(0, 0, -180, 0)
time.sleep(0.5)
Drone.send_rc_control(0, 0, 0, 0)

Direction = ['Forward', 'Left', 'Right']

plt.ion()


while True:
    image = frame_read.frame
    cv2.imshow('', image)
    key = cv2.waitKey(1) & 0xff

    if key == 27: # ESC
        break

    img = Gray_Binary(image, (100, 100), 200)[0]

    predictions = loaded_svm_model.predict(img.reshape(1, -1))[0]

    D = Direction[predictions]

    plt.clf()
    array = np.reshape(img, (100, 100))
    img = array[::-1]
    plt.imshow(img, cmap='gray')
    plt.title(str(D))
    plt.axis('off')
    plt.pause(0.1)

    if str(D) == 'Forward':
        Drone.send_rc_control(0, 10, 0, 0)
    if str(D) == 'Left':
        Drone.send_rc_control(0, 0, 0, -10)
    if str(D) == 'Right':
        Drone.send_rc_control(0, 0, 0, 10)

Drone.land()