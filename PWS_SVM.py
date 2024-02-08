import cv2
import time
import joblib
import numpy as np
from djitellopy import Tello
import matplotlib.pyplot as plt


loaded_svm_model = joblib.load('Machine-Learning/svm_model.pkl')

def Gray_Binary(file_path, shape, thresh):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(gray_image, shape, interpolation=cv2.INTER_AREA)
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
    cv2.imshow('', np.array(image)[::-1])
    key = cv2.waitKey(1) & 0xff

    if key == 27: # ESC
        break

    img = Gray_Binary(image, (100, 100), 180)
    img_Gray = img.flatten() / 255

    predictions = loaded_svm_model.predict(img_Gray.reshape(1, -1))[0]

    D = Direction[predictions]
    print(D)

    plt.clf()
    array = np.reshape(img[1], (100, 100))
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