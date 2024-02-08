import cv2
import time
import numpy as np
from djitellopy import Tello
import matplotlib.pyplot as plt
from keras.models import load_model


MLP = load_model("ANN_Lego-Road/Lego-Road_MLP.h5")

def Gray_Binary(file_path, shape, thresh):
    img_resize = cv2.resize(file_path, shape, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    img_b = (np.array(img_binary).flatten() / 255)
    return img_resize, img_b

Drone = Tello()

Drone.connect()
Drone.streamon()
frame_read = Drone.get_frame_read()

Drone.takeoff()

print(Drone.get_battery())

Drone.send_rc_control(0, 0, -150, 0)
time.sleep(0.6)
Drone.send_rc_control(0, 0, 0, 0)

Direction = ['Forward', 'Left', 'Right']

plt.ion()

while True:
    image = frame_read.frame
    img_reverse = np.array(image)
    cv2.imshow('', img_reverse[::-1])
    key = cv2.waitKey(1) & 0xff

    if key == 27: # ESC
        break

    img = Gray_Binary(image, (100, 100), 150)[1]

    Tensor = np.array(img).tolist()
    result = MLP.predict([Tensor])
    index = np.argmax(result)
    D = Direction[index]

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

