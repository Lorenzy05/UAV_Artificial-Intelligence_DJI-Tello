import cv2
import time
import numpy as np
from djitellopy import Tello
import matplotlib.pyplot as plt
from keras.models import load_model



def Edge_Detection(image, shape):
    image_resize = cv2.resize(image, shape)
    gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    edge = np.sqrt(np.square(sobelx) + np.square(sobely))
    edge = (edge / np.max(edge)) * 255
    edge = edge.astype(np.uint8)

    return edge


CNN = load_model("ANN_Lego-Road/Lego-Road_CNN_Edge-Detection.h5")

Drone = Tello()

Drone.connect()
Drone.streamon()
frame_read = Drone.get_frame_read()

Drone.takeoff()

print(Drone.get_battery())

Drone.send_rc_control(0, 0, -140, 0)
time.sleep(0.5)
Drone.send_rc_control(0, 0, 0, 0)

Direction = ['Forward', 'Left', 'Right']

plt.ion()

while True:
    image = frame_read.frame

    key = cv2.waitKey(1) & 0xff

    if key == 27: # ESC
        break

    if key == ord('s'):
        Drone.send_rc_control(0, 0, -10, 0)
        time.sleep(0.3)
        Drone.send_rc_control(0, 0, 0, 0)

    if key == ord('w'):
        Drone.send_rc_control(0, 0, 10, 0)
        time.sleep(0.3)
        Drone.send_rc_control(0, 0, 0, 0)

    if key == ord('a'):
        Drone.send_rc_control(0, 0, 0, -10)
        time.sleep(0.3)
        Drone.send_rc_control(0, 0, 0, 0)

    if key == ord('d'):
        Drone.send_rc_control(0, 0, 0, 10)
        time.sleep(0.3)
        Drone.send_rc_control(0, 0, 0, 0)

    img = Edge_Detection(image, (100, 100))
    img = np.array(img).reshape(1, 100, 100, 1)
    # 进行预测
    predictions = CNN.predict(img)
    # 获取最有可能的类别
    Label = Direction[np.argmax(predictions)]

    plt.clf()
    array = np.reshape(img, (100, 100))
    img = array[::-1]
    plt.imshow(img, cmap='gray')
    plt.title(str(Label))
    plt.axis('off')
    plt.pause(0.1)

    if str(Label) == 'Forward':
        Drone.send_rc_control(0, 10, 0, 0)
    if str(Label) == 'Left':
        Drone.send_rc_control(0, 0, 0, -10)
    if str(Label) == 'Right':
        Drone.send_rc_control(0, 0, 0, 10)

    cv2.imshow('', np.array(image)[::-1])

Drone.land()




