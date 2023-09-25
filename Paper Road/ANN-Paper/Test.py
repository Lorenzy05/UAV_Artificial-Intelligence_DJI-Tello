import cv2
import numpy as np
from djitellopy import Tello
from keras.models import load_model

MLP = load_model("MLP-Paper.h5")

def Gray_Binary(file_path, shape, thresh):
    #img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(file_path, shape, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    img_b = (np.array(img_binary).flatten() / 255)
    return img_resize, img_b

Drone = Tello()
Drone.connect()
Drone.streamon()
frame_read = Drone.get_frame_read()

print(Drone.get_battery())

Direction = ['Forward', 'Left', 'Right']
Direction_Chinese = ['前进', '左转', '右转']

while True:
    image = frame_read.frame

    img = Gray_Binary(image, (100, 100), 180)[1]
    cv2.imshow("Image", image)
    Tensor = np.array(img).tolist()
    result = MLP.predict([Tensor])
    index = np.argmax(result)
    D = Direction_Chinese[index]
    print(D)

    key = cv2.waitKey(1) & 0xff
    if key == 27: # ESC
        break

Drone.land()

