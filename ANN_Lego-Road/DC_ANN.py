import cv2
import random
import numpy as np
from keras.models import load_model


First_NN = load_model("Lego-Road_DC_First.h5")
Second_NN = load_model("Lego-Road_DC_Second.h5")

#=======================================================================================================================
F_Forward_Tensor, F_Left_Right_Tensor = [], []
F_Forward_Label, F_Left_Right_Label = [], []

def Gray_Binary(file_path, shape, thresh):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_resize, img_binary

for f in range(1, 321):
    filename = "../File_Lego-Road/Forward/" + str(f) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    F_Forward_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
    #Forward_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    F_Forward_Label.append([0, 1])

for l in range(1, 161):
    filename = "../File_Lego-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    F_Left_Right_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
    #Left_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    F_Left_Right_Label.append([1, 0])

    if l <= 120:
        filename = "../File_Lego-Road/Right/" + str(l) + ".jpg"
        img_bw = Gray_Binary(filename, (100, 100), 170)[0]
        F_Left_Right_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
        # Left_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
        F_Left_Right_Label.append([1, 0])

Tensor, Label = [], []

for f in range(len(F_Forward_Tensor)):
    Tensor.append(F_Forward_Tensor[f])
    Label.append(F_Forward_Label[f])
for lr in range(len(F_Left_Right_Tensor)):
    Tensor.append(F_Left_Right_Tensor[lr])
    Label.append(F_Left_Right_Label[lr])

def Shuffle(T, L, times):
    Element = len(T)
    index = list(range(Element))
    for x in range(times):
        switch = (random.choice(index), random.choice(index))
        first, second = switch[0], switch[1]
        T[first], T[second] = T[second], T[first]
        L[first], L[second] = L[second], L[first]
    return T, L

X = Shuffle(Tensor, Label, 1000)
Tensor, Label = X[0], X[1]

Tensor, Label = np.array(Tensor), np.array(Label)

#=======================================================================================================================

a = First_NN.evaluate(Tensor[0:1], np.array([1, 0]))
print(a)

