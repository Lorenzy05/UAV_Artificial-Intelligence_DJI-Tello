import cv2
import random
import numpy as np

Forward_Tensor, Left_Tensor, Right_Tensor = [], [], []
Forward_Label,  Left_Label,  Right_Label = [], [], []


def Gray_Binary(file_path, shape, thresh):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_resize, img_binary

for f in range(1, 56):
    filename = "../File_Paper-Road/Forward/" + str(f) + ".jpg"
    img_bw = Gray_Binary(filename, (28, 28), 130)[1]
    Forward_Tensor.append(np.array(img_bw).flatten() / 255)
    #Forward_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    Forward_Label.append([1, 0, 0])

for l in range(1, 38):
    filename = "../File_Paper-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (28, 28), 130)[1]
    Left_Tensor.append(np.array(img_bw).flatten() / 255)
    #Left_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    Left_Label.append([0, 1, 0])

for r in range(1, 36):
    filename = "../File_Paper-Road/Right/" + str(r) + ".jpg"
    img_bw = Gray_Binary(filename, (28, 28), 130)[1]
    Right_Tensor.append(np.array(img_bw).flatten() / 255)
    #Right_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    Right_Label.append([0, 0, 1])

Tensor, Label = [], []

for f in range(len(Forward_Tensor)):
    Tensor.append(Forward_Tensor[f])
    Label.append(Forward_Label[f])
for l in range(len(Left_Tensor)):
    Tensor.append(Left_Tensor[l])
    Label.append(Left_Label[l])
for r in range(len(Right_Tensor)):
    Tensor.append(Right_Tensor[r])
    Label.append(Right_Label[r])

def Shuffle(T, L, times):
    Element = len(T)
    index = list(range(Element))
    for x in range(times):
        switch = (random.choice(index), random.choice(index))
        first, second = switch[0], switch[1]
        T[first], T[second] = T[second], T[first]
        L[first], L[second] = L[second], L[first]
    return T, L

X = Shuffle(Tensor, Label, 100)
Tensor, Label = X[0], X[1]

Tensor, Label = np.array(Tensor), np.array(Label)

