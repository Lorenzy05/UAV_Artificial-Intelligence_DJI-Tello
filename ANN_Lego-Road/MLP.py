import cv2
import random
import numpy as np

from keras import layers
from keras import models


#=======================================================================================================================
Forward_Tensor, Left_Tensor, Right_Tensor = [], [], []
Forward_Label,  Left_Label,  Right_Label = [], [], []

def Gray_Binary(file_path, shape, thresh):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_resize, img_binary


for f in range(1, 321):
    filename = "../File_Lego-Road/Forward/" + str(f) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[1]
    Forward_Tensor.append(np.array(img_bw).flatten() / 255)
    #Forward_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    Forward_Label.append([1, 0, 0])

for l in range(1, 161):
    filename = "../File_Lego-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[1]
    Left_Tensor.append(np.array(img_bw).flatten() / 255)
    #Left_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    Left_Label.append([0, 1, 0])

for r in range(1, 121):
    filename = "../File_Lego-Road/Right/" + str(r) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[1]
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

X = Shuffle(Tensor, Label, 1000)
Tensor, Label = X[0], X[1]

Tensor, Label = np.array(Tensor), np.array(Label)

#=======================================================================================================================


MLP = models.Sequential()

MLP.add(layers.Dense(32, activation='relu', input_shape=(100*100,)))
MLP.add(layers.Dense(64, activation='relu'))
MLP.add(layers.Dense(128, activation='relu'))
MLP.add(layers.Dense(3, activation='softmax'))

MLP.compile(optimizer = 'sgd',
            loss      = 'categorical_crossentropy',
            metrics   = ['accuracy'])

Accuracy_Loss = MLP.fit(Tensor[0:400], Label[0:400],
                        epochs=50, batch_size=10,
                        validation_data=(Tensor[400:450], Label[400:450]))

import time
print()
print(len(Tensor))
begin = time.time()
T, L = Tensor[250:251], Label[250:251]
MLP.evaluate(Tensor[450:], Label[450:])
eind = time.time()
print((eind - begin))
print()

Choice = input("Option : ")

if str(Choice) == 'save':
    MLP.save("Lego-Road_MLP.h5")
    print("----- SAVED")
else:
    print("No saving")


import numpy as np
import matplotlib.pyplot as plt

A_L = Accuracy_Loss.history
acc, val_acc = A_L['accuracy'], A_L['val_accuracy']
loss, val_loss = A_L['loss'], A_L['val_loss']

epoches = range(1, 21)
Acc_reg = np.poly1d(np.polyfit(epoches, acc, 3))
Acc_val_reg = np.poly1d(np.polyfit(epoches, val_acc, 3))
Loss_reg = np.poly1d(np.polyfit(epoches, loss, 3))
Loss_val_reg = np.poly1d(np.polyfit(epoches, val_loss, 3))

plt.subplot(1, 2, 1)
plt.title("Training-Accuracy and Validation-Accuracy")
plt.scatter(epoches, acc, label='Training Accuracy', c='green')
plt.scatter(epoches, val_acc, label='Validaion Accuracy', c='blue', marker='+')
plt.plot(epoches, Acc_reg(epoches), label='Accuracy Polynomial Regression', c='green')
plt.plot(epoches, Acc_val_reg(epoches), label='Validation Accuracy Polynomial Regression', c='blue')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Training-loss and Validation-loss")
plt.scatter(epoches, loss, label='Training loss', c='yellow')
plt.scatter(epoches, val_loss, label='Validaion loss', c='red', marker='x')
plt.plot(epoches, Loss_reg(epoches), label='Loss Polynimial Regression', c='yellow')
plt.plot(epoches, Loss_val_reg(epoches), label='Validation Loss Polynomial Regression', c='red')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()

plt.show()