import cv2
import random
import numpy as np

from keras import layers
from keras import models

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
    img_bw = Gray_Binary(filename, (100, 100), 170)[1]
    F_Forward_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
    #Forward_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    F_Forward_Label.append([0, 1])

for l in range(1, 161):
    filename = "../File_Lego-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[1]
    F_Left_Right_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
    #Left_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    F_Left_Right_Label.append([1, 0])

    if l <= 121:
        filename = "../File_Lego-Road/Right/" + str(l) + ".jpg"
        img_bw = Gray_Binary(filename, (100, 100), 170)[1]
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


def First_ANN():
    CNN = models.Sequential()

    CNN.add(layers.Conv2D(8, (3, 3), activation='sigmoid', input_shape=(100, 100, 1)))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(16, (3, 3), activation='relu'))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(16, (3, 3), activation='relu'))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(16, (3, 3), activation='relu'))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(32, activation='relu'))
    CNN.add(layers.Dense(64, activation='relu'))
    CNN.add(layers.Dense(2, activation='softmax'))

    CNN.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])


    return CNN

First_ANN = First_ANN()

Accuracy_Loss = First_ANN.fit(Tensor[0:400], Label[0:400],
        epochs=30, batch_size=40,
        validation_data=(Tensor[400:450], Label[400:450]))

print()
First_ANN.evaluate(Tensor[450:], Label[450:])
print()

First_ANN.evaluate(Tensor[0:1], Label[0:1])


choice = input("save? : ")
if str(choice) == 'save':
    First_ANN.save("Lego-Road_DC_First.h5")

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
plt.plot(epoches, acc, label='Training Accuracy', c='green')
plt.plot(epoches, val_acc, label='Validation Accuracy', c='blue', marker='+')
plt.plot(epoches, Acc_reg(epoches), label='Accuracy Polynomial Regression', c='red')
plt.plot(epoches, Acc_val_reg(epoches), label='Validation Accuracy Polynomial Regression', c='black')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Training-loss and Validation-loss")
plt.plot(epoches, loss, label='Training loss', c='yellow')
plt.plot(epoches, val_loss, label='Validation loss', c='red', marker='x')
plt.plot(epoches, Loss_reg(epoches), label='Loss Polynomial Regression', c='orange')
plt.plot(epoches, Loss_val_reg(epoches), label='Validation Loss Polynomial Regression', c='pink')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()