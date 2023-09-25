import cv2
import random
import numpy as np

from keras import layers
from keras import models

#=======================================================================================================================
Left_Tensor, Right_Tensor = [], []
Left_Label,  Right_Label = [], []

def Gray_Binary(file_path, shape, thresh):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_resize, img_binary


for l in range(1, 160):
    filename = "../File_Lego-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    #Left_Tensor.append(np.array(img_bw).flatten() / 255)
    Left_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
    Left_Label.append([0, 1])

for r in range(1, 120):
    filename = "../File_Lego-Road/Right/" + str(r) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    #Right_Tensor.append(np.array(img_bw).flatten() / 255)
    Right_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
    Right_Label.append([1, 0])


Tensor, Label = [], []

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

#=======================================================================================================================

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

CNN.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


Accuracy_Loss = CNN.fit(Tensor[:150], Label[:150],
                        epochs=20, batch_size=10,
                        validation_data=(Tensor[150:200], Label[150:200]))

print()
CNN.evaluate(Tensor[200:], Label[200:])

print()

Choice = input("Option : ")

if str(Choice) == 'save':
    CNN.save("Lego-Road_DC_Second.h5")
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
