from Data import *
from keras import layers
from keras import models

CNN = models.Sequential()

CNN.add(layers.Conv2D(64, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(128, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(128, (3, 3), activation='sigmoid'))

CNN.add(layers.Flatten())
CNN.add(layers.Dense(256, activation='relu'))
CNN.add(layers.Dense(3, activation='softmax'))

CNN.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

Accuracy_Loss = CNN.fit(Tensor[:80], Label[:80],
                        epochs=20, batch_size=5,
                        validation_data=(Tensor[80:100], Label[80:100]))

print()
CNN.evaluate(Tensor[100:], Label[100:])

L = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
T, Totale = 0, 0
for x in range(100, 126):
    Totale = Totale + 1
    result = CNN.predict([Tensor[x].tolist()])
    index = np.argmax((result))
    #print(L[index], Label[x].tolist())
    if L[index] == Label[x].tolist():
        T = T + 1
print("Procent : ", int(T / Totale * 100))



print()
Choice = input("Option : ")

if str(Choice) == 'save':
    CNN.save("CNN-Paper Road.h5")
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
