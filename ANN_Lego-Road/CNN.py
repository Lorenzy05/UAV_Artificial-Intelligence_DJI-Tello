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
    #Forward_Tensor.append(np.array(img_bw).flatten() / 255)
    Forward_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
    Forward_Label.append([1, 0, 0])

for l in range(1, 161):
    filename = "../File_Lego-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[1]
    #Left_Tensor.append(np.array(img_bw).flatten() / 255)
    Left_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
    Left_Label.append([0, 1, 0])

for r in range(1, 121):
    filename = "../File_Lego-Road/Right/" + str(r) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[1]
    #Right_Tensor.append(np.array(img_bw).flatten() / 255)
    Right_Tensor.append(np.array(img_bw).reshape(100, 100, 1))
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


CNN = models.Sequential()

CNN.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(100, 100, 1)))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(16, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(16, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(16, (3, 3), activation='relu'))

CNN.add(layers.Flatten())
CNN.add(layers.Dense(32, activation='relu'))
CNN.add(layers.Dense(64, activation='relu'))
CNN.add(layers.Dense(3, activation='softmax'))

CNN.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

Accuracy_Loss = CNN.fit(Tensor[:400], Label[:400],
                        epochs=20, batch_size=10,
                        validation_data=(Tensor[400:500], Label[400:500]))

print()
import time
begin = time.time()
CNN.evaluate(Tensor[450:], Label[450:])
eind = time.time()
print((eind - begin))
print()

Choice = input("Option : ")

if str(Choice) == 'save':
    CNN.save("Lego-Road_CNN.h5")
    print("----- SAVED")
else:
    print("No saving")
    from keras.models import Model
    import matplotlib.pyplot as plt

    # 创建一个新的模型，只保留卷积层和池化层
    layer_outputs = [layer.output for layer in CNN.layers if 'conv2d' in layer.name or 'max_pooling2d' in layer.name]
    activation_model = Model(inputs=CNN.input, outputs=layer_outputs)

    # 获取特定输入的特征图
    specific_input = np.expand_dims(Tensor[0], axis=0)  # 将第一个输入张量扩展为(batch_size, height, width, channels)
    activations = activation_model.predict(specific_input)

    # 显示特定层的特征图
    for layer_activation in activations:
        if len(layer_activation.shape) > 2:
            plt.figure(figsize=(8, 8))
            for i in range(layer_activation.shape[-1]):
                plt.subplot(8, 8, i + 1)
                plt.imshow(layer_activation[0, :, :, i], cmap='viridis')
                plt.axis('off')
            plt.show()




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