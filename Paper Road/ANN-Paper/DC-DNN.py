import cv2
import random
import numpy as np

Forward_Tensor, Left_Tensor, Right_Tensor = [], [], []
Forward_Label,  Left_Label,  Right_Label = [], [], []
FL, LL, RL = [], [], []

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
    FL.append(0)
    Forward_Label.append([1, 0, 0])

for l in range(1, 38):
    filename = "../File_Paper-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (28, 28), 130)[1]
    Left_Tensor.append(np.array(img_bw).flatten() / 255)
    #Left_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    LL.append(1)
    Left_Label.append([0, 1, 0])

for r in range(1, 36):
    filename = "../File_Paper-Road/Right/" + str(r) + ".jpg"
    img_bw = Gray_Binary(filename, (28, 28), 130)[1]
    Right_Tensor.append(np.array(img_bw).flatten() / 255)
    #Right_Tensor.append(np.array(img_bw).reshape(28, 28, 1))
    RL.append(1)
    Right_Label.append([0, 0, 1])

Tensor, Label = [], []
First_Label = []

for f in range(len(Forward_Tensor)):
    Tensor.append(Forward_Tensor[f])
    Label.append(Forward_Label[f])
    First_Label.append(FL[f])
for l in range(len(Left_Tensor)):
    Tensor.append(Left_Tensor[l])
    Label.append(Left_Label[l])
    First_Label.append(LL[l])
for r in range(len(Right_Tensor)):
    Tensor.append(Right_Tensor[r])
    Label.append(Right_Label[r])
    First_Label.append(RL[r])

def Shuffle(T, L, F, times):
    Element = len(T)
    index = list(range(Element))
    for x in range(times):
        switch = (random.choice(index), random.choice(index))
        first, second = switch[0], switch[1]
        T[first], T[second] = T[second], T[first]
        L[first], L[second] = L[second], L[first]
        F[first], F[second] = F[second], F[first]
    return T, L, F

X = Shuffle(Tensor, Label, First_Label, 20)
Tensor, Label, First_Label = X[0], X[1], X[2]

Tensor, Label, First_Label = np.array(Tensor), np.array(Label), np.array(First_Label)

#=======================================================================================================================
from keras import models
from keras import layers

def ANN():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(28*28, )))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

DC_DNN = ANN()
Accuracy_Loss = DC_DNN.fit(Tensor[0:80], First_Label[0:80],
                        epochs=10, batch_size=80,
                        validation_data=(Tensor[80:100], First_Label[80:100]))

def Activation(x):
    if x >=0.5:
        return 1
    else:
        return 0

Lijst = [0, 1]
T, Totale = 0, 0
for x in range(100, 127):
    Totale = Totale + 1
    result = DC_DNN.predict([Tensor[x].tolist()])
    #print(L[index], Label[x].tolist())
    if Activation(result[0][0]) == First_Label[x].tolist():
        T = T + 1
    else:
        print(result, First_Label[x])
print("Procent : ", int(T / Totale * 100))

