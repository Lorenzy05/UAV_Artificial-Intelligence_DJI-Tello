import cv2
import time
import random
import operator
import numpy as np
import matplotlib.pyplot as plt


def Gray_Binary(file_path, shape, thresh):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_resize, img_binary

Forward_Tensor, Left_Tensor, Right_Tensor = [], [], []
Forward_Label,  Left_Label,  Right_Label = [], [], []

for f in range(1, 320):
    filename = "../File_Lego-Road/Forward/" + str(f) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    Forward_Tensor.append(img_bw.flatten() / 255)
    Forward_Label.append('Forward')

for l in range(1, 160):
    filename = "../File_Lego-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    Left_Tensor.append(img_bw.flatten() / 255)
    Left_Label.append('Left')

for r in range(1, 120):
    filename = "../File_Lego-Road/Right/" + str(r) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    Right_Tensor.append(img_bw.flatten() / 255)
    Right_Label.append('Right')

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

X = Shuffle(Tensor, Label, 800)
Tensor, Label = X[0], X[1]

Tensor = np.array(Tensor)


def KNN_Classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDist = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDist[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


Dataset_Tensor  = Tensor[:400]
Dataset_Label   = Label[:400]


Percentages = []


K3 = [97.33333333333334, 97.33333333333334, 92.66666666666666, 93.33333333333333, 90.0, 90.66666666666666, 90.0, 90.0, 89.33333333333333, 88.0]

import matplotlib.pyplot as plt

plt.title("K = 3")
plt.plot(list(range(1, 11)), K3)
plt.ylabel("Procenten")
plt.xlabel("10 keer verschillende testen")
plt.axis([1, 10, 80, 100])
plt.show()




for k in range(1, 11):
   Correct = 0
   start = time.time()
   for x in range(401, 551):
      Predict = KNN_Classify(Tensor[x].tolist(), Dataset_Tensor, Dataset_Label, k)
      True_Label = Label[x]
      if str(Predict) == str(True_Label):
         Correct += 1
   end = time.time()
   print("K =", k, "-->",  Correct / 150 * 100 , "% correct ", "   ", (end-start) / 150)
   perc = Correct / 150 * 100
   Percentages.append(perc)

print(Percentages)



'''
print()
print(str(len(Tensor[0])) + " pixels (100 * 100)")
print("Tensor : " + str(Tensor[0]))
print("Label  : " + str(Label[0]))

plt.imshow(np.reshape(Tensor[0], (100, 100)), cmap='gray')
plt.title(str(Label[0]))
plt.axis('off')
plt.show()



Test = [
        np.array([97.33333333333334, 97.33333333333334, 94.66666666666667, 96.0, 94.0, 93.33333333333333, 92.66666666666666, 92.66666666666666, 90.66666666666666, 90.66666666666666]),
        np.array([98.66666666666667, 98.66666666666667, 95.33333333333334, 97.33333333333334, 92.0, 94.0, 90.66666666666666, 92.66666666666666, 89.33333333333333, 90.0]),
        np.array([97.33333333333334, 97.33333333333334, 92.66666666666666, 93.33333333333333, 90.0, 90.66666666666666, 90.0, 90.0, 89.33333333333333, 88.0]),
        np.array([98.0, 98.0, 92.0, 91.33333333333333, 88.66666666666667, 90.0, 86.66666666666667, 86.0, 84.0, 84.66666666666667]),
        np.array([96.66666666666667, 96.66666666666667, 90.0, 90.66666666666666, 88.0, 89.33333333333333, 89.33333333333333, 88.66666666666667, 87.33333333333333, 87.33333333333333]),
        np.array([98.0, 98.0, 94.66666666666667, 94.66666666666667, 91.33333333333333, 92.66666666666666, 88.0, 88.66666666666667, 86.0, 88.0]),
        np.array([97.33333333333334, 97.33333333333334, 94.0, 94.0, 88.0, 89.33333333333333, 84.66666666666667, 86.66666666666667, 84.0, 84.0]),
        np.array([96.66666666666667, 96.66666666666667, 92.0, 94.0, 88.66666666666667, 92.0, 87.33333333333333, 88.0, 84.66666666666667, 84.66666666666667]),
        np.array([98.66666666666667, 98.66666666666667, 94.66666666666667, 95.33333333333334, 93.33333333333333, 93.33333333333333, 88.0, 88.0, 85.33333333333334, 86.0]),
        np.array([95.33333333333334, 95.33333333333334, 88.0, 90.0, 88.0, 89.33333333333333, 86.66666666666667, 87.33333333333333, 84.0, 83.33333333333334])
]

Test = [
np.array([97.3, 97.3, 94.7, 96.0, 94.0, 93.3, 92.7, 92.7, 90.7, 90.7]),
np.array([98.7, 98.7, 95.3, 97.3, 92.0, 94.0, 90.7, 92.7, 89.3, 90.0]),
np.array([97.3, 97.3, 92.7, 93.3, 90.0, 90.7, 90.0, 90.0, 89.3, 88.0]),
np.array([98.0, 98.0, 92.0, 91.3, 88.7, 90.0, 86.7, 86.0, 84.0, 84.7]),
np.array([96.7, 96.7, 90.0, 90.7, 88.0, 89.3, 89.3, 88.7, 87.3, 87.3]),
np.array([98.0, 98.0, 94.7, 94.7, 91.3, 92.7, 88.0, 88.7, 86.0, 88.0]),
np.array([97.3, 97.3, 94.0, 94.0, 88.0, 89.3, 84.7, 86.7, 84.0, 84.0]),
np.array([96.7, 96.7, 92.0, 94.0, 88.7, 92.0, 87.3, 88.0, 84.7, 84.7]),
np.array([98.7, 98.7, 94.7, 95.3, 93.3, 93.3, 88.0, 88.0, 85.3, 86.0]),
np.array([95.3, 95.3, 88.0, 90.0, 88.0, 89.3, 86.7, 87.3, 84.0, 83.3])
]


k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.style.use('seaborn-pastel')

plt.subplot(1, 2, 1)
for x in range(10):
    list = []
    for y in range(10):
        list.append(Test[y][x])
    plt.plot(k, list, label=("K = " + str(k[x])))
    plt.scatter(k, list)


plt.title('Results depends on K-parameter')
plt.axis([1, 10, 80, 100])
plt.xlabel('Between 1 - 10 times Test')
plt.ylabel('Percentage')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.style.use('classic')
plt.title("Average percentage of 10 times")
plt.xlabel("K-parameter")
plt.ylabel("Percentage")

Averages = [[] for x in range(10)]

for x in range(10):
    for y in range(10):
        Averages[y].append(Test[x][y])

Lists = [0]
for z in range(10):
    average = sum(Averages[z]) / 10
    Lists.append(round(average, 1))

K = [str(x) for x in range(11)]
plt.bar(K, Lists, label='Average of every k')

for x, y, z in zip(K, Lists, Lists):
    plt.text(int(x)-0.4, y+0.1, z)




plt.grid()
plt.legend()
plt.axis([0, 11, 85, 100])


plt.show()
'''

