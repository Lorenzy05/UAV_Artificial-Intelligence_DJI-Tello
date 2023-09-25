import cv2
import random
import numpy as np


def Gray_Binary(file_path, shape, thresh):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_resize, img_binary

Forward_Tensor, Left_Tensor, Right_Tensor, = [], [], []
Forward_Label,  Left_Label,  Right_Label = [], [], []

for f in range(1, 320):
    filename = "../File_Lego-Road/Forward/" + str(f) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    Forward_Tensor.append(img_bw.flatten() / 255)
    Forward_Label.append(0)

for l in range(1, 160):
    filename = "../File_Lego-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    Left_Tensor.append(img_bw.flatten() / 255)
    Left_Label.append(1)

for r in range(1, 120):
    filename = "../File_Lego-Road/Right/" + str(r) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    Right_Tensor.append(img_bw.flatten() / 255)
    Right_Label.append(2)

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
Tensor, Label = np.array(X[0]), np.array(X[1])

#-----------------------------------------------------------------------------------------------------------------------



from sklearn import svm
from sklearn.metrics import accuracy_score


C_P = [0.1, 1, 3, 5, 10]
for c in range(5):
    clf_lin = svm.SVC(kernel='linear', C=C_P[c])
    clf_rbf = svm.SVC(kernel='rbf', C=C_P[c])
    clf_poly = svm.SVC(kernel='poly', C=C_P[c])

    clf_lin.fit(Tensor[:400], Label[:400])
    clf_rbf.fit(Tensor[:400], Label[:400])
    clf_poly.fit(Tensor[:400], Label[:400])

    y_pred_lin = clf_lin.predict(Tensor[400:])
    accuracy_lin = accuracy_score(Label[400:], y_pred_lin)

    y_pred_rbf = clf_rbf.predict(Tensor[400:])
    accuracy_rbf = accuracy_score(Label[400:], y_pred_rbf)

    y_pred_poly = clf_poly.predict(Tensor[400:])
    accuracy_poly = accuracy_score(Label[400:], y_pred_poly)

    print()
    print("C = " + str(C_P[c]))
    print('Linear : ' + str(accuracy_lin))
    print('Poly   : ' + str(accuracy_poly))
    print('RBF    : ' + str(accuracy_rbf))





'''
import time
begin = time.time()
accuracy = accuracy_score(Label[400:], y_pred)
end = time.time()
print("Accuracy : " + str(accuracy) + " %")
print("Time     : " + str(end - begin) + " s")
print("Average  : " + str((end - begin) / 200))

import joblib
choice = input("Save ? : ")
if str(choice) == "save":
    joblib.dump(clf, 'svm_model.pkl')
    print("Saving")
if str(choice) != "save":
    print("No saving")


import joblib

loaded_svm_model = joblib.load('svm_model.pkl')

b = time.time()
predictions = loaded_svm_model.predict(Tensor[501].reshape(1, -1))
e = time.time()

print(predictions[0], Label[501], (e - b))
'''




























