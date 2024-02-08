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

for f in range(1, 321):
    filename = "../File_Lego-Road/Forward/" + str(f) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    Forward_Tensor.append(img_bw.flatten() / 255)
    Forward_Label.append(0)

for l in range(1, 161):
    filename = "../File_Lego-Road/Left/" + str(l) + ".jpg"
    img_bw = Gray_Binary(filename, (100, 100), 170)[0]
    Left_Tensor.append(img_bw.flatten() / 255)
    Left_Label.append(1)

for r in range(1, 121):
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


import time
from sklearn import svm
from sklearn.metrics import accuracy_score

'''
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



'''
clf_rbf = svm.SVC(kernel='rbf', C=1)
clf_rbf.fit(Tensor[:400], Label[:400])

begin = time.time()
y_pred_rbf = clf_rbf.predict(Tensor[400:])
end = time.time()

accuracy_rbf = accuracy_score(Label[400:], y_pred_rbf)

print("Accuracy : " + str(accuracy_rbf) + " %")
print("Time     : " + str(end - begin) + " s")
print("Average  : " + str((end - begin) / 200))


import joblib
choice = input("Save ? : ")
if str(choice) == "save":
    joblib.dump(clf, 'svm_model.pkl')
    print("Saving")
if str(choice) != "save":
    print("No saving")
'''

import joblib

loaded_svm_model = joblib.load('svm_model.pkl')

b = time.time()
predictions = loaded_svm_model.predict(Tensor[501].reshape(1, -1))
e = time.time()

print(predictions)

print(predictions[0], Label[501], (e - b))





'''
import numpy as np

# 准备数据
# X_train：训练数据特征
# y_train：训练数据标签，标签值应为0、1、2表示三个类别
X_train = Tensor[:400]  # 替换为您的训练数据
y_train = Label[:400]  # 替换为您的训练标签

# 初始化参数
num_samples, num_features = X_train.shape
num_classes = 3  # 三分类问题

# SVM参数
C = 1.0  # 正则化参数
tolerance = 0.001  # 收敛容差
max_iterations = 1000  # 最大迭代次数

# 初始化alpha，b和E
alpha = np.zeros((num_samples, num_classes))
b = np.zeros(num_classes)
E = np.zeros((num_samples, num_classes))

# 训练SVM
for class_label in range(num_classes):
    for i in range(num_samples):
        E[i, class_label] = b[class_label]
        for j in range(num_samples):
            E[i, class_label] += alpha[j, class_label] * y_train[j] * np.dot(X_train[i], X_train[j])
        E[i, class_label] -= y_train[i]

# SMO算法主循环
num_iterations = 0
while num_iterations < max_iterations:
    num_changed_alphas = 0
    for i in range(num_samples):
        for class_label in range(num_classes):
            if (y_train[i] * E[i, class_label] < -tolerance and alpha[i, class_label] < C) or \
                    (y_train[i] * E[i, class_label] > tolerance and alpha[i, class_label] > 0):
                j = np.random.choice([j for j in range(num_samples) if j != i])
                delta_alpha = (y_train[i] * (E[i, class_label] - E[j, class_label])) / (
                            np.dot(X_train[i], X_train[i]) + np.dot(X_train[j], X_train[j]) -
                            2 * np.dot(X_train[i], X_train[j]))
                alpha_i_old = alpha[i, class_label]
                alpha_j_old = alpha[j, class_label]
                alpha[i, class_label] = np.clip(alpha_i_old + delta_alpha, 0, C)
                alpha[j, class_label] = alpha_j_old + y_train[i] * y_train[j] * (alpha_i_old - alpha[i, class_label])
                b[class_label] = b[class_label] + y_train[i] * (alpha[i, class_label] - alpha_i_old) * np.dot(
                    X_train[i], X_train[j])
                b[class_label] = b[class_label] + y_train[j] * (alpha[j, class_label] - alpha_j_old) * np.dot(
                    X_train[i], X_train[j])
                num_changed_alphas += 1

    if num_changed_alphas == 0:
        num_iterations += 1
    else:
        num_iterations = 0

# 训练完毕，现在您可以使用模型进行预测

# 定义预测函数
def predict(X_test):
    predictions = []
    for i in range(X_test.shape[0]):
        class_scores = []
        for class_label in range(num_classes):
            score = np.dot(X_test[i], X_train.T) @ (y_train == class_label) - b[class_label]
            class_scores.append(score)
        predicted_class = np.argmax(class_scores)
        predictions.append(predicted_class)
    return predictions

# 使用训练好的模型进行预测
X_test = np.array(Tensor[400:])  # 替换为您的测试数据
predictions = predict(X_test)

# 打印预测结果
print("预测结果:", predictions)
'''

























