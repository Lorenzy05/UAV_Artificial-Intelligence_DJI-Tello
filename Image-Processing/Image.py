import cv2
import numpy as np
import matplotlib.pyplot as plt


filename = "D:\MS\Coderen_Edge.png"

def Gray_Binary(file_path, shape, thresh):
    img_org = cv2.imread(file_path)
    img_org = cv2.resize(img_org, shape, interpolation=cv2.INTER_AREA)
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_org, img_resize, img_binary

org, resize, binary = Gray_Binary(filename, (1000, 600), 40)

array = np.array(binary).tolist()

print(len(array))
print(len(array[0]))

pixels = []

for x in range(600):
    list = []
    for y in range(1000):
        if array[x][y] == 0:
            list.append(255)
        if array[x][y] == 255:
            list.append(0)
    pixels.append(list)


plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(pixels, cmap='gray')

plt.show()



