import cv2
import numpy as np
import matplotlib.pyplot as plt


def Edge_Detection(image, shape):
    image = cv2.imread(image)
    image_resize = cv2.resize(image, shape)
    gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    edge = np.sqrt(np.square(sobelx) + np.square(sobely))
    edge = (edge / np.max(edge)) * 255
    edge = edge.astype(np.uint8) / 255

    return edge


edges_L = Edge_Detection("../Image-Processing/T.jpg", (100, 100))

plt.imshow(edges_L, cmap='gray')
plt.show()




cv2.imshow('', edges_L)
cv2.waitKey(0)
cv2.destroyAllWindows()