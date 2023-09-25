import cv2

def Gray_Binary(file_path, shape, thresh):
    img_org = cv2.imread(file_path)
    img_org = cv2.resize(img_org, shape, interpolation=cv2.INTER_AREA)
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_org, img_resize, img_binary


import matplotlib.pyplot as plt

file_path_F = "../File_Lego-Road/Test-Image/Forward.jpeg"
img_org_F, gray_img_F, binary_img_F = Gray_Binary(file_path_F, (100, 100), 200)

file_path_L = "../File_Lego-Road/Test-Image/Left.jpeg"
img_org_L, gray_img_L, binary_img_L = Gray_Binary(file_path_L, (100, 100), 200)

plt.subplot(2, 3, 1)
plt.title("Org_Forward")
plt.imshow(img_org_F)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Grayscale_Forward")
plt.imshow(gray_img_F, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Binary_Forward")
plt.imshow(binary_img_F, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Org_Left")
plt.imshow(img_org_L)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Grayscale_Left")
plt.imshow(gray_img_L, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Binary_Left")
plt.imshow(binary_img_L, cmap='gray')
plt.axis('off')

plt.show()
