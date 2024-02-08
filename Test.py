import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio


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



def Gray_Binary(file_path, shape, thresh):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_resize, img_binary

plt.ion()  # 启用交互模式

'''
frames_Forward = []
for f in range(1, 320):
    plt.clf()
    file = "File_Lego-Road/Forward/" + str(f) + ".jpg"
    binary = Edge_Detection(file, (100, 100))
    array = np.array(binary)
    binary = array[::-1]
    plt.imshow(binary, cmap='gray')
    plt.axis('off')
    plt.title(("Forward " + str(f)))
    plt.pause(0.1)
    fig = plt.gcf()
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames_Forward.append(frame)
plt.close()
#imageio.mimsave('File_Lego-Road/Move_Forward.gif', frames_Forward, duration=1)
'''

frames_Right = []
plt.figure()
for r in range(1, 128):
    plt.clf()
    file = "File_Lego-Road/Right/" + str(r) + ".jpg"
    binary = Edge_Detection(file, (100, 100))
    array = np.array(binary)
    binary = array[::-1]
    plt.imshow(binary, cmap='gray')
    plt.axis('off')
    plt.title(("Right " + str(r)))
    plt.pause(0.1)
    fig = plt.gcf()
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames_Right.append(frame)
plt.close()
#imageio.mimsave('File_Lego-Road/Turn_Right.gif', frames_Right, duration=1)

frames_Left = []
plt.figure()
for l in range(1, 163):
    plt.clf()
    file = "File_Lego-Road/Left/" + str(l) + ".jpg"
    binary = Edge_Detection(file, (100, 100))
    array = np.array(binary)
    binary = array[::-1]
    plt.imshow(binary, cmap='gray')
    plt.axis('off')
    plt.title(("Left " + str(l)))
    plt.pause(0.1)
    fig = plt.gcf()
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames_Left.append(frame)
#imageio.mimsave('File_Lego-Road/Turn_Left.gif', frames_Left, duration=1)


