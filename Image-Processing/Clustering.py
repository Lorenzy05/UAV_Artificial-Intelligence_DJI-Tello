import cv2
import numpy as np
import matplotlib.pyplot as plt

def ncut_graph_matrix(img, sigma_d = 1e2, sigma_g = 1e-2):
    # Maak een matrix voor de genormaliseerde snede, waarbij sigma_d en sigma_g de gewichtsparameters voor pixelafstand en pixelovereenkomst zijn.
    m, n = img.shape[:2]
    N = m * n

    # Normaliseren en creëren van RGB of grijswaarden kenmerkvectoren
    if len(img.shape) == 3:
        for i in range(3):
            img[:, :, i] = img[:, :, i] / img[:, :, i].max()
        vim = img.reshape((-1, 3))
    else:
        im = img / img.max()
        vim = im.flatten()

    # x,y coördinaten voor afstandsberekening
    xx, yy = np.meshgrid(range(n), range(m))
    x, y = xx.flatten(), yy.flatten()

    # Maak randgewicht matrix
    W = np.zeros((N, N), dtype='float32')
    for i in range(N):
        for j in range(i, N):
            d = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            W[i, j] = W[j, i] = np.exp(-1.0 * np.sum((vim[i] - vim[j]) ** 2) / sigma_g) * np.exp(-d / sigma_d)
    return W


from scipy.cluster.vq import *
def cluster(S,k,ndim):
    #  Spectrale clustering van similariteitsmatrices "
    # De symmetrie controleren
    if np.sum(np.abs(S - S.T)) > 1e-10:
        print('not symmetric')

    # Creëer Laplacian matrix
    rowsum = np.sum(np.abs(S), axis=0)
    D = np.diag(1 / np.sqrt(rowsum + 1e-6))
    L = np.dot(D, np.dot(S, D))

    # Bereken de eigenvectoren van L
    U, sigma, V = np.linalg.svd(L)

    # Maak eigenvectoren van de eerste ndim eigenvectoren
    # Gestapelde eigenvectoren als kolommen van een matrix
    features = np.array(V[:ndim]).T

    # K-means clustering
    features = whiten(features)
    centroids, distortion = kmeans(features, k)
    code, distance = vq(features, centroids)

    return code, V


img_org = cv2.imread("../Paper Road/File_Paper-Road/Test_Image_Paper/Figuur_331.jpg")
img_resize = cv2.resize(img_org, (50, 50)) # Vorm veranderen naar 28 * 28
img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)  # vorm van BGR --> RGB

img_array = np.array(img) # Waarden van pixels
m,n = img_array.shape[:2]
wid = 50 # Vorm veranderen

from PIL import Image
rim = np.array(Image.fromarray(img_array).resize((wid,wid),resample=Image.BILINEAR),'f')

A = ncut_graph_matrix(rim,sigma_d=1,sigma_g=1e-2)

code,V = cluster(A,k=2,ndim=3)

img_clustering = np.array(Image.fromarray(code.reshape(wid,wid)).resize((n,m),resample=Image.NEAREST))

plt.subplot(1, 2, 1)
plt.title("Origineel image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_clustering, cmap='gray')
plt.title("Segmentation using clustering")
plt.axis('off')

plt.show()


