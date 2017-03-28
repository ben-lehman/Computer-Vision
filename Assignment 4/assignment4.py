from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from array import array

from scipy.ndimage import filters
from skimage.feature import peak_local_max
from skimage import feature, io


# Load Image
def loadImage(im_file_name):
    im = Image.open( im_file_name ).convert("L")
    data = np.array( im)
    return data

im = loadImage('line_original.jpg')

print im

# Compute Canny Edge Detection

def edges(img, sig):
    out = np.uint8(feature.canny(img, sigma=sig) * 255)
    return out

edges1 = edges(im, 1)
edges2 = edges(im, 3)

# Display results
# plt.figure(figsize=(8,3))

# plt.subplot(131)
# plt.imshow(im, cmap=plt.cm.jet)
# plt.axis('off')
# plt.title('noisy image', fontsize=20)

# plt.subplot(132)
# plt.imshow(edges1, cmap=plt.cm.gray)
# plt.axis('off')
# plt.title('Canny Filter', fontsize=20)

# plt.subplot(133)
# plt.imshow(edges2, cmap=plt.cm.gray)
# plt.axis('off')
# plt.title('Canny Filter, sig3', fontsize=20)


# plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
#                     bottom=0.02, left=0.02, right=0.98)

# plt.show()

# -------- Step 2 & 3--------- #

def houghTransform(im):
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = im.shape
    d_len = np.ceil(np.sqrt(width * width + height * height))
    rhos = np.linspace(-d_len, d_len, d_len * 2.0)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    H = np.zeros((2 * d_len, num_thetas/3), dtype=np.uint64)
    y_idx, x_idx = np.nonzero(im)
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        for t_idx in range(0, num_thetas, 3):
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + d_len
            H[rho, t_idx/3] += 1
    return H, thetas, rhos



def prepareImage(arr):
    """ Returns array ready to be converted to Image """
    arr_max = float(np.amax(arr))
    arr_min = float(np.amin(arr))
    return np.uint8(((arr - arr_min) / (arr_max - arr_min)) * 255)

# H, thetas, rhos = houghTransform(edges2)

# print H.shape, thetas.shape, rhos.shape

# print np.unravel_index(np.argmax(H), H.shape)
# print H[280][0]


# i = Image.fromarray(prepareImage(H))
# i.save("line_voting.jpg", "JPEG")
# i.show()

# --------- Step 4 ----------- #

# def prepareCoords(coords, H):
#     new_c = []
#     for c in coords:
#         rho = rhos[c/ H.shape[1]]
#         theta = thetas[c % H.shape[1]]
#         print "rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta))
#     return new_c

# coordinates = prepareCoords(peak_local_max(H, min_distance = 20), H)

# print coordinates

# fig, ax = plt.subplots(1, 2, figsize=(10,10))

# ax[0].imshow(im, cmap=plt.cm.gray)
# ax[0].set_title('Input Image')
# ax[0].axis('image')

# ax[1].imshow(H, cmap='jet', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
# ax[1].set_aspect('equal', adjustable='box')
# ax[1].set_title('Hough Transform')
# ax[1].axis('image')

# plt.show()

# fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey= True, subplot_kw={'adjustable': 'box-forced'})

# ax = axes.ravel()
# ax[0].imshow(im, cmap=plt.cm.gray)
# ax[0].axis('off')
# ax[0].set_title('Original')

# ax[1].imshow(im, cmap=plt.cm.gray)
# ax[1].autoscale(False)
# ax[1].plot(280, 0, 'r')
# ax[1].axis('off')
# ax[1].set_title('Peak Local Max')

# ax[2].imshow(im, cmap=plt.cm.gray)
# ax[2].axis('off')
# ax[2].set_title('Original')

# plt.show()

# -------- Step 5--------- #

imx = np.zeros(im.shape)
filters.sobel(edges2, 1, imx)

imy = np.zeros(im.shape)
filters.sobel(edges2, 0, imy)

magnitude = np.sqrt(imx**2+imy**2)
thet = np.arctan(imy/imx)


pil_im = Image.fromarray(np.uint8(thet))
# pil_im.save("test.jpg", 'JPEG')
pil_im.show()

def houghTransform(im, direction):
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = im.shape
    d_len = np.ceil(np.sqrt(width * width + height * height))
    rhos = np.linspace(-d_len, d_len, d_len * 2.0)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    H = np.zeros((2 * d_len, num_thetas), dtype=np.uint64)
    y_idx, x_idx = np.nonzero(im)
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        theta = direction[x_idx][y_idx]
        rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + d_len
        H[rho, t_idx/3] += 1
    return H, thetas, rhos
