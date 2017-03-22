from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from array import array

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

# -------- Step 2 --------- #

def houghTransform(img):

    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width * width + height * height))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)

    print accumulator.shape
    y_idxs, x_idxs = np.nonzero(img)
    print len(y_idxs), len(x_idxs)

    for i in range(len(x_idxs)):
        print i
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos



def prepareImage(arr):
    """ Returns array ready to be converted to Image """
    arr_max = float(np.amax(arr))
    arr_min = float(np.amin(arr))
    return np.uint8(((arr - arr_min) / (arr_max - arr_min)) * 255)

H, thetas, rhos = houghTransform(edges2)

print prepareImage(H)

i = Image.fromarray(prepareImage(H))

i.show()
