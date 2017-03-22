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

def houghTransform(im):
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
        print i
        x = x_idx[i]
        y = y_idx[i]
        for t_idx in range(num_thetas):
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + d_len
            H[rho, t_idx] += 1
    return H, thetas, rhos



def prepareImage(arr):
    """ Returns array ready to be converted to Image """
    arr_max = float(np.amax(arr))
    arr_min = float(np.amin(arr))
    return np.uint8(((arr - arr_min) / (arr_max - arr_min)) * 255)

H, thetas, rhos = houghTransform(edges2)

i = Image.fromarray(prepareImage(H))
i.save("line_hough.jpg", "JPEG")
i.show()
