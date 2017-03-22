from PIL import Image
from scipy import ndimage as ndi
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

def houghTransform(img_bin, theta_res=1, rho_res=1):
    nR, nC = img_bin.shape
    theta = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1)
    theta = np.concatenate((theta, -theta[len(theta)-2::-1]))

    D = np.sqrt((nR - 1)**2 + (nC - 1)**2)
    q = np.ceil(D/rho_res)
    nrho= 2*q + 1
    rho = np.linspace(-1*rho_res, q*rho_res, nrho)
    H = np.zeros((len(rho), len(theta)))
    for rowIdx in range(nR):
        print rowIdx
        for colIdx in range(nC):
            print colIdx
            if img_bin[rowIdx, colIdx]:
                for thIdx in range(len(theta)):
                    rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) +  rowIdx*np.sin(theta[thIdx]*np.pi/180)
                    rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
                    H[rhoIdx[0], thIdx] += 1
    return rho, theta, H

def prepareImage(arr):
    """ Returns array ready to be converted to Image """
    arr_max = float(np.amax(arr))
    arr_min = float(np.amin(arr))
    return np.uint8(((arr - arr_min) / (arr_max - arr_min)) * 255)

rhos, thetas, H = houghTransform(edges2)

print prepareImage(H)

i = Image.fromarray(prepareImage(H))

i.show()
