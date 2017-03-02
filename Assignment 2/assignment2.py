from PIL import Image, ImageDraw
from pylab import *
from scipy.ndimage import *
from scipy import signal, ndimage
# import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

# ----------- 2.1 ----------- #


def prepareImage(arr):
    """ Returns array ready to be converted to Image """
    arr_max = amax(arr)
    arr_min = amin(arr)
    return uint8(((arr - arr_min) / (arr_max - arr_min)) * 255)


def gkern(kernlen=51, nsig=2):
    """ Returns a 2d Gaussian filter"""

    inp = np.zeros((kernlen, kernlen))
    inp[kernlen // 2, kernlen // 2] = 1
    return filters.gaussian_filter(inp, nsig)


def con(arr_1, arr_2):
    """ Returns convolved array """

    new_arr = ndimage.convolve(arr_1, arr_2, mode='constant', cval=0.0)
    return new_arr

# Guasian Filters

gfilter2 = gkern(51, 2)
gfilter4 = gkern(51, 4)
gfilter8 = gkern(51, 8)

# FilterBank

filter1 = con(gfilter2, [[1, -1]])
filter2 = con(gfilter4, [[1, -1]])
filter3 = con(gfilter8, [[1, -1]])
filter4 = con(gfilter2, [[1], [-1]])
filter5 = con(gfilter4, [[1], [-1]])
filter6 = con(gfilter8, [[1], [-1]])
filter7 = gfilter4 - gfilter2
filter8 = gfilter8 - gfilter4


#Convert to Image, save, and show

# pil_im = Image.fromarray(prepareImage(filter8))
# pil_im.save('filter8.jpg', 'JPEG')
# pil_im.show()

# ----------- 2.2 ----------- #

zebra_im = array(Image.open('zebra.jpg').convert("L"))

za1 = con(zebra_im, filter1)
za2 = con(zebra_im, filter2)
za3 = con(zebra_im, filter3)
za4 = con(zebra_im, filter4)
za5 = con(zebra_im, filter5)
za6 = con(zebra_im, filter6)
za7 = con(zebra_im, filter7)
za8 = con(zebra_im, filter8)

f_bank = [za1, za2, za3, za4, za5, za6, za7, za8]
pil_im = Image.fromarray(prepareImage(za2))
# pil_im.save('zebra_activation8.jpg', 'JPEG')
# pil_im.show()

# ----------- 2.3 ----------- #

def filterValues(i, j, filter_bank):
    new_arr = zeros(len(filter_bank))
    for x in range(len(filter_bank)):
        new_arr[x] = filter_bank[x][i][j]
    return new_arr

def zebraInitialize(arr, filter_bank):
    new_arr = zeros(arr.shape, dtype=object)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = filterValues(i, j, filter_bank)
            new_arr[i][j] = val
            # print new_arr[i][j]
    return new_arr

def lackOfName(arr):
    center = arr[len(arr) // 2, len(arr[0]) // 2]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            cur = arr[i][j]
            dist = np.linalg.norm(cur - center)
            arr[i][j] = dist
    return arr

texture_array = lackOfName(zebraInitialize(zebra_im, f_bank))
pil_im = Image.fromarray(prepareImage(texture_array))
# pil_im.save('zebra_texture_comparison.jpg', 'JPEG')
pil_im.show()
