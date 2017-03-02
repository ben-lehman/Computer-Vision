from PIL import Image
from pylab import *
from scipy.ndimage import filters
import numpy as np
import time

# # ------- 0-2 --------- #
# pil_im = Image.open('empire.jpg').crop([100,100,400, 400]).resize((128, 128)).rotate(45)

# # pil_im.show()

# # ------- 0-3 --------- #

# im = array(Image.open('empire.jpg'))

# # imshow(im)

# x = [100, 100, 400, 400]
# y = [200, 500, 200, 500]

# plot(x, y, 'r*')

# plot(x[:2], y[:2])

# title('Plotting: "empire.jpg:"')
# # show()

# im = Image.open('empire.jpg').convert('L')
# # im.show()

# # ------- 0-4 --------- #

# pil_im = Image.open('empire.jpg').convert("L")

# # ------- 0-5 --------- #

# im = array(Image.open('empire.jpg'))
# # print im.shape, im.dtype
# row = im.shape[0]
# column = im.shape[1]
# new_im = np.zeros((800, 569))
# # print new_im.shape

# for i in range(0, row):
#     for j in range(0, column):
#         red = im[i, j][0]
#         green = im[i, j][1]
#         blue = im[i, j][2]
#         gray_value = red / 3
#         gray_value += green / 3
#         gray_value += blue / 3
#         # print 'i: ', i, " j: ", j
#         new_im[i, j] = gray_value

# pil_im = Image.fromarray(new_im)
# # pil_im.show()

# ------- 1-1 --------- #

def convolve(image, filter):
    con_filter = np.rot90(filter, 2)
    row = image.shape[0]
    column = image.shape[1]
    for x in range(1, row - 1):
        # print 'row is: ', x
        for y in range(1, column - 1):
            f_value = 0
            filter_element = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    current_value = image[x + i][y + j]
                    f_value += int(current_value * filter.item(filter_element))
                    filter_element += 1
            new_im[x][y] = f_value

# ------- 1-2 --------- #

# im = array(Image.open('empire_gray.jpg'))
# new_im = np.copy(im)

# impulse filter
# filter = np.matrix('0 0 0; 0 0 1; 0 0 0')

# box filter
# filter = np.matrix('1 1 1; 1 1 1; 1 1 1') * 1/9.0

# Gaussian filter
# filter = np.matrix('1 2 1; 2 4 2; 1 2 1') * 1/16.0

# convolve(im, filter)

# ------- 1-3 --------- #

# im = array(Image.open('empire_gray.jpg'))
# new_im = np.copy(im)
# filter = np.matrix('1 2 1') * 1/4.0

def fastConvolve(image, filter):
    row = image.shape[0]
    column = image.shape[1]
    for phase in range(1,3):
        print 'phase: ', phase
        for x in range(1, row - 1):
            for y in range(1, column - 1):
                f_value = 0
                filter_element = 0
                for i in range(-1, 1):
                    if phase is 1:
                        current_val = image[x + i][y]
                    else:
                        current_val = image[x][y + 1]
                    f_value += int(current_val * filter.item(filter_element))
                    filter_element += 1
                new_im[x][y] = f_value
        filter = filter.transpose()

# time was 16.292 seconds
# start_time = time.time()
# fastConvolve(im, filter)
# print "--- %s seconds ---" % (time.time() - start_time)

#time was 24.739 seconds
# start_time = time.time()
# convolve(im, filter)
# print "--- %s seconds ---" % (time.time() - start_time)

# pil_im = Image.fromarray(new_im)
# pil_im.show()

# ------- 1-4 --------- #

# im = array(Image.open('empire_gray.jpg'))
# im2 = filters.gaussian_filter(im,5)

# pil_im = Image.fromarray(im2)
# pil_im.show()

# ------- 1-5 --------- #
im = array(Image.open('empire.jpg').convert('L'))
new_im = np.zeros(im.shape)

#sobel x filter
# filter = np.matrix('-1 0 1; -2 0 2; -1 0 1')

#sobel y filter
# filter = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')

#SciPy sobel x
filters.sobel(im,1,new_im)

#SciPy sobel y
# filters.sobel(im,0,new_im)

# convolve(im, filter)

print Image.VERSION
pil_im = Image.fromarray(new_im)
pil_im.show()


