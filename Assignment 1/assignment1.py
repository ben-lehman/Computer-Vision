from PIL import Image, ImageDraw
from pylab import *
from scipy.ndimage import filters
import numpy as np

# ---------- 1.2 --------- #

# im = array(Image.open('seam_carving_input3.jpg').convert('L'))

# imx = zeros(im.shape)
# filters.sobel(im, 1, imx)

# imy = zeros(im.shape)
# filters.sobel(im, 0, imy)

# magnitude = sqrt(imx**2+imy**2)

# pil_im = Image.fromarray(uint8(magnitude))
# pil_im.save("test.jpg", 'JPEG')
# pil_im.show()

# ---------- 1.3 --------- #

im = array(Image.open('seam_carving_input1.jpg').convert('L'))

imx = zeros(im.shape)
filters.sobel(im, 1, imx)

imy = zeros(im.shape)
filters.sobel(im, 0, imy)

magnitude = sqrt(imx**2 + imy**2)


def cumulativeMinumumEnergyMap(image, seam_direction):
    energy_map = np.copy(image)
    if seam_direction == 0:
        image = image.transpose()
        energy_map = energy_map.transpose()
    # print image.shape
    row = image.shape[0]
    column = image.shape[1]
    for i in range(0, row):
        for j in range(0, column):
            value = image[i][j]
            parents = []
            for q in range(-1, 2):
                if i == 0:
                    parents.append(0)
                elif j + q < 0 or j + q >= (column - 1):
                    parents.append(float(inf))
                else:
                    parents.append(energy_map[i - 1][j + q])
            min_parent = min(parents)
            cum_value = value + min_parent
            energy_map[i][j] = cum_value
    if seam_direction == 0:
        energy_map = energy_map.transpose()
    return energy_map


em = cumulativeMinumumEnergyMap(magnitude, 1)
# pil_im = Image.fromarray(uint8(em))
# pil_im.save('cem_horizontal3.jpg', "JPEG")
# pil_im.show()

# ---------- 1.4 --------- #


def findOptimalVerticalSeam(energy_map):
    vertical_seam = []
    # print "shape: ", energy_map.shape
    row = energy_map.shape[0]
    column = energy_map.shape[1]
    bottom_row = energy_map[row - 1]
    initial_min = min(bottom_row)
    min_col = np.where(bottom_row == initial_min)[0][0]
    vertical_seam.append(min_col)
    cur_col = min_col
    for i in range(row - 2, -1, -1):
        parents = []
        for j in range(-1, 2):
            if cur_col + j < 0 or cur_col + j >= column:
                parents.append(float(inf))
            else:
                value = energy_map[i][j + cur_col]
                parents.append(value)
        min_parent = min(parents)
        cur_col = cur_col + parents.index(min_parent) - 1
        vertical_seam.append(cur_col)
    # displaySeam('seam_carving_input2.jpg', vertical_seam, 1)
    return vertical_seam


# ---------- 1.5 --------- #


def findOptimalHorizontalSeam(energy_map):
    energy_map = energy_map.transpose()
    horizontal_seam = findOptimalVerticalSeam(energy_map)
    return horizontal_seam


# ---------- 1.6 --------- #


def displaySeam(im, seam, seam_direction):
    image = Image.open(im)
    # image = Image.fromarray(reduced_color_image)
    draw = ImageDraw.Draw(image)
    line_arr = []
    for x in range(0, len(seam)):
        if seam_direction == 1:
            cur_cord = (seam[x], x)
        else:
            cur_cord = (x, seam[x])
        line_arr.append(cur_cord)
    draw.line(line_arr, fill=128)
    image.save("seam_h3.jpg", 'JPEG')
    image.show()


# im = "seam_carving_input3.jpg"
# vertical_seam = findOptimalVerticalSeam(em)
# horizontal_seam = findOptimalHorizontalSeam(em)
# displaySeam(im, vertical_seam, 1)
# displaySeam(im, horizontal_seam, 0)

# ---------- 1.7 --------- #


def findGreedyVerticalSeam(energy_map):
    vertical_seam = []
    row = energy_map.shape[0]
    column = energy_map.shape[1]
    initial_min = min(energy_map[0])
    cur_col = np.where(energy_map[0] == initial_min)[0][0]
    for x in range(1, row - 1):
        parents = []
        for y in range(-1, 2):
            if cur_col + y < 0 or cur_col + y >= column:
                parents.append(float(inf))
            else:
                value = energy_map[x][cur_col + y]
                parents.append(value)
        min_parent = min(parents)
        cur_col = cur_col + parents.index(min_parent) - 1
        vertical_seam.append(cur_col)
    return vertical_seam


def findGreedyHorizontalSeam(energy_map):
    energy_map = energy_map.transpose()
    horizontal_seam = findGreedyVerticalSeam(energy_map)
    return horizontal_seam

# em_v = cumulativeMinumumEnergyMap(im, 1)
# em_h = cumulativeMinumumEnergyMap(im, 0)
# greedy_vertical_seam = findGreedyVerticalSeam(em)
# greedy_horizontal_seam = findGreedyHorizontalSeam(em)

# im = "seam_carving_input1.jpg"
# displaySeam(im, greedy_vertical_seam, 1)
# displaySeam(im, greedy_horizontal_seam, 0)


# ---------- 1.8 --------- #

def setSeamPath(seam, seam_direction):
    path = []
    for x in range(0, len(seam)):
        if seam_direction == 1:
            cur_cord = (x, seam[x])
        else:
            cur_cord = (seam[x], x)
        path.append(cur_cord)
    return path

def reduceWidth(im, energy_map):
    print 'shape: ', im.shape
    row = im.shape[0]
    column = im.shape[1]
    reduced_im = np.copy(im)
    # print reduced_im.shape
    reduced_im = np.delete(reduced_im, -1, 1)
    # print reduced_im.shape
    reduced_em = np.zeros((row, column - 1))
    seam = findOptimalVerticalSeam(energy_map)
    path = setSeamPath(seam, 1)
    path_set = set(path)
    seen_set = set()
    for i in range(0, row):
        for j in range(0, column - 1):
            if (i,j) not in path_set and i not in seen_set:
                reduced_im[i][j] = im[i][j]
                reduced_em[i][j] = energy_map[i][j]
            elif (i,j) in path_set:
                seen_set.add(i)
            else:
                reduced_im[i][j - 1] = im[i][j]
                reduced_em[i][j - 1] = energy_map[i][j]
    # print seen_set
    return (reduced_im, reduced_em)

def reduceHeight(im, energy_map):
    print 'shape: ', im.shape
    row = im.shape[0]
    column = im.shape[1]
    reduced_im = np.copy(im)
    reduced_im = np.delete(reduced_im, -1, 0)
    reduced_em = np.zeros((row - 1, column))
    seam = findOptimalHorizontalSeam(energy_map)
    path = setSeamPath(seam, 0)
    path_set = set(path)
    seen_set = set()
    for j in range(0, column):
        # print j, '/ ', column
        for i in range(0, row - 1):
            if (i,j) not in path_set and j not in seen_set:
                reduced_im[i][j] = im[i][j]
                reduced_em[i][j] = energy_map[i][j]
            elif (i,j) in path_set:
                seen_set.add(j)
            else:
                reduced_im[i - 1][j] = im[i][j]
                reduced_em[i - 1][j] = energy_map[i][j]
    # print seen_set
    return (reduced_im, reduced_em)

im = array(Image.open('seam_carving_input1.jpg'))
# im = "seam_carving_input1.jpg"

# reduced_color_image, reduced_energy_map = reduceWidth(im, em)
# pil_im = Image.fromarray(reduced_color_image)
# pil_im.show()

# ---------- 1.8 --------- #

reduced_color_image = array(Image.open('seam_carving_input1.jpg'))

p = 500
while(p > 0):
    print p
    reduced_color_image, reduced_energy_map = reduceWidth(im, em)
    im = reduced_color_image
    em = reduced_energy_map
    p -= 1

pil_im = Image.fromarray(reduced_color_image)
pil_im.save("test1.jpg", "JPEG")
pil_im.show()

