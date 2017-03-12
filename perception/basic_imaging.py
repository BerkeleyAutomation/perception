from scipy import signal
import h5py
import numpy as np
import cv2
import math

"""
parameters:
    matrix = (n, m) binary matrix
applies sobel kernel to highlight edges/areas of high contrast
returns:
    resulting (n, m) binary matrix
"""
def contourise(matrix):
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.transpose(sobelX)

    sobConvX = signal.fftconvolve(matrix, sobelX, mode = "same")
    sobConvY = signal.fftconvolve(matrix, sobelY, mode = "same")

    return np.hypot(sobConvX, sobConvY)

"""
parameters:
    matrix = (n, m) binary matrix
used to prepare image for neueral net format
returns:
    (n, m, 1) binary matrix
"""
def add_dim(matrix):
    zeros = np.zeros([matrix.shape[0],matrix.shape[1], 1])
    zeros[:,:,0] = matrix
    return zeros

"""
parameters:
    matrix = (n, m) matrix
stacks grayscale matrix to obtain rgb formatted version
returns:
    resulting (n, m, 3) matrix
"""
def formatRGB(matrix):
    a, b = np.copy(matrix), np.copy(matrix)
    return np.stack((matrix, a, b), axis = -1)

"""
splits image into 2**bits number of discrete color bins
"""
def bins(matrix, bits):
    matrix = matrix/255
    matrix = np.ceil(matrix * 2**bits)/2**bits
    return matrix * 255

"""
parameters:
    matrix = (n, m) matrix
    ly = lowY, hy = highY, lx = lowX, hy = highX (integers)
returns:
    matrix with white bounding box, specified by coords, drawn
"""
def drawBox(matrix, ly, hy, lx, hx):
    matrix = np.copy(matrix)

    #draw the vertical lines
    for x in range(lx, hx):
      matrix[ly][x] = 255
      matrix[hy][x] = 255

    #draw the horizontal lines
    for y in range(ly, hy):
      matrix[y][lx] = 255
      matrix[y][hx] = 255

    return matrix

"""
parameters:
    matrix = (n, m) matrix
    moreBlur = boolean specifying whether to increase size of kernel
applies gaussian kernel
returns:
    resulting (n, m) matrix
"""
def deNoise(matrix, moreBlur = False):
    blurAmt = 3
    if moreBlur:
        blurAmt = 5

    return cv2.medianBlur(matrix, blurAmt)

"""
parameters:
    folder = file path to image, name = image name
    extension = jpg or png
reads image from memory
returns:
    specified image as array
"""
def readIm(folder, name, extension):
    imgName = folder + "/" + name + "." + extension
    return cv2.imread(imgName)

"""
parameters:
    matrix = to be converted to image
    folder = file path to new location for image, name = image name
    extension = jpg or png
writes image to memory
returns:
    None
"""
def writeIm(matrix, folder, name, extension):
    imgName = folder + "/" + name + "." + extension
    cv2.imwrite(imgName, matrix)

"""
parameters:
    matrix = to be saved in HDF5 file as a dataset
    folder = name of folder corresponding to policy
    name = name of particular
writes image to memory
returns:
    None
"""
def writeData(matrix, folder, rolloutNum, frameNum):
    with h5py.File('../' + folder + '/' + 'database.h5', 'w') as hf:
        try:
            for _ in range(0, 2):
                hf.create_group('rollout' + rolloutNum)
        except KeyError:
            hf[group].create_dataset('frame' + frameNum, data = matrix)

"""
parameters:
    folder = file path to binary data
reads image from datbase of binary images
returns:
    specified image as array
"""
def readData(folder, rolloutNum, frameNum):
    with h5py.File('../' + folder + '/' + 'database.h5', 'w') as hf:
        data = hf['rollout' + rolloutNum]['frame' + frameNum]
        return numpy.array(data)

"""
parameters:
    matrix = (n, m) matrix
    ly = lowY, h = height, lx = lowX, w = width (integers)
returns:
    crop of matrix, specified by coords
"""
def crop(matrix, ly, h, lx, w):
    matrix = np.copy(matrix)

    matrix = matrix[:,lx:lx + w]
    matrix = matrix[ly:ly + h]

    return matrix

"""
sine function for degrees
"""
def sin(deg):
    rad = math.radians(deg)
    return math.sin(rad)

"""
cosine function for degrees
"""
def cos(deg):
    rad = math.radians(deg)
    return math.cos(rad)
