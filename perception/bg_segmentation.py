import numpy as np
import cv2
import IPython

"""
parameters:
    matrix = (n, m, 3) rgb matrix
    lowerBound, upperBound = arrays of length 3
lb and ub specify a 3D box in RGB
sets points within the box to black; sets all other points to foreground color
see for reference: http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#inrange
returns:
    resulting (n, m) binary matrix
"""
def segmentBG(matrix, lowerBound, upperBound, grayForeground = False):
    matrix = cv2.inRange(matrix, lowerBound, upperBound)
    matrix[:,:,] = (255 - matrix[:,:,]) #switch black and white

    if grayForeground:
        matrix[:,:,] = matrix[:,:,]/2 #black --> black, white --> gray

    return matrix

def bgBoundsByMode(matrix, tolerance):
    '''
    finds mode of image, and sets it as the center of the background range
    increasing tolerance increases the range included in the background
    see for reference: http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html


        Paramaters
        ----------
        matrix: (n, m, 3) numpy array
        tolerance:  int

        Returns:
            two length 3 array
    '''
    dim = 3

    if(matrix == None):
        raise Exception('No Image Given')

    #create image histogram
    split = cv2.split(matrix)
    histSize = 256
    hists = np.array([cv2.calcHist([split[i]], [0], None, [histSize], [0, histSize]) for i in range(dim)])

    #find the mode of the image
    modes = [0 for i in range(dim)]
    for i in range(dim):
        modes[i] = np.argmax(hists[i])



    #add the tolerance to the bounds
    lowerBound = np.array([modes[i] - tolerance for i in range(dim)])
    upperBound = np.array([modes[i] + tolerance for i in range(dim)])

    lowerBound = np.array([ 18, 158,  94])
    upperBound = np.array([ 78, 218, 154])


    return lowerBound, upperBound
