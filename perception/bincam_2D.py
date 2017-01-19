import cv2
import numpy as np
# from alan.lfd_slware.options import SLVOptions
#from alan.lfd_amazon.options import AmazonOptions
from bgSegmentation import segmentBG, bgBoundsByMode
from basicImaging import addDim, deNoise

class BinaryCamera():


    def __init__(self, Amazon = False):
        self.vc = None
        self.Amazon = Amazon


        self.lowerBound = -1
        self.upperBound = -1


    """ idNum usually 0, 1, 2, 3"""
    def open(self, idNum = 0, threshTolerance = 40):
        for idNum in range(idNum,100):
            self.vc = cv2.VideoCapture(idNum)
            print idNum
            for i in range(10):
                self.vc.read()
            r, f = self.vc.read()
            if(not f == None):
                break
        #self.vc = cv2.VideoCapture(idNum)

        # self.vc.set(cv2.cv.CV_CAP_P]ROP_BUFFERSIZE, 1)
        print "EXPOSURE ",self.vc.get(cv2.cv.CV_CAP_PROP_SATURATION)
        #for more saturated version, set saturation to .35
        self.vc.set(cv2.cv.CV_CAP_PROP_SATURATION ,0.15)
        self.vc.set(cv2.cv.CV_CAP_PROP_CONTRAST ,0.12)
        self.vc.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS ,0.5)

        for i in range(10):
            sampleImg = self.read_raw()

        self.lowerBound, self.upperBound = bgBoundsByMode(sampleImg, threshTolerance)

    def release(self):
        self.vc.release()

    def close(self):
        if self.is_open():
            self.vc.release()

    def is_open(self):
        return self.vc is not None and self.vc.isOpened()

    def read_raw(self):
        #clear buffer
        # for i in range(10):
        #     self.vc.read()
        rval, frame = self.vc.read()

        return frame

    def read_frame(self, show=False, record=False, state=None):
        """ Returns cropped frame of raw video """
        frame = self.read_raw()

        if not self.Amazon:
            frame = frame[0+SLVOptions.OFFSET_Y:SLVOptions.HEIGHT+SLVOptions.OFFSET_Y, 0+SLVOptions.OFFSET_X:SLVOptions.WIDTH+SLVOptions.OFFSET_X]
        else:
            frame = frame[0+AmazonOptions.OFFSET_Y:AmazonOptions.HEIGHT+AmazonOptions.OFFSET_Y, 0+AmazonOptions.OFFSET_X:AmazonOptions.WIDTH+AmazonOptions.OFFSET_X]
        #frame = cv2.resize(frame, (420, 420))

        if record:
            self.recording.append(frame)
            self.states.append(state)
        if show:
            cv2.imshow("preview", frame)
        return frame

    def jpg_correct(self,frame):
        cv2.imwrite('get_jp.jpg',frame)
        frame= cv2.imread('get_jp.jpg')
        return frame

    def display_frame(self,show=False):
        frame = self.read_binary_frame()
        #frame = self.read_frame()
        frame = cv2.resize(frame,(400,400))
        if show:
            cv2.imshow("preview", frame)
            cv2.waitKey(30)
        return frame

    def read_binary_frame(self, show=False, record=False, state=None):
        frame = self.read_frame(show, record, state)
        return addDim(self.apply_mask(frame))

    def apply_mask(self, img):
        "threshhold background"
        img = segmentBG(img, self.lowerBound, self.upperBound)
        # img = deNoise(img)
        # img = np.array(img,dtype='float32')
        return img

    def read_single_channel_binary(self, show=False, record=False, state=None):
        frame = self.read_frame(show, record, state)
        return frame, self.apply_mask(frame)

if __name__ == "__main__":
    bincam = BinaryCamera(Amazon = True)


    bincam.open(threshTolerance= 80)

    frame = bincam.read_raw()


    while (1):
        # frame = bincam.display_frame()

        # out = frame + o
        # cv2.imshow("camera", out)
        # print("reading")
        a = cv2.waitKey(30)
        if a == 1048603:
            cv2.destroyWindow("camera")
            break

        frame = bincam.read_raw()
        cv2.imshow("cam", frame)
        cv2.waitKey(30)
