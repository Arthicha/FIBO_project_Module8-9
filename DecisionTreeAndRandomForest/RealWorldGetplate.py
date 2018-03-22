import cv2
import numpy as np
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
cam = cv2.VideoCapture(1)

while 1:
    img = cam.read()
    grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    binarize = IP.binarize(grey_img,IP.SAUVOLA_THRESHOLDING,value=13)
    cv2.imshow("binarize",binarize)
    # IP.get_plate(grey_img,)
    cv2.waitKey(10)
