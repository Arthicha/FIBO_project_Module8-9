import cv2
import copy
import numpy as np
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP


def Adapt_Image(image):
    output_shape =(60,30) #
    ''' (width,height) of output(return) picture'''
    dilate_kernel_shape=(10,10)
    '''2d (x,y) can adjust offset if too less can't extract'''

    inv_image = 255 - image
    dilate = cv2.dilate(inv_image, np.ones(dilate_kernel_shape))
    ret, cnt, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    rect = cv2.minAreaRect(cnt[0])
    print(rect)
    y1 = int(rect[1][0] / 2 + rect[0][0])
    y2 = int(rect[0][0] - rect[1][0] / 2)
    x1 = int(rect[1][1] / 2 + rect[0][1])
    x2 = int(rect[0][1] - rect[1][1] / 2)
    return cv2.resize(image[x2:x1, y2:y1], output_shape)

img = cv2.imread("twoTH.jpg",0)
img = Adapt_Image(img)
cv2.imshow("adapt",img)
cv2.waitKey(0)