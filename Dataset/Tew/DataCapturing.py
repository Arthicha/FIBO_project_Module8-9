__author__ = 'Zumo Arthicha Srisuchinnawong'
__date__ = '27/3/2018'
__program__ = 'capturing image'

'''
program description
    this program captures images from webcam, then crops box of number or word and finally save image as 'png'. This
program has 3 configuration parameters,
        1. SAVE_PATH: path to save image
        2. CAM_PORT: port of the webcam
        3. IMG_SIZE: shape of output image
        4. IMG_COPY: amount of output image
        5. CLASS: class of output image, for example 4T1 means first front of the words สี่ ('four' in thai)
        6. THRES: threshold to capture box
        7. MINIMUM_LENGTH: minimum length of box
        8. AREA_RATIO: minimum area (percentage from image size)
'''

import numpy as np
import cv2
import os
import copy
import sys
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ip

IMG_SIZE = (100,100)
IMG_COPY = 100
CLASS = '4T0'
CAM_PORT = 1

THRES = 130
MINIMUM_LENGTH = 0.05
AREA_RATIO = 0.10

PATH = os.getcwd()
SAVE_PATH = PATH + '\\CAPTURE\\'+CLASS

NUM = ['0','1','2','3','4','5','6','7','8','9']
TYPE = ['E','N','T']
N_FONT = 5
FONT = list(map(str, range(1,N_FONT+1)))

cap = cv2.VideoCapture(CAM_PORT)
count = 0

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(gray, THRES, 255,0)
    frame = copy.copy(img)
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (0,0,255), 3)
    plate = []
    for c in range(0,len(contours)):
        cnt = contours[c]
        hi = hierarchy[0][c]
        epsilon =0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        area = cv2.contourArea(cnt)
        if (len(approx) == 4) and (hi[1]!=-1) and (area>=AREA_RATIO*IMG_SIZE[0]*IMG_SIZE[1]):
            cv2.drawContours(frame, [approx], 0, (0,0,255), 3)
            plate.append(approx)
    if len(plate) == 1:
        plate = np.array([plate[0][0][0],plate[0][1][0],plate[0][2][0],plate[0][3][0]])

        #img = ip.four_point_transform(gray,plate)
        img = ip.four_point_transform(gray,plate)
        img = cv2.resize(img,(IMG_SIZE[1],IMG_SIZE[0]))
        #img = ip.remove_perspective(gray,plate,(IMG_SIZE[0],IMG_SIZE[1]),auto_sort=True)
        #img = ip.rotation(img,(IMG_SIZE[1]//2,IMG_SIZE[0]//2),180)
        # Display the resulting frame
        cv2.imwrite(SAVE_PATH+'\\image'+str(count)+'.png',img)

        cv2.imshow('frame',frame)
        cv2.imshow('plate',img)
        count += 1
        if count > IMG_COPY:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()