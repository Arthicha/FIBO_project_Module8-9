import tensorflow as tf
from tensorflow.python.framework import ops

# mathematical module
import numpy as np
import math
import random as ran
import random

# display module
import matplotlib.pyplot as plt

# system module
import os
import sys
import cv2
import copy

# my own library
from Tenzor import TenzorCNN, TenzorNN, TenzorAE

from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP

def aspectRatio(img_f):
    img_fc = copy.deepcopy(img_f)
    img_fc, cfc, hfc = cv2.findContours(img_fc, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    try:
        xfc,yfc,wfc,hfc = cv2.boundingRect(cfc[-1])
    except:
        return 1.0
    aspect_ratio = float(wfc)/hfc
    return aspect_ratio

def getWordSize(img_f):
    img_fc = copy.deepcopy(img_f)
    img_fc, contours, hfc = cv2.findContours(img_fc, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    try:
        cnt = contours[-1]
    except:
        return 0,0
    leftmost = np.array(cnt[cnt[:,:,0].argmin()][0])
    rightmost = np.array(cnt[cnt[:,:,0].argmax()][0])
    topmost = np.array(cnt[cnt[:,:,1].argmin()][0])
    bottommost = np.array(cnt[cnt[:,:,1].argmax()][0])
    return np.linalg.norm(leftmost-rightmost),np.linalg.norm(topmost-bottommost)


def Adapt_Image(image):
    inv_image = 255 - image
    dilate = cv2.dilate(inv_image, np.ones((10, 10)))
    ret, cnt, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    rect = cv2.minAreaRect(cnt[0])
    if rect[1][0] > rect[1][1]:
        y1 = int(rect[1][0] / 2) + int(rect[0][0])
        y2 = int(rect[0][0]) - int(rect[1][0] / 2)
        x1 = int(rect[1][0] / 2) + int(rect[0][1])
        x2 = int(rect[0][1]) - int(rect[1][0] / 2)
    else:
        y1 = int(rect[1][1] / 2) + int(rect[0][0])
        y2 = int(rect[0][0]) - int(rect[1][1] / 2)
        x1 = int(rect[1][1] / 2) + int(rect[0][1])
        x2 = int(rect[0][1]) - int(rect[1][1] / 2)
    return cv2.resize(image[x2:x1, y2:y1], (30, 60))


def Get_Plate(img, sauvola_kernel=11, perc_areaTh=[0.005, 0.5], numberOword=(0.5, 1.5), minimumLength=0.05,
              plate_opening=3, char_opening=13, Siz=60.0):
    org = copy.deepcopy(img)
    x, y, c = org.shape
    areaTh = (perc_areaTh[0] * x * y, perc_areaTh[1] * x * y)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_o = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
    img_s = IP.binarize(gray, IP.SAUVOLA_THRESHOLDING, sauvola_kernel)

    img_s = np.array(img_s, dtype=np.uint8)
    cv2.imshow("S", img_s)
    cv2.waitKey(0)
    img = cv2.bitwise_and(img_s, img_o)
    cv2.imshow("S1", img)
    cv2.waitKey(0)
    img_c = copy.deepcopy(img)
    img_c = IP.morph(img_c, mode=IP.ERODE, value=[plate_opening, plate_opening])
    cv2.imshow("S2", img_c)
    cv2.waitKey(0)
    # org = copy.deepcopy(img_c)

    # cv2.imshow('frame',img_c)
    # cv2.waitKey(0)
    img_c, contours, hierarchy = cv2.findContours(img_c, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    subImg = []

    for ic in range(0, len(contours)):
        cnt = contours[ic]
        hi = hierarchy[0][ic]
        epsilon = minimumLength * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)
        if (len(approx) == 4) and (hi[0] != -1) and (hi[1] != -1) and (area > areaTh[0]) and (area < areaTh[1]):

            img_p = IP.remove_perspective(img, approx, (int(Siz), int(Siz)), org_shape=(x, y))
            white = np.count_nonzero(img_p) / (Siz * Siz)
            if (white > 0.1):
                img_m = IP.morph(img_p, mode=IP.OPENING, value=[char_opening, char_opening])

                aspect_ratio = aspectRatio(img_m)
                sz = min(getWordSize(img_m))
                if (aspect_ratio > numberOword[0]) and (aspect_ratio < numberOword[1]):
                    if aspect_ratio < 1.00:
                        rotating_angle = [0, 180]
                    else:
                        rotating_angle = [90, -90]
                else:
                    if aspect_ratio < 1.00:
                        rotating_angle = [90, -90]
                    else:
                        rotating_angle = [0, 180]
                diff = [0, 0]

                for a in range(0, len(rotating_angle)):
                    angle = rotating_angle[a]
                    img_r = IP.rotation(img_p, (img_p.shape[0] / 2, img_p.shape[1] / 2), angle)
                    ctr = int(Siz / 2)
                    img_r = img_r[ctr - 15:ctr + 15, ctr - 30:ctr + 30]
                    img_r[:, 0:5] = 255
                    img_r[:, 60 - 6:60 - 1] = 255
                    img_r[0:5, :] = 255
                    img_r[30 - 6:30 - 1, :] = 255
                    # img_r = Adapt_Image(img_r)
                    subImg.append(img_r)
                    '''chkO = checkOreantation(img_r)
                    diff[a] = [chkO,copy.deepcopy(img_r)]
                if diff[0][0] > diff[1][0]:
                    subImg.append(diff[0][1])
                else:
                    subImg.append(diff[1][1])'''
                cv2.drawContours(org, [approx], 0, (0, 0, 255), 3)

    return org, subImg


while (True):

    # Capture frame-by-frame
    frame = cv2.imread("Dataset\\Real\\sample_1.jpg",1)
    org = copy.deepcopy(frame)  # Our operations on the frame come here
    org, LoM = Get_Plate(frame,15,[0.5,100000000],minimumLength=0.00000000000001)
    LoM = np.array(LoM)
    # Display the resulting frame
    LoM = cv2.imread('file', 0)
    LoM = np.array(LoM)
    LoC = copy.deepcopy(LoM)
    LoC = LoC // 255
    LoC = np.reshape(LoC, (LoC.shape[0], 30 * 60))

    # LoC = pred_class.eval(feed_dict={x: LoC, keep_prob: 1.0})

    for i in range(0, len(LoM)):
        LoMi = cv2.resize(LoM[i], (300, 150))
        cv2.imshow('output_' + str(i) + '_' + str(LoC[i]), LoMi, )
        cv2.moveWindow('output' + str(i), 300 * i, 80)
    cv2.imshow('original', org)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
    for i in range(0, len(LoM)):
        cv2.destroyWindow('output_' + str(i) + '_' + str(LoC[i]))
