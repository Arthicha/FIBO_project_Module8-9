__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 3.1
__description__ = 'Main Program'


'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

# 1. system module
import os
import sys
import copy

# 2. machine learning module
import tensorflow as tf

# 3. mathematical module
import numpy as np
import math
import random

# 4. our own module
from Tenzor import TenzorCNN,TenzorNN,TenzorAE
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from Retinutella_theRobotEye import Retinutella

# 5. visualization module
import matplotlib.pyplot as plt

# 6. image processing module
import cv2

'''*************************************************
*                                                  *
*             configuration variable               *
*                                                  *
*************************************************'''

N_CLASS = 30
IMG_SIZE = (30,60)

DATASET_DIR = '\\data0-9compress'
MODEL_DIR = ''


'''*************************************************
*                                                  *
*                 global variable                  *
*                                                  *
*************************************************'''

# set numpy to print/show all every element in matrix
np.set_printoptions(threshold=np.inf)

PATH = os.getcwd()
MODEL_PATH = PATH + MODEL_DIR


eye = [Retinutella('front',0,0,1)]

'''*************************************************
*                                                  *
*                     function                     *
*                                                  *
*************************************************'''


def getData(foldername='data0-9compress',n=-1,ttv=[0,1,2],dtype=np.uint8):

    '''
    this function get dataset from compress file in
    data0-9compress folder,
    :parameter: foldername = folder name
    :parameter: n = amount of data in each class
    :parameter: ttv = number of data to get in testing,
                training abd validating dataset
    :parameter: dtype is numpy data type np.uint8 or np.float32
    :return: testing, training and validating dataset
    '''

    global N_CLASS, IMG_SIZE

    TestTrainValidate = [[],[],[]]
    LabelTTT = [[],[],[]]

    suffix = ['test','train','validate']
    listOfClass = [0,1,2,3,4,5,6,7,8,9]+['zero','one','two','three','four','five','six',
                       'seven','eight','nine']+['ZeroTH','OneTH','TwoTH','ThreeTH','FourTH','FiveTH','SixTH',
                       'SevenTH','EightTH','NineTH']

    for s in range(1,3):
        print('STATUS: process data',str(100.0*s/3.0))
        for j in range(10,N_CLASS):
            object = listOfClass[j]
            f = open(os.getcwd()+'\\'+foldername+'\\dataset_'+str(object)+'_'+suffix[s]+'.txt','r')
            image = str(f.read()).split('\n')[:n]
            f.close()
            for i in range(len(image)):
                image[i] = np.fromstring(image[i], dtype=dtype, sep=',')
                image[i] = np.array(image[i])
                image[i] = np.reshape(image[i],(IMG_SIZE[0]*IMG_SIZE[1]))
            TestTrainValidate[s] += image
            obj = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            obj[j] = 1
            LabelTTT[s] += np.full((len(image),N_CLASS),copy.deepcopy(obj)).tolist()
        if s == 1:
            if 1:
                print('STATUS: shuffle-ing')
                def shuffly(a,b):
                    c = list(zip(a, b))
                    random.shuffle(c)
                    a, b = zip(*c)
                    return a,b
                a,b = shuffly(TestTrainValidate[1],LabelTTT[1])
                trainingSet = [np.array(a),np.array(b)]
                print('STATIS: complete shuffle-ing')
        del image
        del object
    testingSet  = [TestTrainValidate[ttv[0]],LabelTTT[ttv[0]]]
    validationSet = [TestTrainValidate[ttv[2]],LabelTTT[ttv[2]]]
    return testingSet,trainingSet,validationSet


def crop_image(img,msk,tol=0):
    # img is image data
    # tol  is tolerance
    mask = msk>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def Get_Plat2(org,thres_kirnel=21,min_area=0.05,max_area=0.9):
    image = copy.deepcopy(org)
    image = IP.binarize(image,method=IP.SAUVOLA_THRESHOLDING,value=thres_kirnel)
    image_area = image.shape[0]*image.shape[1]
    img, contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    plate = []
    for i in range(0,len(contours)):
        cnt = contours[i]
        hi = hierarchy[0][i]
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        area = cv2.contourArea(cnt)
        if (area>min_area*image_area) and (area < image_area*max_area)and(len(approx) == 4) and(hi[2]!=-1)and(hi[1]==-1):
            plate.append(approx)
            cv2.drawContours(org, [approx], -1, (255, 255, 255), 2)
    return plate

def Get_Word2(plate,thres_kirnel=21,boundary=20,black_tollerance=10,image_size=(60,30)):
    listOfWord = []
    for i in range(0,len(plate)):
        plate[i] = np.array(plate[i])
        plate[i] = np.reshape(plate[i],(4,2))
        plate[i] = IP.four_point_transform(org,plate[i])
        word = IP.binarize(plate[i],method=IP.SAUVOLA_THRESHOLDING,value=thres_kirnel)
        wx,wy = word.shape
        bou = boundary
        word = 255-np.array(word)
        word = word[bou:wy-bou,bou:wx-bou]
        plate[i] = plate[i][bou:wy-bou,bou:wx-bou]
        #word = IP.morph(word,mode=IP.OPENING,value=[5,5])
        word = crop_image(plate[i],word,tol=black_tollerance)
        if word != []:
            word = cv2.resize(word,image_size)
        listOfWord.append(word)
    return listOfWord

'''*************************************************
*                                                  *
*                   main program                   *
*                                                  *
*************************************************'''

closing = 3
SAU_KIR = 21
min_area = 0.05
while(1):
    image = eye[0].getImage()
    org = copy.deepcopy(image)
    plate = IP.Get_Plate2(org)
    word = IP.Get_Word2(org,plate)
    for i in range(0,len(word)):
        if len(word[i]) != 0:
            eye[0].show(word[i],frame='plate'+str(i))
    eye[0].show(org,frame='original',wait=10)
    for i in range(0,len(word)):
        cv2.destroyWindow('plate'+str(i))
    #eye[0].show(image,wait=10)
eye[0].close()




