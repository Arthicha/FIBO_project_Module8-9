__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 3.0
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

'''*************************************************
*                                                  *
*                 global variable                  *
*                                                  *
*************************************************'''

N_CLASS = 30
IMG_SIZE = (30,60)
np.set_printoptions(threshold=np.inf)

eye = [Retinutella('front',1,0,1)]

'''*************************************************
*                                                  *
*                   main program                   *
*                                                  *
*************************************************'''

while(1):
    image = eye[0].getListOfPlate()
    print(image[0])
    eye[0].show(image,wait=10)

eye[0].close()

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


