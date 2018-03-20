import numpy as np
import cv2
from os import listdir
import os

current_dir = os.getcwd()
print(current_dir)
filelist = [x for x in listdir(current_dir + "\\") if x[-4:] == ".txt"]
# print(filelist)
for file in filelist:
    f = open(current_dir + "\\" + file, 'r')
    data = f.read()
    f.close()
    data = data.split('\n')[:-1]
    data = list(map(lambda x: (1-(np.array(x.split(',')).reshape(-1, (60))).astype(np.uint8))*255, data))
    '''pre process image with erode and dilate'''
    data = list(map(lambda x: cv2.dilate(x, np.ones([4, 3])), data))
    data = list(map(lambda x: cv2.erode(x, np.ones([1, 4])), data))
    '''find contour and hierachy'''
    data = list(map(lambda x: cv2.findContours(x,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_NONE),data))

    '''filter outer most hierachy (-1) '''
    data = list(map(lambda x:[x[1],list(map(lambda z: (x[2][0].tolist()).index(z),list(filter(lambda y: y[3] == -1,x[2][0].tolist()))))] ,data))
    # data = list(map(lambda  x : [x[1],list(map(lambda z: ((x[2]).tolist()).index(z.tolist()),list(filter(lambda y: y[0,3] == -1,x[2]))))],data))
    '''list of contour (character) left '''
    character= list(map(lambda x: [x[0][y] for y in x[1] ],data))

    ''' find possible number of contour (charaacter) in 1 file'''
    possibility= set(list(map(lambda x:len(x),character)))
    print(possibility)
    print(file)
    print('********************')