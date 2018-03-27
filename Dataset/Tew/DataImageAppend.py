__author__ = 'Zumo Arthicha Srisuchinnawong'
__date__ = '27/3/2018'
__program__ = 'append image'

import numpy as np
import cv2
import os
import copy
import sys
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ip


PATH = os.getcwd()
SAVE_PATH = PATH + '\\BOTTOM_SECRET'

MAXIMUM_IMG = 10000

NUM = ['0','1','2','3','4','5','6','7','8','9']
TYPE = ['E','N','T']
N_FONT = 5
FONT = list(map(str, range(6,6+N_FONT+1)))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

for n in NUM:
    for t in TYPE:
        for f in FONT:

            clas = n+t+f
            if clas == '4T6':
                image_path = PATH + '\\CAPTURE\\'+clas
                image_name = os.listdir(image_path)
                img_x = np.array([])
                for img_n in image_name:
                    img = cv2.imread(image_path+'\\'+img_n,0)
                    if img_x.shape[0] == 0:
                        img_x = copy.deepcopy(img)
                    else:
                        img_x = np.concatenate((img_x, img), axis=1)
                    if img_x.shape[1] >= MAXIMUM_IMG:
                        cv2.imwrite(SAVE_PATH+'\\'+clas+'.png',img_x)
                        break
                    cv2.imshow('frame',img)
                    cv2.waitKey(10)


