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

NUM = ['0','1','2','3','4','5','6','7','8','9']
TYPE = ['E','N','T']
N_FONT = 5
FONT = list(map(str, range(6,6+N_FONT+1)))

for n in NUM:
    for t in TYPE:
        for f in FONT:

            clas = n+t+f
            if clas == '4T6':
                print(clas)
                image_path = PATH + '\\CAPTURE\\'+clas
                print(listdir(image_path))

