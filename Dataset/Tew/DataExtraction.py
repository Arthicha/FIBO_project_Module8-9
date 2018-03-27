__author__ = 'Zumo Arthicha Srisuchinnawong'
__date__ = '27/3/2018'
__program__ = 'dataset from jointed image'

'''
program description
    this program reads set of jointed image and compress them in to text file. This program has 5 configuration
parameters,
        1. DATASET_PATH: path of jointed image
        2. COMPRES_PATH: path to save text file
        3. N_FONT: amount of font
        4. IMG_SIZE: output image shape (x,y) or (---,|||)
        5. THRES: threshold for binarization
    this program will exit immediately if path of front doesn't exist, in this case it will return an error massage
as "ERROR: file /filepath/ does not exist".
'''

import numpy as np
import cv2
import os
import sys
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ip



PATH = os.getcwd()
DATASET_PATH = PATH + '\\TOP_SECRET'
COMPRES_PATH = PATH + '\\compress_dataset'

np.set_printoptions(threshold=np.inf)

NUM = ['0','1','2','3','4','5','6','7','8','9']
TYPE = ['E','N','T']
N_FONT = 5
FONT = list(map(str, range(1,N_FONT+1)))

WORDLIST = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "0", "1", "2", "3", "4",
            "5", "6", "7", "8", "9", "ศูนย์ ", "หนึ่ง ", "สอง ", "สาม ", "สี่ ", "ห้า ", "หก ", "เจ็ด ", "แปด ",
            "เก้า "]
FILEDICT = {"zero": "zero", "one": "one", "two": "two", "three": "three", "four": "four", "five": "five",
            "six": "six", "seven": "seven", "eight": "eight", "nine": "nine", "0": "0", "1": "1", "2": "2",
            "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "ศูนย์ ": "ZeroTH",
            "หนึ่ง ": "OneTH", "สอง ": "TwoTH", "สาม ": "ThreeTH", "สี่ ": "FourTH", "ห้า ": "FiveTH",
            "หก ": "SixTH",
            "เจ็ด ": "SevenTH", "แปด ": "EightTH", "เก้า ": "NineTH"}

DATASET = ['test','train','validate']

IMG_SIZE = (60,30)
THRES = 150


print(PATH)

# create directory
if not os.path.exists(COMPRES_PATH):
    os.makedirs(COMPRES_PATH)

write = ''

# loop through each font
for n in range(0,len(NUM)):
    num = NUM[n]
    for t in range(0,len(TYPE)):
        type = TYPE[t]
        '''if (num != '9') or (type != 'T'):
            continue'''
        process = 0
        for font in FONT:

            filename = num+type+font+'.png'
            filepath = DATASET_PATH+'\\'+filename
            compressname = FILEDICT[WORDLIST[n+t*10]]

            if font in FONT[:int(len(FONT)*0.2)]:
                set = DATASET[0]
            elif font in FONT[int(len(FONT)*0.2):int(len(FONT)*0.8)]:
                set = DATASET[1]
            else:
                set = DATASET[2]

            print('STATUS:', 'process',filename, 'as', set)

            if os.path.exists(filepath):

                img = cv2.imread(filepath,0)
                for x in range(img.shape[0],img.shape[1],img.shape[0]):
                    plate = img[:,x-img.shape[0]:x]

                    _, plate = cv2.threshold(plate, THRES, 255,0)

                    cv2.imshow('frame2',plate)
                    plate = ip.get_plate(plate,IMG_SIZE,dilate=35)
                    plate = plate[0].UnrotateWord

                    stringy = np.array2string(((plate.ravel())).astype(int),max_line_width=int(IMG_SIZE[0]*IMG_SIZE[1]*(5*img.shape[1]/img.shape[0]))
                                              ,separator=',')

                    write += stringy[1:-1] + "\n"


                    cv2.imshow('frame',plate)
                    cv2.waitKey(1)



            else:
                sys.exit('ERROR: file '+filepath+' does not exist')

            if process==len(FONT)*0.2:
                open(COMPRES_PATH+"\\dataset" + "_" + compressname+"_"+"test" + '.txt', 'w').close()
                file = open(COMPRES_PATH+"\\dataset"+"_"+compressname +"_"+"test"+ '.txt', 'a')
                file.write(write)
                file.close()
                write = ''
            elif process == len(FONT)*0.4:
                open(COMPRES_PATH+"\\dataset" + "_" + compressname + "_" + "validate" + '.txt', 'w').close()
                file = open(COMPRES_PATH+"\\dataset" + "_" + compressname + "_" + "validate" + '.txt', 'a')
                file.write(write)
                file.close()
                write = ''
            elif process == len(FONT)-1:
                open(COMPRES_PATH+"\\dataset" + "_" + compressname + "_" + "train" + '.txt', 'w').close()
                file = open(COMPRES_PATH+"\\dataset" + "_" + compressname + "_" + "train"+ '.txt', 'a')
                file.write(write)
                file.close()
                write = ''


            # increment variable
            process += 1


