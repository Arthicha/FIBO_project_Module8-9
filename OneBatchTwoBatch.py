__author__ = 'Zumo Arthicha Srisuchinnawong'


import os
import cv2
import numpy as np

from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from Retinutella_theRobotEye import Retinutella

CLASS = 25
MAIN_PATH = os.getcwd()

PATH = MAIN_PATH+'\\real_image\\image_class_'+str(CLASS)

cam = Retinutella('CaptureCam',1,0,1)

try:
    f = open(PATH+'\\number_config.txt','r')
    index = int(f.read())
    f.close()
except:
    index = 0
try:
    while 1:
        image,LoP = cam.getListOfPlate()
        #image = cam.getImage()


        if LoP != None:
            cam.show(image)
            if 1:#len(LoP) == 1:
                imp = LoP[0].UnrotateWord
                cam.show(image)
                cam.show(imp,frame='plate')

                key = cv2.waitKey(0)
                cv2.destroyWindow('plate')
                if key == 8:
                    print('INPUT: delete image')
                    pass
                elif key in [32,13]:
                    print('INPUT: save image')

                    if not os.path.isdir(PATH):
                        os.makedirs(PATH)
                        f = open(PATH+'\\number_config.txt','w')
                        f.write(str(0))
                        f.close()

                    cv2.imwrite(PATH+'\\image'+str(index)+'.jpg',imp)
                    index += 1
                    f = open(PATH+'\\number_config.txt','w')
                    f.write(str(index))
                    f.close()


        else:
            cam.show(image,wait=10)
except KeyboardInterrupt:
    print('ucita !')

cam.close()

