import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr

# C:\\Users\cha45\Desktop\\fuc\TH_dataset_img3\\
# C:\\Users\cha45\Desktop\\fuc\TH_dataset_img\\"C:\\Users\cha45\Desktop\\fuc\ENG_dataset_img2\\
savePath = "C:\\Users\cha45\Desktop\\fuc\EN_dataset_img3\\"
# # C:\\Users\cha45\Downloads\\front thai\\front thai\\
# # C:\\Users\cha45\Downloads\DATA SET - fonts2-20180226T150229Z-001\DATA SET - fonts2\\
# # C:\\Users\cha45\Downloads\DATA SET - fonts-20180224T101407Z-001\DATA SET - fonts\\
Font_Path = "C:\\Users\cha45\Downloads\DATA SET - fonts-20180224T101407Z-001\DATA SET - fonts\\"
s = listdir(Font_Path)
s = [x for x in s if ".ttf" in x or ".otf" in x or ".ttc" in x or ".TTF" in x or ".OTF" in x or ".TTC" in x]
s = s[:-72]
s = [x for x in s if s.index(x) % 4 == 0]
# s = s[s.index("TH Dan Vi Vek Bold Italic ver 1.03.otf"):]
max_x = 0.0
max_y = 0.0
# three (115,46) 81.31
# eight (113,50)
# fourth 1/2 (38,42)
# fourth 1/2 (40,65)
print(len(s))
for x in s:
    print(x)
    image = ipaddr.font_to_image(Font_Path + x, 32,
                                 0, "2")
    k = ipaddr.get_plate(image, (40, 30))
    shape=sorted(k[0].Original_Word_Size)
    blur_image = ipaddr.blur(k[0].UnrotateWord, 11, [5, 250, 250])
    # blur_image = ipaddr.blur(k[0].UnrotateWord, 10, 3)
    # blur_image = ipaddr.blur(k[0].UnrotateWord, 9, 3)
    binarize_image =ipaddr.binarize(blur_image,method=ipaddr.ADAPTIVE_CONTRAST_THRESHOLDING,value=[3,0.5])

    final=ipaddr.remove_noise(binarize_image, 8, 3)
    # print(shape)
    # if shape[1] > max_x:
    #     max_x = shape[1]
    # if shape[0] > max_y:
    #     max_y = shape[0]
    cv2.imshow("8",final)
    cv2.imshow("8_show", blur_image )
    cv2.imshow("8_rot", k[0].UnrotateWord)
    cv2.waitKey(0)
# print("MAX :"+str((max_x,max_y)))