import cv2
import numpy as np
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
from os import listdir
# path ='C:\\Users\\cha45\\Desktop\\fuc\\EN_dataset_img3\\'
# filelist = listdir('C:\\Users\\cha45\\Desktop\\fuc\\EN_dataset_img3\\')
# for x in filelist:
#     img = cv2.imread(path+x,0)
#     # img = ipaddr.binarize(img, ipaddr.OTSU_THRESHOLDING)
#     img = ipaddr.binarize(img,ipaddr.ADAPTIVE_CONTRAST_THRESHOLDING,(7,3))
#     img = ipaddr.remove_noise(img,8,3)
#     img = img/255
#     img =img.astype(int)
#     cv2.imshow("show",img)
#     cv2.waitKey(0)

f = open('C:\\Users\cha45\PycharmProjects\module8-9proj\Project8-9\CompressFork\dataset_ZeroTH_all_test.txt','r')
data = f.read()
f.close()
data=data.split('\n')
print(len(data))
data=data[:-1]
num =0
for x in data:
    lisss=x.split(',')
    # print(x)
    img = np.array(list(lisss[:-1]))
    # print(img)
    img = img.reshape(-1,(40))
    img = img.astype(np.uint8)*255
    # img=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    num += 1
    print(num)
    cv2.imshow("show",img)
    cv2.waitKey(0)

