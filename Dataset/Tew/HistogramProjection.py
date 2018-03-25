import numpy as np
import cv2
from os import listdir
import os

current_dir = os.getcwd()
print(current_dir)
filelist = [x for x in listdir(current_dir + "\\Augmented_dataset") if x[-4:] == ".txt"]
print(filelist)
for file in filelist:
    f = open(current_dir + "\\Augmented_dataset\\" + file, 'r')

    data = f.read()
    f.close()
    data = data.split('\n')[:-1]
    data = list(map(lambda x: np.array(x.split(',')).reshape(-1, (60)), data))
    histogram_x = list(map(lambda x: [x[:,y:y+3] for y in range(0,60,3)],data))
    histogram_x = list(map(lambda x: list(map(lambda y:np.sum(1-y.astype(float)),x)),histogram_x))
    histogram_x = list(map(lambda x:np.array(x)/max(x),histogram_x))
    histogram_y = list(map(lambda x: [x[y:y+3,:] for y in range(0,30,3)],data))
    histogram_y = list(map(lambda x: list(map(lambda y:np.sum(1-y.astype(float)),x)),histogram_y))
    histogram_y = list(map(lambda x: np.array(x) / max(x), histogram_y))
    all_histogram=np.array(list(map(lambda x,y: np.concatenate([x,y]),histogram_x,histogram_y)))
    all_histogram=np.around(all_histogram,4)
    np.savetxt("project2\\histogram_"+file,all_histogram,fmt="%1.4f",delimiter=",")
    print(file)


# data=data.split('\n')
# print(len(data))
# data=data[:-1]
# num =0
# for x in data:
#     lisss=x.split(',')
#
#     # print(x)
#     img = np.array(list(lisss[:]))
#     # print(img)
#     img = img.reshape(-1,(60))
#     img = img.astype(np.uint8)*255
#     # img=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
#     num += 1
#     print(num)
#     cv2.imshow("show",img)
#     cv2.waitKey(0)
