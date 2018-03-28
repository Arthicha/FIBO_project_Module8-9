import numpy as np
import cv2
from os import listdir
import os

area_thresh = 35
current_dir = os.getcwd()
print(current_dir)
filelist = [x for x in listdir(current_dir + "\\compress_dataset\\") if x[-4:] == ".txt"]
# print(filelist)
dataset_outer = []
for file in filelist:
    f = open(current_dir + "\\compress_dataset\\" + file, 'r')
    data = f.read()
    # print(len(data))
    f.close()
    data = data.split('\n')[:-1]
    print(len(data))
    data = list(map(lambda x:  255-np.array(x.split(',')).reshape(-1, (60)).astype(np.uint8) * 255, data))

    for_render = data
    '''pre process image with erode and dilate'''
    # data = list(map(lambda x: cv2.medianBlur(x, 11), data))
    # data = list(map(lambda x: cv2.dilate(x, np.ones([2, 2])), data))
    # a
    # data = list(map(lambda x: cv2.erode(x, np.ones([1, 4])), data))
    if "EightTH" in file:
        # for m in for_render:
        #     cv2.imshow("lol",m)
        #     cv2.waitKey(100)
        print()
        pass
    # for i in data:
    #         img = cv2.cvtColor(255-i,cv2.COLOR_GRAY2BGR)
    #         org,cnt,hier=cv2.findContours(i, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    #         hier[0][:][3]
    #         print(len(cnt))
    #         cv2.drawContours(img,cnt,-1,[0,0,255])
    #         cv2.imshow('test',img)
    #         cv2.waitKey(2)
    # print(file)
    # print("************************")
    '''find contour and hierachy'''
    data = list(map(lambda x: cv2.findContours(x, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE), data))

    '''filter outer most hierachy (-1) '''
    hierachy = list(map(lambda x: x[2], data))
    suckma = []
    n = 0
    blank = 0
    for m in data:
        n += 1
        try:
            hier = m[2][0].tolist()
            con = []
            for j in hier:
                if j[3] == -1:
                    rect = cv2.minAreaRect(m[1][hier.index(j)])
                    M = cv2.moments(m[1][hier.index(j)])
                    data_to_append = []
                    try:
                        cx = int(M['m10'] / M['m00'])
                        if rect[0][0] > cx:
                            data_to_append.append(1)
                        elif rect[0][0] < cx:
                            data_to_append.append(-1)
                        else:
                            data_to_append.append(0)
                    except:
                        data_to_append.append(0)
                    try:
                        cy = int(M['m01'] / M['m00'])
                        if rect[0][1] > cy:
                            data_to_append.append(1)
                        elif rect[0][1] < cy:
                            data_to_append.append(-1)
                        else:
                            data_to_append.append(0)
                    except:
                        data_to_append.append(0)
                    if rect[1][0] * rect[1][1] > area_thresh:
                        con.append(data_to_append)  # m[1][hier.index(j)]
                        # con = [len(con)]+con+[[0,0] for sc in range(len(con),6)]
        except:
            blank += 1
            print(blank)
            con = []
        suckma.append(con)

    # outermost_contour = list(map(lambda x:[x[1],list(map(lambda z: (x[2][0].tolist()).index(z),list(filter(lambda y: y[3] == -1,x[2][0].tolist()))))] ,data))
    # # data = list(map(lambda  x : [x[1],list(map(lambda z: ((x[2]).tolist()).index(z.tolist()),list(filter(lambda y: y[0,3] == -1,x[2]))))],data))
    # '''filter innermost contour'''
    # innermost_contour = list(map(lambda x: [x[1], list(
    #     map(lambda z: (x[2][0].tolist()).index(z), list(filter(lambda y: y[2] == -1, x[2][0].tolist()))))], data))
    #
    # '''list of contour (character) left '''
    # dataset_outer= list(map(lambda x: [x[0][y] for y in x[1] ],outermost_contour))
    # dataset_outer_ =[]
    # for i in dataset_outer:
    #     cnt=[]
    #     for k in i:
    #         rect = cv2.minAreaRect(k)
    #         area = rect[1][0]*rect[1][1]
    #         if area> area_thresh:
    #             cnt.append(k)
    #     dataset_outer_.append(cnt)
    # dataset_outer=dataset_outer_
    # dataset_inner =  list(map(lambda x: [x[0][y] for y in x[1] ],innermost_contour))
    # ''' find possible number of contour (charaacter) in 1 file'''

    dataset_outer += list(map(lambda x: [len(x)], suckma))
    # dataset_inner=list(map(lambda y:len(y),dataset_inner))
    # dataset_all =list(map(lambda z:len(z[1]),data))
    lensin =list(map(lambda x: len(x), suckma))
    possibility_x = set(list(map(lambda x: len(x), suckma)))
    # possibility_y= set(dataset_inner)
    # possibility_z= set(dataset_all)
    for i in possibility_x:
        print(str(i) + "  : " + str(lensin.count(i) / len(suckma)))
    print(" space ")
    # for i in possibility_y:
    #     print(str(i)+"  : "+str(dataset_inner.count(i)/len(dataset_outer)))
    # print(" space ")
    # for i in possibility_z:
    #     print(str(i)+"  : "+str(dataset_all.count(i)/len(dataset_outer)))
    # print(" space ")
    # for i in possibility:
    #     percentage_in_file = 0
    #     ''''''
    #     print(percentage_in_file)
    # print(possibility_x)
    # print(possibility_y)
    # print(possibility_z)
    print(file)
    print('********************')
