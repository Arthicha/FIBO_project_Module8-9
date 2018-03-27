import numpy as np
import cv2
from os import listdir
from scipy.signal import find_peaks_cwt
import os

current_dir = os.getcwd()
print(current_dir)
filelist = [x for x in listdir(current_dir + "\\Augmented_dataset\\") if x[-4:] == ".txt"]
print(filelist)

def find_chara_block(list_of_his,weight,minimum=2):
    thresh = sum(list_of_his)*weight/len(list_of_his)
    state = 0
    block = 0
    last = True
    count=1
    for x in  list_of_his:
        if x >= thresh and state == 0 and last:
            count = count+1
            if count>minimum:
                block += 1
                state=1
                count=1
        elif x >= thresh and state == 0 and not last:
            last = True
            count = 1
        elif x<thresh and state==1 and not last:
            count += 1
            if count > minimum:
                count =1
                state = 0
        elif x<thresh and state==1 and last:
            last =False
            count = 1
        # x < thresh and state == 1 and not last
        # else:
        #     if x<thresh:
        #         if last == False:
        #             count = count+1
        #         else:
        #             count = 1
        #         last = False
    return block

for file in filelist:
    f = open(current_dir + "\\Augmented_dataset\\" + file, 'r')
    data = f.read()
    f.close()
    data = data.split('\n')[:-1]
    data = list(map(lambda x: np.array(x.split(',')).reshape(-1, (60)), data))
    ''' find X and Y histogram divide by 3 
        x result in 20 value
        y result in 10 value
                                '''
    histogram_x = list(map(lambda x: [x[:,y:y+1] for y in range(0,60,1)],data))
    histogram_x = list(map(lambda x: list(map(lambda y:np.sum(1-y.astype(float)),x)),histogram_x))
    histogram_x = list(map(lambda x:np.array(x)/max(x),histogram_x))
    histogram_y = list(map(lambda x: [x[y:y+1,:] for y in range(0,30,1)],data))
    histogram_y = list(map(lambda x: list(map(lambda y:np.sum(1-y.astype(float)),x)),histogram_y))
    histogram_y = list(map(lambda x: np.array(x) / max(x), histogram_y))

    histogram_y = list(map(lambda x: find_chara_block(x,0.35,0), histogram_y))
    # ''' using convolution to smooth image'''
    # histogram_x = list(map(lambda x: np.convolve(x, np.array([ 1,1, 1, 1,1,1,1]),"same"), histogram_x))
    # ''' using scipy find peak'''
    # histogram_x = list(map(lambda x:find_peaks_cwt(x , [7]),histogram_x))#,max_distances=[4 for x in range(0,21)],gap_thresh=1
    # '''  possible number of peak in one file'''
    # histogram_x = list(map(lambda x: x.shape,histogram_x))
    for i in set(histogram_y):
        print(str(i)+"  : "+str(histogram_y.count(i)/len(histogram_y)))
    print(" space ")
    print()
    ''' concatenate histogram into one row for each pic'''
    # all_histogram=np.array(list(map(lambda x,y: np.concatenate([x,y]),histogram_x,histogram_y)))
    # all_histogram=np.around(all_histogram,4)
    ''' save to text'''
    # np.savetxt("project\\histogram_"+file,all_histogram,fmt="%1.4f",delimiter=",")
    print(file)