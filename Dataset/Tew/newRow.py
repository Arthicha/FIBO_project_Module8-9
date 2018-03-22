import numpy as np
import cv2
from os import listdir
from scipy.signal import find_peaks_cwt
import os

current_dir = os.getcwd()
print(current_dir)
filelist = [x for x in listdir(current_dir + "\\") if x[-4:] == ".txt"]
print(filelist)
for file in filelist:
    f = open(current_dir + "\\" + file, 'r')
    data = f.read()
    f.close()
    data = data.split('\n')[:-1]
    data = list(map(lambda x: np.array(x.split(',')).reshape(-1, (60)), data))
    ''' find X and Y histogram divide by 3 
        x result in 20 value
        y result in 10 value
                                '''
    histogram_x = list(map(lambda x: [x[:,y:y+3] for y in range(0,60,3)],data))
    histogram_x = list(map(lambda x: list(map(lambda y:np.sum(1-y.astype(float)),x)),histogram_x))
    histogram_x = list(map(lambda x:np.array(x)/max(x),histogram_x))
    histogram_y = list(map(lambda x: [x[y:y+3,:] for y in range(0,30,3)],data))
    histogram_y = list(map(lambda x: list(map(lambda y:np.sum(1-y.astype(float)),x)),histogram_y))
    histogram_y = list(map(lambda x: np.array(x) / max(x), histogram_y))

    ''' using convolution to smooth image'''
    histogram_x = list(map(lambda x: np.convolve(x, np.array([ 1,1, 1, 1,1]),"same"), histogram_x))
    ''' using scipy find peak'''
    histogram_x = list(map(lambda x:find_peaks_cwt(x , [3,4,5,6],max_distances=[4 for x in range(0,21)],gap_thresh=1),histogram_x))
    '''  possible number of peak in one file'''
    histogram_x = list(map(lambda x: x.shape,histogram_x))
    print(set(histogram_x))
    ''' concatenate histogram into one row for each pic'''
    # all_histogram=np.array(list(map(lambda x,y: np.concatenate([x,y]),histogram_x,histogram_y)))
    # all_histogram=np.around(all_histogram,4)
    ''' save to text'''
    # np.savetxt("project\\histogram_"+file,all_histogram,fmt="%1.4f",delimiter=",")
    print(file)