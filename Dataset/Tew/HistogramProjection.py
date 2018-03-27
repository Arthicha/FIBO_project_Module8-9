import numpy as np
import cv2
from os import listdir
import os
from sklearn.preprocessing import OneHotEncoder

RowOrColPerBar = 1
area_thresh = 35

# HOG parameters
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True



current_dir = os.getcwd()
print(current_dir)
filelist = [x for x in listdir(current_dir + "\\Augmented_dataset") if x[-4:] == ".txt"]
print(filelist)

''' deskewed image function'''
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5**skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (60, 30), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

''' Hog intialization'''
def HOG_int() :
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    return hog

'''function to one hot encoding data'''
def encoder(datum):
    enc = OneHotEncoder()
    enc.fit(datum)
    a = enc.transform(datum).toarray()
    a=a.tolist()
    a = list(map(lambda x:np.array(x),a))
    return a

'''function to find number of row'''
def find_chara_block(list_of_his, weight, minimum=2):
    thresh = sum(list_of_his) * weight / len(list_of_his)
    state = 0
    block = 0
    last = True
    count = 1
    for x in list_of_his:
        if x >= thresh and state == 0 and last:
            count = count + 1
            if count > minimum:
                block += 1
                state = 1
                count = 1
        elif x >= thresh and state == 0 and not last:
            last = True
            count = 1
        elif x < thresh and state == 1 and not last:
            count += 1
            if count > minimum:
                count = 1
                state = 0
        elif x < thresh and state == 1 and last:
            last = False
            count = 1
    return block

def create_center_feature():
    pass

# print(filelist)
hog = HOG_int()
all_dataset_outer_number = []
all_label =[]
all_histogram_perfile = []
all_row = []
all_center=[]
all_hog=[]

for file in filelist:
    f = open(current_dir + "\\Augmented_dataset\\" + file, 'r')

    data = f.read()
    f.close()
    data = data.split('\n')[:-1]
    all_label+=[[file.split("_")[1]] for x in range(0,len(data))]
    data = list(map(lambda x: np.array(x.split(',')).reshape(-1, (60)), data))
    '''histogram'''
    histogram_x = list(map(lambda x: [x[:, y:y + RowOrColPerBar] for y in range(0, 60, RowOrColPerBar)], data))
    histogram_x = list(map(lambda x: list(map(lambda y: np.sum(1 - y.astype(float)), x)), histogram_x))
    histogram_x = list(map(lambda x: np.array(x) / max(x), histogram_x))
    histogram_y = list(map(lambda x: [x[y:y + RowOrColPerBar, :] for y in range(0, 30, RowOrColPerBar)], data))
    histogram_y = list(map(lambda x: list(map(lambda y: np.sum(1 - y.astype(float)), x)), histogram_y))
    histogram_y = list(map(lambda x: np.array(x) / max(x), histogram_y))
    # Row = list(map(lambda x: np.array(find_chara_block(x, 0.35, 0)).tolist(), histogram_y))
    ''' hog '''
    all_hog_perfile = list(map(lambda x:hog.compute(x.astype(np.uint8),winStride=(20,20)),data))
    '''use hist y to find row'''
    Row = list(map(lambda x: [find_chara_block(x, 0.35, 0)], histogram_y))

    data_use_to_find_letter = list(map(lambda x: (1 - x.astype(np.uint8)) * 255, data))
    '''find contour and hierachy'''
    data_use_to_find_letter = list(
        map(lambda x: cv2.findContours(x, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE), data_use_to_find_letter))

    '''filter outer most hierachy (-1) '''
    hierachy = list(map(lambda x: x[2], data_use_to_find_letter))
    suckma = []
    n = 0
    blank = 0
    for m in data_use_to_find_letter:
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

    ''' collect_data'''
    all_center+=suckma
    all_dataset_outer_number += list(map(lambda x: [len(x)], suckma))
    # suckma = list(map(lambda x:,suckma))
    all_histogram = list(map(lambda x, y: np.concatenate([x, y]), histogram_x, histogram_y))


    all_hog += all_hog_perfile
    all_row += Row
    all_histogram_perfile += all_histogram

    # np.savetxt("project2\\histogram_"+file,all_histogram,fmt="%1.4f",delimiter=",")
    print(file)


all_row = encoder(all_row)
all_dataset_outer_number = encoder(all_dataset_outer_number)
all_hog=list(map(lambda x:x.reshape((-1,)),all_hog))
all_data=list(map(lambda w,x,y,z:np.concatenate([w,x,y,z]).tolist(),all_histogram_perfile,all_hog,all_row,all_dataset_outer_number))
print()
