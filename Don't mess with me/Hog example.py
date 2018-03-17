import cv2
import numpy as np
import os

def showPic(img):
    cv2.imshow("show",img)
    cv2.waitKey(0)


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*60*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (60, 30), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


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

hog = HOG_int()
hog_descriptors = []

path = 'C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\data0-9compress\\Extract\\'
dirs = os.listdir(path)

for files in dirs:
    a = files.split('_')
    d = a[len(a)-1]
    if d == 'train.txt':
        director = open('C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\data0-9compress\\Extract\\'+str(files),'r')
        data = director.read()
        director.close()
        data=data.split('\n')
        # print(len(data))
        data=data[:-1]
        num =0
        for x in data:
            lisss=x.split(',')
            img = np.array(list(lisss[:-1]))
            img = img.reshape(-1,(60))
            img = img.astype(np.uint8)*255
            num += 1
            # print(num)
            img = deskew(img)
            trest =hog.compute(img,winStride=(20,20))
            # print(trest)
            hog_descriptors.append(hog.compute(img,winStride=(10,10)))
        hog_descriptors = np.squeeze(hog_descriptors)
        print(hog_descriptors.shape)


