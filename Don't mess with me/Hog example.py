import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix


def confusionMat(correct_Labels, Predicted_Labels):
    con_mat = confusion_matrix(correct_Labels, Predicted_Labels)
    print(con_mat)
    print(con_mat.shape)
    siz = con_mat.shape
    size = siz[0]
    total_pres = 0
    for i in range(size):
        total_pres = total_pres + (con_mat[i, i])
        print('Class accuracy '+str(i)+': '+str(con_mat[i, i] / float(np.sum(con_mat[i, :]))))
    print('total_accuracy : ' + str(total_pres/float(np.sum(con_mat))))
#correct_lables = matrix of true class of the test data
#Predicted_labels = matrix of the predicted class

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
lables = []
test_hog_descriptors = []
test_lables = []
path = 'C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\data0-9compress\\Extract\\'
dirs = os.listdir(path)

#Import test and training data
for files in dirs:
    a = files.split('_')
    d = a[len(a)-1]
    if d == 'train.txt':
        lab = a[len(a)-3]
        director = open('C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\data0-9compress\\Extract\\'+str(files),'r')
        data = director.read()
        director.close()
        data=data.split('\n')
        data=data[:-1]
        num =0
        for x in data:
            lisss=x.split(',')
            img = np.array(list(lisss[:-1]))
            img = img.reshape(-1,(60))
            img = img.astype(np.uint8)*255
            num += 1
            img = deskew(img)
            hog_descriptors.append(hog.compute(img,winStride=(20,20)))
            lables.append(lab)
        print('appended train '+str(files))
    if d == 'test.txt':
        labs = a[len(a)-3]
        director = open('C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\data0-9compress\\Extract\\'+str(files),'r')
        data = director.read()
        director.close()
        data=data.split('\n')
        data=data[:-1]
        num =0
        for x in data:
            lisss=x.split(',')
            img = np.array(list(lisss[:-1]))
            img = img.reshape(-1,(60))
            img = img.astype(np.uint8)*255
            num += 1
            img = deskew(img)
            test_hog_descriptors.append(hog.compute(img,winStride=(20,20)))
            test_lables.append(labs)
        print('appended test '+str(files))
hog_descriptors = np.squeeze(hog_descriptors)
lables = np.squeeze(lables)
test_hog_descriptors = np.squeeze(test_hog_descriptors)
test_lables = np.squeeze(test_lables)
print(hog_descriptors.shape)
print('Begining feature selection...')
#feature selection
forest = ExtraTreesClassifier()
forest.fit(hog_descriptors, lables)
modeltree = SelectFromModel(forest,prefit=True)
X_new = modeltree.transform(hog_descriptors)
test_new = modeltree.transform(test_hog_descriptors)
print('Begining Knn fitting...')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_new, lables)
joblib.dump(neigh, 'C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\Don\'t mess with me\\knn_model.pkl')
print('Model saved!')
pred = neigh.predict(test_new)
print(pred.shape)
print('Generate confusion matrix...')
confusionMat(test_lables, pred)


