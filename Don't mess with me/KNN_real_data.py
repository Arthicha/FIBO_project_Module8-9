import cv2
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, train_test_split,StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report

def confusionMat(correct_Labels, Predicted_Labels):
    labels = ['0','1','2','3','4','5','6','7','8','9','zero','one','two','three','four','five','six','seven','eight','nine','ZeroTH','OneTH','TwoTH','ThreeTH','FourTH','FiveTH','SixTH','SevenTH','EightTH','NineTH']
    print(labels)
    con_mat = confusion_matrix(correct_Labels, Predicted_Labels,labels=labels)
    print(con_mat)
    print(con_mat.shape)
    siz = con_mat.shape
    size = siz[0]
    total_pres = 0
    for i in range(size):
        total_pres = total_pres + (con_mat[i, i])
        print('Class accuracy '+str(i)+': '+str(con_mat[i, i] / float(np.sum(con_mat[i, :]))))
    print('total_accuracy : ' + str(total_pres/float(np.sum(con_mat))))
    df = pd.DataFrame (con_mat)
    filepath = 'my_excel_file_GEN.xlsx'
    df.to_excel(filepath, index=False)
#correct_lables = matrix of true class of the test data
#Predicted_labels = matrix of the predicted class

def showPic(img):
    cv2.imshow("show",img)
    cv2.waitKey(0)
#chg

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

N_SPLIT = 10
Kfold_Type = "StratifiedShuffleSplit"
TEST_AND_VALIDATE_PERCENT =0.4 # use float value from 1 -0


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
val_hog_descriptors = []
val_lables = []
path = 'C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\Dataset\\Tew\\Augmented_dataset\\'
dirs = os.listdir(path)

#Import test and training data
for files in dirs:
    a = files.split('_')
    d = a[len(a)-1]
    if d == 'train.txt':
        lab = a[len(a)-2]
        director = open('C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\Dataset\\Tew\\Augmented_dataset\\'+str(files),'r')
        data = director.read()
        director.close()
        data=data.split('\n')
        data=data[:-1]
        num =0
        for x in data:
            lisss=x.split(',')
            img = np.array(list(lisss[:]))
            img = img.reshape(-1,(60))
            img = img.astype(np.uint8)*255
            num += 1
            img = deskew(img)
            hog_descriptors.append(hog.compute(img,winStride=(20,20)))
            lables.append(str(lab))
        print('appended train '+str(files))
    if d == 'test.txt':
        labs = a[len(a)-2]
        directors = open('C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\Dataset\\Tew\\Augmented_dataset\\'+str(files),'r')
        datas = directors.read()
        directors.close()
        datas=datas.split('\n')
        datas=datas[:-1]
        num =0
        for z in datas:
            lisss=z.split(',')
            imgs = np.array(list(lisss[:]))
            imgs = imgs.reshape(-1,(60))
            imgs = imgs.astype(np.uint8)*255
            num += 1
            imgs = deskew(imgs)
            test_hog_descriptors.append(hog.compute(imgs,winStride=(20,20)))
            test_lables.append(str(labs))
        print('appended test '+str(files))
    if d == 'validate.txt':
        labs = a[len(a)-2]
        directors = open('C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\Dataset\\Tew\\Augmented_dataset\\'+str(files),'r')
        datas = directors.read()
        directors.close()
        datas=datas.split('\n')
        datas=datas[:-1]
        num =0
        for z in datas:
            lisss=z.split(',')
            imgs = np.array(list(lisss[:]))
            imgs = imgs.reshape(-1,(60))
            imgs = imgs.astype(np.uint8)*255
            num += 1
            imgs = deskew(imgs)
            val_hog_descriptors.append(hog.compute(imgs,winStride=(20,20)))
            val_lables.append(str(labs))
        print('appended test '+str(files))
hog_descriptors = np.squeeze(hog_descriptors)
lables = np.squeeze(lables)

test_hog_descriptors = np.squeeze(test_hog_descriptors)
test_lables = np.squeeze(test_lables)

val_hog_descriptors = np.squeeze(val_hog_descriptors)
val_lables = np.squeeze(val_lables)

print(hog_descriptors.shape)
print(test_hog_descriptors.shape)
print(val_hog_descriptors.shape)
print('Begining feature selection...')
#feature selection
forest = ExtraTreesClassifier()
forest.fit(hog_descriptors, lables)
modeltree = SelectFromModel(forest,prefit=True)
X_new = modeltree.transform(hog_descriptors)
test_new = modeltree.transform(test_hog_descriptors)

feature = X_new
target = lables
print('Begining Knn fitting...')
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_new, lables)
joblib.dump(neigh, 'C:\\Users\\MSI-GE72MVR-7RG\\PycharmProjects\\FIBO_project_Module8-9\\Don\'t mess with me\\knn_model_gen.pkl')
print('Model saved!')
pred = neigh.predict(test_new)
print('predicted....')
print(pred.shape)
print('Generate confusion matrix...')
confusionMat(test_lables, pred)

