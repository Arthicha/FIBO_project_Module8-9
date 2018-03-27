import cv2
from os import listdir
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import graphviz
import os
import pickle
import numpy as np

# Model setting

'''Path Setting'''
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

Datapath = "C:\\Users\cha45\PycharmProjects\FIBO_project_Module8-9\Dataset\Tew\Augmented_dataset\\"

'''Model Parameter setting'''
DecidingFunction = "gini"  # gini or entropy
Data_used = 'Project_data'  # iris for scikit given data of flower
                              # 'Project_data'to load data from text file
MINIMUM_SAMPLES_SPLIT = 0.01  # The minimum number of samples required to split an internal node:
                              # used float for percentage of all sample
                            # used int for exact number of sample
MIN_IMPURITY_DECREASE = 0        #default is 0
multi_output = False

feature_selection = 'None'  # 'Tree_based' for tree
                                    # 'L1_based'  for lasso algorithm
Kbest= False
Kbest_feature_left = 50
k = 3
'''feature_extraction_parameter_setting'''
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


'''Function'''
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

'''function to normalize histogram'''
def Histogram_normalize(histogram):
    if max(histogram)<=0:
        return np.array(histogram)
    else:
        return np.array(histogram)/max(histogram)

def create_center_feature():
    pass


# Data

''' data '''
if Data_used == 'iris':
    iris = load_iris()
    feature = iris.data
    target = iris.target
    # iris_data = iris.data
elif Data_used == 'Project_data':
    hog = HOG_int()
    all_dataset_outer_number = []
    all_label = []
    all_histogram_perfile = []
    all_row = []
    all_center = []
    all_hog = []
    filelist = [x for x in os.listdir(Datapath) if x[-4:] == ".txt"]
    feature = []
    target = []
    for i in filelist:
        f = open(Datapath + i, 'r')
        data = f.read()
        f.close()
        data = data.split('\n')[:-1]
        target = target + [i.split('_')[1] for x in range(0, len(data))]
        data = list(map(lambda x: np.array(x.split(',')).reshape(-1, (60)), data))
        '''histogram'''
        histogram_x = list(map(lambda x: [x[:, y:y + RowOrColPerBar] for y in range(0, 60, RowOrColPerBar)], data))
        histogram_x = list(map(lambda x: list(map(lambda y: np.sum(1 - y.astype(float)), x)), histogram_x))
        histogram_x = list(map(lambda x: Histogram_normalize(x), histogram_x))
        histogram_y = list(map(lambda x: [x[y:y + RowOrColPerBar, :] for y in range(0, 30, RowOrColPerBar)], data))
        histogram_y = list(map(lambda x: list(map(lambda y: np.sum(1 - y.astype(float)), x)), histogram_y))
        histogram_y = list(map(lambda x: Histogram_normalize(x), histogram_y))
        # Row = list(map(lambda x: np.array(find_chara_block(x, 0.35, 0)).tolist(), histogram_y))
        ''' hog '''
        all_hog_perfile = list(map(lambda x: hog.compute(x.astype(np.uint8), winStride=(20, 20)), data))
        '''use hist y to find row'''
        Row = list(map(lambda x: [find_chara_block(x, 0.35, 0)], histogram_y))

        data_use_to_find_letter = list(map(lambda x: (1 - x.astype(np.uint8)) * 255, data))
        '''find contour and hierachy'''
        data_use_to_find_letter = list(
            map(lambda x: cv2.findContours(x, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE),
                data_use_to_find_letter))

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
        all_center += suckma
        all_dataset_outer_number += list(map(lambda x: [len(x)], suckma))
        # suckma = list(map(lambda x:,suckma))
        all_histogram = list(map(lambda x, y: np.concatenate([x, y]), histogram_x, histogram_y))

        all_hog += all_hog_perfile
        all_row += Row
        all_histogram_perfile += all_histogram

        # np.savetxt("project2\\histogram_"+file,all_histogram,fmt="%1.4f",delimiter=",")
        print(i)

    all_row = encoder(all_row)
    all_dataset_outer_number = encoder(all_dataset_outer_number)
    all_hog = list(map(lambda x: x.reshape((-1,)), all_hog))
    feature= list(
        map(lambda w, x, y, z: np.concatenate([w, x, y, z]).tolist(), all_histogram_perfile, all_hog, all_row,
            all_dataset_outer_number))
        # f = open(Datapath + i, 'r')
        # data = f.read()
        # f.close()
        # data = data.split('\n')[:-1]
        # data = list(map(lambda x: np.array(x.split(',')).reshape(90).astype(np.float64).tolist(), data))
        # feature = feature + data
        # target = target + [i.split('_')[2] for x in range(0, len(data))]
else:
    raise NameError('No such data')

''' feature Selection'''
if feature_selection == 'Tree_based':
    clf = ExtraTreesClassifier()
    clf = clf.fit(feature, target)
    print(clf.feature_importances_)
    print(len(clf.feature_importances_))

    feature_selected = SelectFromModel(clf, prefit=True)
    feature = feature_selected.transform(feature)
    print(feature.shape)
elif feature_selection == 'L1_based':
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(feature, target)
    model = SelectFromModel(lsvc, prefit=True)
    feature = model.transform(feature)
    print(feature.shape)
else:
    pass

if Kbest:
    feature = SelectKBest(chi2, k=Kbest_feature_left).fit_transform(feature, target)
    print(feature.shape)

if multi_output:
    encoder = set(list(map(lambda x:x.split('_')[2],filelist)))
    encoder = list(sorted(encoder))
    diction={}
    for i in encoder:
        diction[i]=np.array([x== encoder.index(i) for x in range(0,len(encoder)) ]).astype(np.int).tolist()
    encoder = diction
    target = list(map(lambda x:encoder[x],target))
    dummy_target = list(map(lambda x:str(x),target))


# if multi_output:
#     target = list(map(lambda x : encoding[x],target))
# other =
#    print("**")

# X =[]      feature 1 2 3 4...
#           sample1
#           sample2

X = [[0, 1], [1, 1]]

# Y =[]             output
#           sample1
#

Y = [0, 1]

model = tree.DecisionTreeClassifier(criterion=DecidingFunction, min_samples_split=MINIMUM_SAMPLES_SPLIT,min_impurity_decrease=MIN_IMPURITY_DECREASE)

# train
# model = model.fit(iris.data, iris.target)

# verify

skf = StratifiedKFold(n_splits=3)
best_score = 0
if multi_output:
    splitted= skf.split(feature, dummy_target)
else:
    splitted = skf.split(feature, target)

for train, test in splitted:
    #    train
    train_in_set = list(map(lambda x: feature[x], train))
    train_target_set = list(map(lambda x: target[x], train))
    test_validate_input_data = list(map(lambda x: feature[x], test))
    test_validate_target_data = list(map(lambda x: target[x], test))
    test_data, val_data, test_target, val_target = train_test_split(test_validate_input_data, test_validate_target_data,
                                                                    test_size=0.5)
    model.fit(train_in_set, train_target_set)
    if multi_output:
        for j in range(len(test_data)):
            print(model.predict([test_data[j]]))
    train_score = model.score(train_in_set, train_target_set)
    print("train score :   " + str(train_score))
    val_score = model.score(val_data, val_target)
    print("validation score :   " + str(val_score))
    test_score = model.score(test_data, test_target)
    print("test score :   " + str(test_score))
    if test_score > best_score:
        best_score = test_score
        s = model
    print("best_score :   " + str(best_score))
    print(model.feature_importances_)


    #    print(model.decision_path())
pickle.dump(s, open("best_run.sav", 'wb'))
dot_data = tree.export_graphviz(s, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("s")
