import cv2
from os import listdir
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split,StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix,classification_report
import graphviz
import os
import pickle
import numpy as np


'''Path Setting'''
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

Datapath = "C:\\Users\cha45\PycharmProjects\FIBO_project_Module8-9\Dataset\Tew\compress_dataset\\"

'''Model Parameter setting'''
N_ESTIMATOR = 20
Data_used = 'Project_data'  # iris for scikit given data of flower
                            # 'Project_data'to load data from text file
Data_Sep = "Test_Train_Validate"           # for all data as one set used "None"
                            # "Test_Train_Validate" for separating data according to Group 2 files Test train and validate will be collect in each parameter
MINIMUM_SAMPLES_SPLIT = 2   # The minimum number of samples required to split an internal node:
                              # used float for percentage of all sample
                            # used int for exact number of sample
# MIN_IMPURITY_DECREASE = 0        #default is 0
multi_output = False
N_SPLIT = 4
Kfold_Type = "Stratified"
TEST_AND_VALIDATE_PERCENT =0.4 # use float value from 1 -0


feature_selection = 'None'  # 'Tree_based' for tree
                                    # 'L1_based'  for lasso algorithm
Kbest= False
Kbest_feature_left = 50



'''feature_extraction_parameter_setting'''
RowOrColPerBar = 1
image_shape_x_axis = 60
image_shape_y_axis = 30
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
'''Confusion Matrix'''
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

elif Data_used == "Project_data" and Data_Sep=="Test_Train_Validate":
    hog = HOG_int()
    feature_dictionary ={"test":[],"train":[],"validate":[]}
    target_dictionary ={"test":[],"train":[],"validate":[]}
    filelist = [x for x in os.listdir(Datapath) if x[-4:] == ".txt"]
    feature = []
    target = []
    for i in filelist:
        f = open(Datapath + i, 'r')
        data = f.read()
        f.close()
        data = data.split('\n')[:-1]
        target = [i.split('_')[1] for x in range(0, len(data))]
        data = list(map(lambda x: np.array(x.split(',')).reshape(-1, (image_shape_x_axis)), data))
        '''histogram_x'''
        histogram_x = list(
            map(lambda x: [x[:, y:y + RowOrColPerBar] for y in range(0, image_shape_x_axis, RowOrColPerBar)], data))
        histogram_x = list(map(lambda x: list(map(lambda y: np.sum(y.astype(float)), x)), histogram_x))  # 1-
        ''' normalization histogram x of each image '''
        # histogram_x = list(map(lambda x: Histogram_normalize(x), histogram_x))
        '''histogram_y'''
        histogram_y = list(
            map(lambda x: [x[y:y + RowOrColPerBar, :] for y in range(0, image_shape_y_axis, RowOrColPerBar)], data))
        histogram_y = list(map(lambda x: list(map(lambda y: np.sum(y.astype(float)), x)), histogram_y))  # 1-
        ''' normalization histogram x of each image '''
        # histogram_y = list(map(lambda x: Histogram_normalize(x), histogram_y))
        ''' hog '''
        all_hog_perfile = list(map(lambda x: hog.compute(x.astype(np.uint8), winStride=(20, 20)), data))
        '''change hog feature shape form (243,1) to (243,)'''
        all_hog_perfile = list(map(lambda x: x.reshape((-1,)), all_hog_perfile))
        '''combine all feature'''
        all_feature = list(map(lambda x, y,z: np.concatenate([x, y,z]).tolist(), histogram_x, histogram_y,all_hog_perfile))
        '''collect data'''
        feature_dictionary[i[:-4].split('_')[2]]+=all_feature
        target_dictionary[i[:-4].split('_')[2]]+=target
        print(i)

elif Data_used == 'Project_data' :
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
        data = list(map(lambda x: np.array(x.split(',')).reshape(-1, (image_shape_x_axis)), data))

        '''histogram_x'''
        histogram_x = list(map(lambda x: [x[:, y:y + RowOrColPerBar] for y in range(0, image_shape_x_axis, RowOrColPerBar)], data))
        histogram_x = list(map(lambda x: list(map(lambda y: np.sum( y.astype(float)), x)), histogram_x))#1-
        ''' normalization histogram x of each image '''
        # histogram_x = list(map(lambda x: Histogram_normalize(x), histogram_x))
        '''histogram_y'''
        histogram_y = list(map(lambda x: [x[y:y + RowOrColPerBar, :] for y in range(0, image_shape_y_axis, RowOrColPerBar)], data))
        histogram_y = list(map(lambda x: list(map(lambda y: np.sum( y.astype(float)), x)), histogram_y))#1-
        ''' normalization histogram x of each image '''
        # histogram_y = list(map(lambda x: Histogram_normalize(x), histogram_y))
        # Row = list(map(lambda x: np.array(find_chara_block(x, 0.35, 0)).tolist(), histogram_y))
        ''' hog '''
        all_hog_perfile = list(map(lambda x: hog.compute(x.astype(np.uint8), winStride=(20, 20)), data))

        ''' collect_data'''
        all_histogram = list(map(lambda x, y: np.concatenate([x, y]), histogram_x, histogram_y))
        all_hog += all_hog_perfile

        all_histogram_perfile += all_histogram
        # np.savetxt("project2\\histogram_"+file,all_histogram,fmt="%1.4f",delimiter=",")
        print(i)
    '''change hog feature shape form (243,1) to (243,)'''
    all_hog = list(map(lambda x: x.reshape((-1,)), all_hog))
    '''combine all feature'''
    feature= list(
        map(lambda w, x: np.concatenate([w, x]).tolist(), all_histogram_perfile, all_hog))#, y, z

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



best_score = 0
model = RandomForestClassifier(n_estimators=N_ESTIMATOR , max_depth=None,
                            min_samples_split=MINIMUM_SAMPLES_SPLIT, random_state=0)
# if multi_output:
#     splitted= skf.split(feature, dummy_target)
# else:
if Data_used== 'Project_data' and Data_Sep == "Test_Train_Validate":
    key =[*feature_dictionary.keys()]
    for i in range(0,len(key)-1):
        train_feature = feature_dictionary[key[i]]+feature_dictionary[key[i+1]]
        train_target =  target_dictionary[key[i]]+target_dictionary[key[i+1]]
        model.fit(train_feature, train_target)
        train_score = model.score(train_feature, train_target)
        print("train score :   " + str(train_score))
        test_score = model.score(feature_dictionary[key[(i+2)%3]], target_dictionary[key[(i+2)%3]])
        print("test score :   " + str(test_score))
        Label_Pred = model.predict(feature_dictionary[key[(i+2)%3]])
        confusionMat(target_dictionary[key[(i+2)%3]], Label_Pred)
        print(classification_report( target_dictionary[key[(i+2)%3]], Label_Pred))
        if test_score > best_score:
            best_score = test_score
            s = model

else:
    if Kfold_Type == "StratifiedShuffleSplit":
        skf = StratifiedShuffleSplit(n_splits=N_SPLIT,test_size=TEST_AND_VALIDATE_PERCENT)
    else:
        skf = StratifiedKFold(n_splits=N_SPLIT)
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
        scores = cross_val_score(model, test_validate_input_data, test_validate_target_data )
        print("cross_val_score :   " + str(scores.mean()))
        Label_Pred = model.predict(test_data)
        Prob_Pred = model.predict_proba(test_data)
        # confusionMat(test_target,Label_Pred)
        print(classification_report(test_target,Label_Pred))
        print(Prob_Pred)
        if test_score > best_score:
            best_score = test_score
            s = model
        print("best_score :   " + str(best_score))
        print(model.feature_importances_)
pickle.dump(s, open("Random_Forest_best_run.sav", 'wb'))

