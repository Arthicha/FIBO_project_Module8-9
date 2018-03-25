from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

import graphviz
import os
import pickle
import numpy as np

# Model setting

'''Path Setting'''
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

Datapath = "C:\\Users\\cha45\\PycharmProjects\\FIBO_project_Module8-9\\Dataset\\Tew\\project2\\"

'''Parameter setting'''
DecidingFunction = "gini"  # gini or entropy
Data_used = 'Project_data'  # iris for scikit given data of flower
                              # 'Project_data'to load data from text file
minimum_samples_split = 0.01  # The minimum number of samples required to split an internal node:
                              # used float for percentage of all sample
                            # used int for exact number of sample
multi_output = True

feature_selection = 'Tree_based'  # 'Tree_based' for tree
# 'L1_based'  for lasso algorithm

k = 3

# Data
''' data '''
if Data_used == 'iris':
    iris = load_iris()
    feature = iris.data
    target = iris.target
    # iris_data = iris.data
elif Data_used == 'Project_data':
    filelist = [x for x in os.listdir(Datapath) if x[-4:] == ".txt"]
    feature = []
    target = []
    for i in filelist:
        f = open(Datapath + i, 'r')
        data = f.read()
        f.close()
        data = data.split('\n')[:-1]
        data = list(map(lambda x: np.array(x.split(',')).reshape(30).astype(np.float64).tolist(), data))
        feature = feature + data
        target = target + [i.split('_')[2] for x in range(0, len(data))]
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

model = tree.DecisionTreeClassifier(criterion=DecidingFunction, min_samples_split=0.01)

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
# pickle.dump(s, open("best_run.sav", 'wb'))
# dot_data = tree.export_graphviz(s, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("s")
