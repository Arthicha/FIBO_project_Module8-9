from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
# Model setting
import graphviz
import os
import pickle
import numpy as np

'''Path Setting'''
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
DecidingFunction = "gini"  # gini or entropy
Datapath = "C:\\Users\\cha45\\PycharmProjects\\FIBO_project_Module8-9\\Dataset\\Tew\\project2\\"

'''Parameter setting'''
Data_used = 'Project_data'  # iris for scikit given data of flower
                            # test data for
k = 3

# Data
if Data_used=='iris':
    iris = load_iris()
    feature = iris.data
    target = iris.target
    # iris_data = iris.data
elif Data_used=='Project_data':
    filelist=[x for x in os.listdir(Datapath) if  x[-4:] == ".txt"]
    feature = []
    target =[]
    for i in filelist:
        f = open(Datapath + i, 'r')
        data = f.read()
        f.close()
        data = data.split('\n')[:-1]
        data = list(map(lambda x: np.array(x.split(',')).reshape(30).astype(np.float64).tolist(), data))
        feature=feature+data
        target =target+ [i.split('_')[2] for x in range(0,len(data))]
else:
    raise NameError('No such data')
#    other =
#    print("**")

# X =[]      feature 1 2 3 4...
#           sample1
#           sample2

X = [[0, 1], [1, 1]]

# Y =[]             output
#           sample1
#

Y = [0, 1]

model = tree.DecisionTreeClassifier(criterion=DecidingFunction)

# train
# model = model.fit(iris.data, iris.target)

# verify

skf = StratifiedKFold(n_splits=3)
best_score = 0
for train, test in skf.split(feature, target):
    #    train
    train_in_set = list(map(lambda x: feature[x], train))
    train_target_set = list(map(lambda x: target[x], train))
    test_validate_input_data = list(map(lambda x: feature[x], test))
    test_validate_target_data = list(map(lambda x: target[x], test))
    test_data, val_data, test_target, val_target = train_test_split(test_validate_input_data, test_validate_target_data,
                                                                    test_size=0.5)
    model.fit(train_in_set, train_target_set)
    train_score = model.score(train_in_set, train_target_set)
    print("train score :   " + str(train_score))
    val_score = model.score(val_data, val_target)
    print("validation score :   " + str(val_score))
    test_score = model.score(test_data, test_target)
    print("test score :   " + str(test_score))
    if test_score > best_score:
        best_score = test_score
        pickle.dump(model, open("best_run.sav", 'wb'))
    print("best_score :   " + str(best_score))
    dot_data = tree.export_graphviz(model, out_file=None)
    graph = graphviz.Source(dot_data)

    graph.render("model")
    #    print(model.decision_path())
