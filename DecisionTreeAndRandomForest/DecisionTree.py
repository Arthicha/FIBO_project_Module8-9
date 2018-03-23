from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
# Model setting
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
DecidingFunction = "gini"  # gini or entropy

# Data

k = 3
iris = load_iris()
iris_data = iris.data

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
for train, test in skf.split(iris.data, iris.target):
    #    train
    train_in_set = list(map(lambda x: iris_data[x], train))
    train_target_set = list(map(lambda x: iris.target[x], train))
    test_validate_input_data = list(map(lambda x: iris.data[x], test))
    test_validate_target_data = list(map(lambda x: iris.target[x], test))
    test_data,val_data, test_target , val_target = train_test_split(test_validate_input_data, test_validate_target_data,
                                                                    test_size=0.5)
    model.fit(train_in_set,train_target_set)
    score = model.score(train_in_set, train_target_set)
    print("train score :   " + str(score))
    score=model.score(val_data,val_target)
    print("validation score :   "+str(score))
    score = model.score(test_data,test_target)
    print("test score :   " + str(score))

    dot_data =tree.export_graphviz(model,out_file=None)
    graph = graphviz.Source(dot_data)

    graph.render("model")
#    print(model.decision_path())