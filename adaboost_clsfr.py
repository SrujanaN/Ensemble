from numpy import *
import numpy as np
import json
import sys
import DecisionTree as dt


def getArgs():
    np.random.seed(0)
    if len(sys.argv) == 5:
        T = int(sys.argv[1])
        depth = int(sys.argv[2])
        train = str(sys.argv[3])
        test = str(sys.argv[4])
    else:
        sys.exit('Illegal Arg Exception')
    return train, test, depth, T


def loadData(fileName):
    feature = []
    with open(fileName, 'r') as write_file:
        data = json.load(write_file)
        metadata = np.array(data['metadata']['features'])
        for b in data['metadata']['features']:
            feature.append(b[1])
    return np.array(data['data']), metadata, np.array(feature)


def adaboost_classifier(train_x, train_y, test_x, test_y, T, depth):
    n_train = len(train_x)
    n_test = len(test_x)
    k = len(classes)
    mytree = dt.DecisionTree()

    wts = np.zeros((n_train, T + 1))
    wts[:, 0] = 1 / n_train
    pred_test = np.zeros((n_test, T + 2)).astype(object)
    alpha = np.zeros(T)
    for i in range(T):
        mytree.fit(train_x, train_y, training_metadata, depth, wts[:, i])
        predictions_y = mytree.predict(train_x)
        err = 0

        for j in range(n_train):
            if predictions_y[j] != train_y[j]:
                err += wts[j, i]

        if err >= (1 - (1/k)):
            break

        alpha[i] = np.log((1 - err)/err) + np.log(k-1)

        for j in range(n_train):
            if predictions_y[j] != train_y[j]:
                wts[j, i+1] = wts[j, i] * np.exp(alpha[i])
            else:
                wts[j, i+1] = wts[j, i]
#         normalize
        wts[:, i+1] = [x / sum(wts[:, i+1]) for x in wts[:, i+1]]
        pred_test[:, i] = mytree.predict(test_x)

        pred_test[:, -1] = test_y

    # first column of output
    for i in range(n_train):
        print(",".join("%.12f" % x for x in wts[i, :-1]))
    print()

    # second
    print(",".join("%.12f" % t for t in alpha))
    print()

    correct = 0

    alphas = np.zeros(k)
    for i in range(n_test):
        for cls_idx, cls in enumerate(classes):
            idxs = np.argwhere(pred_test[i, :-2] == cls)
            alphas[cls_idx] = len(idxs) * alpha[idxs].sum()
        pred_test[i, -2] = classes[np.argmax(alphas)]
        if pred_test[i, -2] == pred_test[i, -1]:
            correct += 1

    for i in range(n_test):
        print(",".join(pred_test[i, :].astype(str)))
    print()

    # accuracy
    print(correct/n_test)


training_set, test_set, depth, T = getArgs()
training_data, training_metadata, feature_range = loadData(training_set)
classes = feature_range[-1]
features = training_metadata[0:-1, 0]
feature_types = training_metadata[0:-1, 1]
train_labels = np.array(training_data[:, -1])
train_X = training_data[:, :-1]
train_y = training_data[:, -1]

test_data, test_metadata, feature_range_test = loadData(test_set)
test_data = np.array(test_data)
test_X = test_data[:, :-1]
test_y = test_data[:, -1].astype(object)

adaboost_classifier(train_X, train_y, test_X, test_y, T, depth)