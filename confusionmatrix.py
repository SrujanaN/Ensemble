from numpy import *
import numpy as np
import json
import sys
import DecisionTree as dt


def getArgs():
    np.random.seed(0)
    if len(sys.argv) == 6:
        clsfr = str(sys.argv[1])
        T = int(sys.argv[2])
        depth = int(sys.argv[3])
        train = str(sys.argv[4])
        test = str(sys.argv[5])
    else:
        sys.exit('Illegal Arg Exception')
    return train, test, depth, T, clsfr


def loadData(fileName):
    feature = []
    with open(fileName, 'r') as write_file:
        data = json.load(write_file)
        metadata = np.array(data['metadata']['features'])
        for b in data['metadata']['features']:
            feature.append(b[1])
    return np.array(data['data']), metadata, np.array(feature)


def learner(train_data, T, depth):
    mytree = dt.DecisionTree()
    total_predict_y = []
    m = len(train_data)
    n_test = len(test_data)
    training_sample_indices = np.zeros((T, m)).astype(int)

    test_output = np.zeros((T, n_test, len(classes)))

    for i in range(T):
        training_sample_indices[i] = np.random.choice(m, m, replace=True)
        training_samples = train_data[training_sample_indices[i], :]

        trainX = training_samples[:, :-1]
        trainy = training_samples[:, -1]
        mytree.fit(trainX, trainy, training_metadata, max_depth=depth)
        predicted_y = mytree.predict(test_X, prob=True)
        test_output[i] = predicted_y

    avg_prob = np.average(test_output, axis=0)
    actual = test_y
    predictions = np.zeros(n_test).astype(object)

    for i in range(len(test_data)):
        for j in range(T):
            tree_pred_idx = np.argmax(test_output[j, i, :])
            tree_pred = classes[tree_pred_idx]

        pred_idx = np.argmax(avg_prob[i, :])
        predictions[i] = classes[pred_idx]
    return actual, predictions


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
        # compute weights
        for j in range(n_train):
            if predictions_y[j] != train_y[j]:
                err += wts[j, i]
        # break the loop based on error criteria
        if err >= (1 - (1/k)):
            break

        alpha[i] = np.log((1 - err)/err) + np.log(k-1)

        for j in range(n_train):
            if predictions_y[j] != train_y[j]:
                wts[j, i+1] = wts[j, i] * np.exp(alpha[i])
            else:
                wts[j, i+1] = wts[j, i]
        # normalize
        wts[:, i+1] = [x / sum(wts[:, i+1]) for x in wts[:, i+1]]
        pred_test[:, i] = mytree.predict(test_x)

        pred_test[:, -1] = test_y


    alphas = np.zeros(k)
    for i in range(n_test):
        for cls_idx, cls in enumerate(classes):
            idxs = np.argwhere(pred_test[i, :-2] == cls)
            alphas[cls_idx] = len(idxs) * alpha[idxs].sum()
        pred_test[i, -2] = classes[np.argmax(alphas)]

    actual = pred_test[:, -1]
    predictions = pred_test[:, -2]

    return actual, predictions


training_set, test_set, depth, T, clsfr = getArgs()
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

k = len(classes)
confusion_matrix = np.zeros((k,k)).astype(int)
actual = []
predicted = []
if clsfr == "bag":
    actual, predicted = learner(training_data, T, depth)
else:
    actual, predicted = adaboost_classifier(train_X, train_y, test_X, test_y, T, depth)

# Populate the confusion matrix
cls_index = dict((y, x) for x, y in enumerate(classes))
for pred_val, actual_val in zip(predicted, actual):
    pred_idx = cls_index.get(pred_val)
    actual_idx = cls_index.get(actual_val)
    confusion_matrix[pred_idx][actual_idx] += 1

    # Print the confusion matrix
for cls_idx in range(k):
    print(','.join(confusion_matrix[cls_idx, :].astype(str)))
