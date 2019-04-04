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
    feature= []
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

    # output indexes
    for train_sample_row in range(len(training_samples)):
        print(','.join(training_sample_indices[:, train_sample_row].astype(str)))

    print()
    avg_prob = np.average(test_output, axis=0)

    predictions = np.zeros(n_test).astype(object)

    for i in range(len(test_data)):
        for j in range(T):
            tree_pred_idx = np.argmax(test_output[j, i, :])
            tree_pred = classes[tree_pred_idx]
            print(tree_pred, end=',')

        pred_idx = np.argmax(avg_prob[i,:])
        predictions[i] = classes[pred_idx]
        print("{0},{1}".format(predictions[i], test_y[i]))

    print()
    print((predictions == test_y).sum() / n_test)
    return


training_set, test_set, depth, T = getArgs()
training_data, training_metadata, feature_range = loadData(training_set)
classes = feature_range[-1]
features = training_metadata[0:-1, 0]
feature_types = training_metadata[0:-1, 1]
train_labels = np.array(training_data[:, -1])
train_X = training_data[:,:-1]
train_y = training_data[:,-1]

# Build and train a decision tree:
# mytree = dt.DecisionTree()
# mytree.fit(train_X, train_y, training_metadata,
#            max_depth=depth)
# look at the structure of the trained tree:
# print(mytree)

test_data, test_metadata, feature_range_test = loadData(test_set)
test_data = np.array(test_data)
test_X = test_data[:,:-1]
test_y = test_data[:,-1].astype(object)

learner(training_data, T, depth)

# Predict the test labels:
# predicted_y = mytree.predict(test_X, prob=True)
# print(predicted_y.shape)
