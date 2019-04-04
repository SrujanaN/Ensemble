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
    training_sample_indices = np.zeros((T, m))
    test_output = np.zeros((T, n_test))

    for i in range(T):
        # randomTrainData = np.random.choice(train_data[:])
        training_sample_indices[i] = np.random.choice(m,
                                                      m, replace=True)
        training_samples = train_data[
                           training_sample_indices[
                             i].astype(int), :]

        trainX = training_samples[:, :-1]
        trainy = training_samples[:, -1]
        mytree.fit(trainX, trainy, training_metadata,
                 max_depth=depth)
        predicted_y = mytree.predict(test_X, prob=False)
        total_predict_y.append(predicted_y)
        test_output[i] = predicted_y
        # print(predicted_y.shape)
    # np.append(test_output, total_predict_y)

    predictions = np.zeros(n_test)
    for i in range(n_test):
      predictions[i] = np.max(test_output[:, i])

    print(training_sample_indices.transpose().astype(int))
    print()
    out1 = np.column_stack((test_output.transpose(),
                            predictions.transpose()))
    out2 = np.column_stack((out1, test_y))
    print(out2.astype(int))

    exit(-2)
    return total_predict_y




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
test_y = test_data[:,-1]

total_prediction = learner(training_data, T, depth)
print(total_prediction)

# Predict the test labels:
# predicted_y = mytree.predict(test_X, prob=True)
# print(predicted_y.shape)
