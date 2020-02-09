#!/usr/bin/python3

# AUTHOR:  *your name here*
# NetID:   *your NetID here (e.g., blackboard)
# csugID:  *your csug login here (if different from NetID*

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# TODO: understand that you should not need any other imports other than those
# already in this file; if you import something that is not installed by default
# on the csug machines, your code will crash and you will lose points

# Return tuple of feature vector (x, as an array) and label (y, as a scalar).
def parse_add_bias(line):
    tokens = line.split()
    x = np.array(tokens[:-1] + [1], dtype=np.float64)
    y = np.float64(tokens[-1])
    return x,y

# Return tuple of list of xvalues and list of yvalues
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_add_bias(line) for line in f]
        (xs,ys) = ([v[0] for v in vals],[v[1] for v in vals])
        return xs, ys

# Do learning.
def perceptron(train_xs, train_ys, iterations):
    pass


# Return the accuracy over the data using current weights.
def test_accuracy(weights, test_xs, test_ys):
    pass

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--train_file', type=str, default=None, help='Training data file.')

    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.iterations: int; number of iterations through the training data.
    args.train_file: str; file name for training data.
    """
    train_xs, train_ys = parse_data(args.train_file)

    #weights = perceptron(train_xs, train_ys, args.iterations)
    #accuracy = test_accuracy(weights, train_xs, train_ys)

    svm = SVC(C = 1000000, kernel = 'linear', random_state = 0)
    svm.fit(train_xs, train_ys)
    y_predict = svm.predict(train_xs)
    accuracy = accuracy_score(train_ys, y_predict)

    print('Train accuracy: {}'.format(accuracy))
    #print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))

if __name__ == '__main__':
    main()
