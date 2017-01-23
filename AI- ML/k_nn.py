import csv
import random
import math
import operator
import numpy as np
import scipy.stats as stats
import sys

def loadData(train, test):
    trainingSet = []
    testSet = []
    testNames = []
    with open(train, 'rb') as f:
        lines = csv.reader(f, delimiter=" ")
        dataset = list(lines)
        for x in range(len(dataset)):
            line = []
            for y in range(len(dataset[x])):
                if y == 0 or y == 1:
                    continue
                else:
                    line.append(float(dataset[x][y]))
            line.append(float(dataset[x][1]))
            trainingSet.append(line)
    with open(test, 'rb') as f:
        lines = csv.reader(f, delimiter=" ")
        dataset = list(lines)
        for x in range(len(dataset)):
            line = []
            for y in range(len(dataset[x])):
                if y == 0:
                    testNames.append(dataset[x][y])
                if y == 0 or y == 1:
                    continue
                else:
                    line.append(float(dataset[x][y]))
            line.append(float(dataset[x][1]))
            testSet.append(line)

    return (np.asmatrix(trainingSet), np.asmatrix(testSet), testNames)

def getNeighbors(X, y, k):
    dist = np.linalg.norm(X[:,:-1] - y[:,:-1], axis=1)  # L2 norm of every row. Equivalent to euclidean distance, but much faster
    distances = []
    for i in range(len(X)):
        distances.append((X[i, -1], dist[i]))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def run(train, test):
    trainingSet, testSet, testNames = loadData(train, test)
    predictions=[]
    k = 9
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = stats.mode(neighbors)[0][0]
        predictions.append(result)
    str = ''
    for i in range(len(predictions)):
        p = predictions[i]
        n = testNames[i]
        str += n + ' ' + repr(p) + '\n'

    f = open('nearest_output.txt', 'w')
    f.write(str)
    f.close()


    return predictions
