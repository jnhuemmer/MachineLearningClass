import pandas
import matplotlib.pyplot as plt
import numpy as np
from random import random


import sklearn
from sklearn import preprocessing
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

def getSetosaAndVirginica():
    iris = datasets.load_iris()
    # retrieve the iris setosa and virginica data
    # This works because the flowers are categorized by number - 0 is setosa, 1 is versicolor, 2 is virginica
    noVersi = (iris.target != 1)

    X = iris.data[noVersi, :] # Extract data from the numpy data frame
    y = iris.target[noVersi] # Extract targets from the target vector
    y = y.reshape((len(y), 1))


    # Gets where the target is 0 or 2
    setIndex = np.where(y==0)
    virIndex = np.where(y==2)

    # Assign target either a 0 or 1
    y[setIndex] = 0. # setosa
    y[virIndex] = 1.0 # virginica

    return X, y

inputData, outputData = getSetosaAndVirginica()
outputData = np.ravel(outputData)


print(inputData)
inputData = preprocessing.normalize(inputData)
print(inputData)

clf = LogisticRegression(random_state=7)
clf.fit(inputData, outputData)

print(clf.predict(inputData))
print(outputData)
