import matplotlib.pyplot as plt
import numpy as np
from random import random

# sklearn only used for the iris data set
import sklearn
from sklearn import datasets

class ArtificialNeuralNetwork:
    
    def __init__(self, inputTrainingData, targetTrainingData, hiddenLayer, alpha=0.1):
        self.inputTrainingData = inputTrainingData
        self.targetTrainingData = targetTrainingData
        
        # Maybe integrate more dynamically?
        self.layer1Weights = ArtificialNeuralNetwork.initializeWeights(len(inputTrainingData[0]), hiddenLayer)
        self.layer2Weights = ArtificialNeuralNetwork.initializeWeights(hiddenLayer, len(targetTrainingData[0]))

        self.layer1Bias = ArtificialNeuralNetwork.initializeBias(1, hiddenLayer)
        self.layer2Bias = ArtificialNeuralNetwork.initializeBias(1, len(targetTrainingData[0]))

        self.alpha = alpha

    def backPropagation(self, inputData, targetData, layer1A, layer2A):
        
        # Back propagation
        errorLayer2 = targetData - layer2A
        deltaLayer2 = errorLayer2 * sigmoidDerivative(layer2A)

        errorLayer1 = deltaLayer2 @ self.layer2Weights.T
        deltaLayer1 = errorLayer1 * sigmoidDerivative(layer1A)

        # Gradient descent
        self.layer2Bias += (np.sum(deltaLayer2, axis=0)) * self.alpha
        self.layer1Bias += (np.sum(deltaLayer1, axis=0)) * self.alpha

        self.layer2Weights += (layer1A.T @ deltaLayer2) * self.alpha
        self.layer1Weights += (inputData.T @ deltaLayer1) * self.alpha

    # Random sample from training/target data
    def batchData(self, batchSize):
        range = np.arange(len(self.inputTrainingData))
        np.random.shuffle(range)
        batchRange = range[0:batchSize]
        inputBatch = np.take(self.inputTrainingData, batchRange, axis=0)
        targetBatch = np.take(self.targetTrainingData, batchRange, axis=0)

        return inputBatch, targetBatch

    # Cross-entropy loss function
    def costFunction(self, prediction, target):
        loss = 0
        for i in range(len(prediction)):
            loss = loss + (-1 * target[i]*np.log(prediction[i] + 1e-10)) # + 1e-10 to avoid division by log(0)
        return loss
    
    # Initialize biases as 0
    def initializeBias(inputParameters, outputNodes):
        return np.zeros((inputParameters, outputNodes), float)
    
    # Initialize weighs as random value between 0 and 1
    def initializeWeights(inputParameters, outputNodes):
        return np.random.rand(inputParameters, outputNodes)

    def forwardPropagation(self, inputData):
        layer1Z = inputData @ self.layer1Weights + self.layer1Bias
        layer1A = sigmoid(layer1Z)

        layer2Z = layer1A @ self.layer2Weights + self.layer2Bias
        layer2A = sigmoid(layer2Z)

        return layer1A, layer2A
    
    # Same as forward propagation but only returns final layer and changes values to 1 and 0
    def predict(self, inputData):
        layer1Z = inputData @ self.layer1Weights + self.layer1Bias
        layer1A = sigmoid(layer1Z)

        layer2Z = layer1A @ self.layer2Weights + self.layer2Bias
        layer2A = sigmoid(layer2Z)

        return fixValues(layer2A)
    
    # Network will train until the change in cost is less than 0.0001
    def trainNetwork(self):
        go = True
        oldCost = 1000000000
        counter = 0
        batchSize = 25
        
        print("Training started... ")
        while go == True:
            
            # Batch data
            inputBatch, targetBatch = self.batchData(batchSize)
            
            # Forward pass
            layer1A, layer2A = self.forwardPropagation(inputBatch)
            cost = self.costFunction(layer2A, targetBatch) # Use this to update layer 2 weights. Update bias with average
            
            # Check if backward pass is necessary
            if abs(cost - oldCost) <= 0.0001 and counter > 150: # 
                go = False
            else:
                self.backPropagation(inputBatch, targetBatch, layer1A, layer2A)
                oldCost = cost
                
                # Feedback every 10 iterations
                if counter % 10 == 0:
                    print("Current iteration: " + str(counter))
                    print("Current cost: " + str(cost))
                
                counter += 1
        
        print("Training complete!")
        print("Total iterations: " + str(counter))
        print("Final cost " + str(cost))

# Values above 0.5 are set to 1, and otherwise set to 0
def fixValues(inputData):
    return ((inputData > 0.5) / inputData).astype("int")

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

def reLu(inputData):
    return inputData * (inputData > 0)

def reLuDerivative(inputData):
    return (inputData > 0) * 1

def sigmoid(inputData):
    return 1/(1+np.exp(-inputData))

def sigmoidDerivative(inputData):
    return inputData * (1 - inputData)

def softMax(inputData):
    return np.exp(inputData) / np.sum(np.exp(inputData))

def normalize(inputData):
    return (inputData-np.amin(inputData))/(np.amax(inputData)-np.amin(inputData)) 

# Randomly select data points to create a training and test data set
def trainTestSplit(inputData, inputTarget, ratTrain):
    
    range = np.arange(len(inputData))
    np.random.shuffle(range)
    
    trainIndex = range[0:int((len(range) * ratTrain))]
    testIndex = range[int((len(range) * ratTrain)):len(range)]

    trainInputData = np.take(inputData, trainIndex, axis=0)
    testInputData = np.take(inputData, testIndex, axis=0)
    
    trainTargetData = np.take(inputTarget, trainIndex, axis=0)
    testTargetData = np.take(inputTarget, testIndex, axis=0)

    return trainInputData, testInputData, trainTargetData, testTargetData

# MAIN
inputData, outputData = getSetosaAndVirginica()
inputNorm = normalize(inputData)
xTrain, xTest, yTrain, yTest = trainTestSplit(inputNorm, outputData, 0.8)

# Input/Output layer size based on length of training/test data entries
irisModel = ArtificialNeuralNetwork(xTrain, yTrain, hiddenLayer=6)
irisModel.trainNetwork()

# TESTING
print("PREDICTION")
prediction = irisModel.predict(xTest)
error = np.concatenate((prediction, yTest), axis = 1)

# 1s and 0s should match exactly
print(error)
