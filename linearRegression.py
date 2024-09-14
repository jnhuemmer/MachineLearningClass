import os
import random

print("Hello world!")

class Theta:

    inputDataPoints = [[1, 2, 3, 4, 5, 6, 7]]
    outputDataPoints = [2, 4, 6, 8, 10, 12, 14]

    alpha = 0.001

    deltaJ = 0.00001

    # This method takes a list of lists. The outer list corresponds to dimensionality and represents data points.
    # Each sub list represent a point value for a single dimension
    def changeDataPoints(newInputDataPoints, newOutputDataPoints):
        if len(newInputDataPoints[0]) < 2:
            raise Exception("Data entered is not of sufficient length. Please enter a list of lists.")
        
        for x in range(len(newInputDataPoints)):
            if len(newInputDataPoints[x]) != len(newInputDataPoints[0]):
                raise Exception("Lists must have the same number of values.")
        
        if len(newInputDataPoints[0]) != len(newOutputDataPoints):
            raise Exception("Input data and output data must have the same number of data points.")

        Theta.inputDataPoints = newInputDataPoints
        Theta.outputDataPoints = newOutputDataPoints

    def getLabel(self):
        return self.label
    
    def getValue(self):
        return self.thetaValue

    def changeValue(self, newValue):
        self.thetaValue = newValue

    def gradientDescent(thetaList):
        continueCalc = True
        
        oldJ = None
        
        while continueCalc:
            derivativeList = []
            for theta in thetaList:
                if theta.getLabel() == 0:
                    derivativeList.append(Theta.thetaZeroDerivative(thetaList))
                else:
                    derivativeList.append(Theta.thetaStandardDerivative(theta, thetaList))
            
            Theta.updateTheta(thetaList, derivativeList)

            j = Theta.calcJ(thetaList)

            if oldJ != None and abs(j - oldJ) < Theta.deltaJ:
                continueCalc = False
            else:
                oldJ = j

            print(j)

        return thetaList


    def calcJ(thetaList):
        totalSum = 0
        for i in range(len(Theta.outputDataPoints)):
            tempCalc = 0
            for theta in thetaList:
                if theta.getLabel() == 0:
                    tempCalc += theta.getValue()
                else:
                    tempCalc += theta.getValue() * Theta.inputDataPoints[theta.getLabel() - 1][i]
            tempCalc -= Theta.outputDataPoints[i]
            tempCalc *= tempCalc
            totalSum += tempCalc
        return totalSum/(2 * len(Theta.outputDataPoints))



    def updateTheta(thetaList, derivativeList):
        for i in range(len(thetaList)):
            thetaList[i].changeValue(thetaList[i].getValue() - Theta.alpha * derivativeList[i])



    def thetaZeroDerivative(thetaList):
        totalSum = 0
        for i in range(len(Theta.outputDataPoints)):
            for theta in thetaList:
                if theta.getLabel() == 0:
                    totalSum += theta.getValue()
                else:
                    totalSum += theta.getValue() * Theta.inputDataPoints[theta.getLabel() - 1][i]
            totalSum -= Theta.outputDataPoints[i]
        return totalSum/len(Theta.outputDataPoints)

    def thetaStandardDerivative(inputTheta, thetaList):
        totalSum = 0
        for i in range(len(Theta.outputDataPoints)):
            tempCalc = 0
            for theta in thetaList:
                if theta.getLabel() == 0:
                    tempCalc += theta.getValue()
                else:
                    tempCalc += theta.getValue() * Theta.inputDataPoints[theta.getLabel() - 1][i]
            tempCalc -= Theta.outputDataPoints[i]
            tempCalc *= Theta.inputDataPoints[inputTheta.getLabel() - 1][i]
            totalSum += tempCalc
        return totalSum/len(Theta.outputDataPoints)

    def __init__(self, theta, label):
        self.thetaValue = float(theta)
        self.label = label

    def __str__(self):
        return "theta" + str(self.label) + " has a value of " + str(self.thetaValue)

    def initializeTheta():
        tempList = []
        
        for x in range(len(Theta.inputDataPoints) + 1):
            tempList.append(Theta(random.randrange(1, 5, 1), x))
        
        return tempList




xData = [1, 2, 3, 4, 5, 6, 7]
yData = [1.5, 3.6, 6.7, 9.0, 11.2, 13.6, 16.0]

Theta.changeDataPoints([xData], yData)
print(Theta.inputDataPoints)
print(Theta.outputDataPoints)

thetaList = []

thetaList = Theta.initializeTheta()

for x in thetaList:
    print(x)

newThetaList = Theta.gradientDescent(thetaList)

for x in newThetaList:
    print(x)

