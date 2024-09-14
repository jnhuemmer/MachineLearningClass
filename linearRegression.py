import os
import random

print("Hello world!")

class ThetaNumerical:

    ################################################## Static Vars and Methods
    inputDataPoints = [[1, 2, 3, 4, 5, 6, 7]]
    outputDataPoints = [2, 4, 6, 8, 10, 12, 14]
    alpha = 0.001
    deltaJ = 0.00001

    # Changes alpha
    def changeAlpha(newAlpha):
        ThetaNumerical.alpha = float(newAlpha)

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

        ThetaNumerical.inputDataPoints = newInputDataPoints
        ThetaNumerical.outputDataPoints = newOutputDataPoints

    # Changes deltaJ
    def changeDeltaJ(newDeltaJ):
        ThetaNumerical.deltaJ = float(newDeltaJ)

    ################################################## Object manipulation

    # Changes value of theta
    def changeValue(self, newValue):
        self.thetaValue = newValue

    # Returns label of theta (theta0, theta1, theta2, etc.)
    def getLabel(self):
        return self.label
    
    # Returns numerical value of theta
    def getValue(self):
        return self.thetaValue

    # Using the dimensionality of the input data, this method creates the proper amount of theta variables and assigns them a random value
    def initializeThetaNumerical():
        tempList = []
        for x in range(len(ThetaNumerical.inputDataPoints) + 1):
            tempList.append(ThetaNumerical(random.randrange(1, 5, 1), x))
        return tempList

    ################################################## Mathematical Operations

    # Calculates a J value based on theta values and input/output data
    def calcJ(thetaList):
        totalSum = 0
        
        # Loop through xi/yi
        for i in range(len(ThetaNumerical.outputDataPoints)):
            tempCalc = 0
            
            # Loop through thetas (left to right within summation)
            for theta in thetaList:
                if theta.getLabel() == 0: # for theta0
                    tempCalc += theta.getValue()
                else:
                    tempCalc += theta.getValue() * ThetaNumerical.inputDataPoints[theta.getLabel() - 1][i]
            
            tempCalc -= ThetaNumerical.outputDataPoints[i] # subtract yi
            tempCalc *= tempCalc # square
            totalSum += tempCalc
        return totalSum/(2 * len(ThetaNumerical.outputDataPoints))

    # This is the main method for the mathematics section, as it calls the other ones
    # This method performs a gradient descent using an input list of initialized theta values and the input/outputdatapoints
    def gradientDescent(thetaList):
        continueCalc = True
        oldJ = None
        
        # Loops until deltaJ specification
        while continueCalc:
            derivativeList = []
            
            # Take the partial derivative of each theta value
            for theta in thetaList:
                if theta.getLabel() == 0:
                    derivativeList.append(ThetaNumerical.thetaZeroDerivative(thetaList))
                else:
                    derivativeList.append(ThetaNumerical.thetaStandardDerivative(theta, thetaList))
            
            # Apply alpha and partial derivative to current theta
            ThetaNumerical.updateThetaNumerical(thetaList, derivativeList)
            
            # Calculate and check j
            j = ThetaNumerical.calcJ(thetaList)
            if oldJ != None and abs(j - oldJ) < ThetaNumerical.deltaJ:
                continueCalc = False
            else:
                oldJ = j
        
        return thetaList

    # Takes the partial derivative of entered theta value (any except for theta0)
    def thetaStandardDerivative(inputThetaNumerical, thetaList):
        totalSum = 0
        
        # Loop through xi/yi
        for i in range(len(ThetaNumerical.outputDataPoints)):
            tempCalc = 0
            
            # Loop through thetas (left to right within summation)
            for theta in thetaList:
                if theta.getLabel() == 0:
                    tempCalc += theta.getValue()
                else:
                    tempCalc += theta.getValue() * ThetaNumerical.inputDataPoints[theta.getLabel() - 1][i]
            
            tempCalc -= ThetaNumerical.outputDataPoints[i] # Subtract yi
            tempCalc *= ThetaNumerical.inputDataPoints[inputThetaNumerical.getLabel() - 1][i] # Multiply by xi before summation
            totalSum += tempCalc
        
        return totalSum/len(ThetaNumerical.outputDataPoints)

    # Takes the partial derivative of entered theta value (any except for theta0)
    # Same as the StandardDerivative method but without multiplying by xi before summation
    def thetaZeroDerivative(thetaList):
        totalSum = 0
        
        # Loop through xi/yi
        for i in range(len(ThetaNumerical.outputDataPoints)):
            
            # Loop through thetas (left to right within summation)
            for theta in thetaList:
                if theta.getLabel() == 0:
                    totalSum += theta.getValue()
                else:
                    totalSum += theta.getValue() * ThetaNumerical.inputDataPoints[theta.getLabel() - 1][i]
            
            totalSum -= ThetaNumerical.outputDataPoints[i] # subtract yi
        
        return totalSum/len(ThetaNumerical.outputDataPoints)

    # This method is most relevant with gradient descent, updating the list of theta objects with new calculated values
    # This is done following derivative calculation as to not mess up the derivative calculations for sequential theta values
    def updateThetaNumerical(thetaList, derivativeList):
        for i in range(len(thetaList)):
            thetaList[i].changeValue(thetaList[i].getValue() - ThetaNumerical.alpha * derivativeList[i])

    ################################################## Class Methods

    def __init__(self, theta, label):
        self.thetaValue = float(theta)
        self.label = label

    def __str__(self):
        return "theta" + str(self.label) + " has a value of " + str(self.thetaValue)





# MAIN
# Numerical 
xData = [1, 2, 3, 4, 5, 6, 7]
yData = [1.5, 3.6, 6.7, 9.0, 11.2, 13.6, 16.0]

ThetaNumerical.changeDataPoints([xData], yData)

thetaList = []
thetaList = ThetaNumerical.initializeThetaNumerical()

for x in thetaList:
    print(x)

newThetaNumericalList = ThetaNumerical.gradientDescent(thetaList)

for x in newThetaNumericalList:
    print(x)

