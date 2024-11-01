import random
import math

print("Hello world!")

# Class handles static data set and creates a list of theta objects corresponding to it
# This class is capable of performing a multiple linear regression
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
            thetaList = ThetaNumerical.updateThetaNumerical(thetaList, derivativeList)

            # Calculate and check j
            j = ThetaNumerical.calcJ(thetaList)
            if oldJ != None and abs(j - oldJ) < ThetaNumerical.deltaJ:
                continueCalc = False
            else:
                oldJ = j
        
        return thetaList

    # Takes a list of theta and calculates corresponding theoretical value with input data
    def runFunction(thetaList):
        
        # Ensures that the the number of thetas is supported by the dataset
        if len(thetaList) != len(ThetaNumerical.inputDataPoints) + 1:
            raise Exception("Theta list is not of sufficient length. Please ensure the the number of theta entries is the same as the dimensionality of the input data")
        
        # Isolate theta0 and remove it from the main list
        thetaZero = thetaList[0]
        thetaList.pop(0)

        # Calculation for a given point
        for dataPoint in range(len(ThetaNumerical.inputDataPoints[0])):
            theoreticalValue = thetaZero.getValue()
            for dimension in range(len(ThetaNumerical.inputDataPoints)):
                theoreticalValue += ThetaNumerical.inputDataPoints[dimension][dataPoint] * (thetaList[dimension]).getValue()
            
            print("at data point " + str(dataPoint) + " the theoretical value is " + str(theoreticalValue))

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
        
        return (totalSum)/len(ThetaNumerical.outputDataPoints)

    # This method is most relevant with gradient descent, updating the list of theta objects with new calculated values
    # This is done following derivative calculation as to not mess up the derivative calculations for sequential theta values
    def updateThetaNumerical(thetaList, derivativeList):
        for i in range(len(thetaList)):
            thetaList[i].changeValue(thetaList[i].getValue() - (ThetaNumerical.alpha * derivativeList[i]))
        return thetaList

    ################################################## Class Methods
    def __init__(self, theta, label):
        self.thetaValue = float(theta)
        self.label = label

    def __str__(self):
        return "theta" + str(self.label) + " has a value of " + str(self.thetaValue)


# Class treats data sets as an object and calculated beta0 and beta1 based on entered x/y data
# This class is capable of performing a simple linear regression
class SimpleLinearAnalytical:

    ################################################## Mathematical Operations

    # Finds the average of a list of numbers
    def average(numbers):
        return sum(numbers)/len(numbers)
    
    # Determines beta1 using x and y data
    def calcBetaOne(self):
        sumNum = 0
        sumDen = 0
        
        # Summation(xi - xAverage)(yi - yAverage)
        for i in range(len(self.xData)):
            sumNum += ((self.xData[i] - self.xMean)*(self.yData[i] - self.yMean))

        # Summation(xi - xAverage)^2
        for i in range(len(self.xData)):
            sumDen += ((self.xData[i] - self.xMean) ** 2)

        return sumNum/sumDen
    
    # Uses calculated y-intercept and slope to determine y-values 
    def runFunction(self, xValues):
        for x in xValues:
            print(self.betaZero + (self.betaOne * x))

    ################################################## Class Methods
    
    # Each value for the simple linear regression is calculated upon initialization
    def __init__(self, xData, yData):
        
        if len(xData) != len(yData):
            raise Exception("X and Y data must have an equal number of data points")
        
        if type(xData) is not list or type(yData) is not list:
            raise TypeError("Data must be entered in list format")
        
        self.xData = xData
        self.yData = yData
        
        self.xMean = SimpleLinearAnalytical.average(xData)
        self.yMean = SimpleLinearAnalytical.average(yData)

        self.betaOne = self.calcBetaOne()
        self.betaZero = self.yMean - (self.betaOne * self.xMean)

    def __str__(self):
        return "Beta1 and Beta0 were determined resulting in the following equation: y = " + str(self.betaZero) + " + " + str(self.betaOne) + "x"





# MAIN

def average(numbers):
    return (sum(numbers)/len(numbers))

# Set list of numbers between -1 and 1
def fixRange(numbers):
    tempList = []
    for x in numbers:
        tempList.append((2 * ((x - min(numbers))/(max(numbers)-min(numbers)))) - 1)
    return tempList

def max(numbers):
    temp = numbers [0]
    for x in numbers:
        if x > temp:
            temp = x
    return temp

def min(numbers):
    temp = numbers [0]
    for x in numbers:
        if x < temp:
            temp = x
    return temp

# For reading in CSV tables
def parseCSV(filePath):
    listOfLists = []
    with open(filePath, "r") as fileContent:
        
        # Read line by line
        for line in fileContent:
            line = line.strip()
            splitLine = line.split(",")
            
            # Breaks line into entries
            for value in range(len(splitLine)):
                
                # If block breaks columns into lists and places those lists into one big list
                if len(listOfLists) != len(splitLine):
                    if splitLine[value] != "":
                        listOfLists.append([splitLine[value]])
                    else:
                        listOfLists.append(["Unlabeled"])
                else:
                    listOfLists[value].append(float(splitLine[value]))
    
    return listOfLists

def standardDeviation(numbers):
        totalSum = 0
        
        # Loop through summation
        for x in numbers:
            totalSum += (x - average(numbers)) ** 2
        
        totalSum = math.sqrt(totalSum/(len(numbers) - 1))
        return totalSum

def zScore(numbers):
    tempData = []
    for x in numbers:
        tempData.append((x - average(numbers))/standardDeviation(numbers))
    return tempData




# MAIN METHOD

# Test data
# xData = [1, 2, 3, 4, 5, 6, 7]
# yData = [1.5, 3.6, 6.7, 9.0, 11.2, 13.6, 16.0]
# xData = [ -1.91912641, -1.715855767, -1.651482801, -0.466233925, -0.305380803, -0.249651155, 0.115579679, 0.179532732, 0.195254016, 0.272178244, 0.411053504, 0.583576452, 0.860757465, 1.112627004, 1.166900152, 1.330479382, 1.480048593, 1.567092003, 1.702386553, 1.854651042]
# yData = [-3.091213284, -3.534290666, -3.146431752, -1.359515719, -1.887256513, -0.172493012, 0.663377457, -0.012017046, 1.525385343, -0.182826349, 0.844986267, 1.07356098, 2.487904538, 2.959933393, 2.411274018, 2.850040024, 2.516204312, 2.143785772, 3.230817032, 3.787476569]

# Pretreatment
filePath = "C:\\Users\\Jacob\\Desktop\\Wormhole\\Fall2024\\MachineLearning\\HW1\\linear_regression_test_data.csv"
allData = parseCSV(filePath)

xData = allData[1][1:] # First entry is removed because it is the data category/header
yData = allData[2][1:] # First entry is removed because it is the data category/header

xData = fixRange(xData)
xData = zScore(xData)


# Numerical solution
ThetaNumerical.changeDataPoints([xData], yData)
thetaList = ThetaNumerical.initializeThetaNumerical()
newThetaNumericalList = ThetaNumerical.gradientDescent(thetaList)
for x in newThetaNumericalList:
    print(x)

print("Theoretical Y values from calculated theta0 and theta1 (numerical):")
ThetaNumerical.runFunction(newThetaNumericalList)


# Analytical solution
simpleLinRegSolution = SimpleLinearAnalytical(xData, yData)

print("Theoretical Y values from calculated theta0 and theta1 (abstract):")
simpleLinRegSolution.runFunction(xData)

print(simpleLinRegSolution)
