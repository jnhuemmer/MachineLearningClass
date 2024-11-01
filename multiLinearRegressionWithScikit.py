import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read in file
rawDataDF = pd.read_csv("C:\\Users\\Jacob\\Desktop\\Wormhole\\Fall2024\\MachineLearning\\HW2\\diabetes_dataset.csv")

# Print dict to show categorical vars
dict = {}
for i in list(rawDataDF.columns):
    dict[i] = rawDataDF[i].value_counts().shape[0]

# Separate categorical and continuous data
catCols = ["sex"]
conCols = ["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6",]

# Converts binary values to True/False
dummiesDF = pd.get_dummies(rawDataDF, columns = catCols, drop_first = True) # Columns are the categories we need dummies for

# Set x and y data
X = dummiesDF.drop(["target"],axis=1)
y = dummiesDF[["target"]]

# Scale data to be between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1)) 
X[conCols] = scaler.fit_transform(X[conCols])

# Create model
dataModel = LinearRegression()
dataModel.fit(X, y)

# Get predicted y values based on model
yPredictions = dataModel.predict(X)

# Print results
print(yPredictions)
print("Theta values are: ")
print(dataModel.intercept_)
print(dataModel.coef_)
print("The data has an r^2 value of: " + str(r2_score(y, yPredictions)))
