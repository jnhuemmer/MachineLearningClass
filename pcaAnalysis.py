import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

class PCA:

    # Initialize and fit data
    def __init__(self, data, nDimension):
        self.data = data
        self.nDimension = nDimension
        self.mean = np.mean(data, axis = 0)
        print(self.mean)
        self.meanCenteredData = self.meanCenter()
        self.cov = self.covariance()
        self.eigenvectors, self.eigenvalues = self.eigen()
        self.principleComponents = self.eigenvectors[0:self.nDimension]

    # Determine covariance matrix
    def covariance(self):
        return np.cov(self.meanCenteredData.T)

    # Determine and order eigenvalues and eigenvectors
    def eigen(self):
        values, vector = np.linalg.eig(self.cov)
        
        vector = vector.T
        values = values[np.argsort(values)[::-1]]
        vector = vector[np.argsort(values)[::-1]]

        return vector, values

    # Mean center the data
    def meanCenter(self):
        return self.data - self.mean

    # Project principle component vectors onto data
    def project(self):
        return np.dot(self.meanCenteredData, self.principleComponents.T)
    
    def varianceRatio(self):
        return ((self.eigenvalues / self.eigenvalues.sum()) * 100).astype(float)


# File paths
filePathQuestion3 = "C:\\Users\\Jacob\\Desktop\\Wormhole\\Fall2024\\MachineLearning\\HW4\\Homework_dataset_prob3.csv"
filePathQuestion4 = "C:\\Users\\Jacob\\Desktop\\Wormhole\\Fall2024\\MachineLearning\\HW4\\Homework_dataset_prob4.csv"

#filePathQuestion3 = "G:\\Other computers\\Gir\\Wormhole\\Fall2024\\MachineLearning\\HW4\\Homework_dataset_prob3.csv"
#filePathQuestion4 = "G:\\Other computers\\Gir\\Wormhole\\Fall2024\\MachineLearning\\HW4\\Homework_dataset_prob4.csv"

# Read in data without header
q3Array = np.loadtxt(filePathQuestion3, delimiter = ",", skiprows = 1)
q4Array = np.loadtxt(filePathQuestion4, delimiter = ",", skiprows = 1, dtype = str)

# Parsing and removing var names
q4Array = q4Array.T
q4Array = q4Array[1::]
q4Array = q4Array.astype(float)

# Determine PC 1
q3PCA = PCA(q3Array, 2)

# Will determine PC1 and PC2
q4PCA = PCA(q4Array, 20)


q3Proj = q3PCA.project()
q4Proj = q4PCA.project()

# Eigenvalue = variance
x = np.arange(len(q4PCA.eigenvalues[0:20]))

# Scree plot
ratios3 = q3PCA.varianceRatio()
plt.plot(x[0:2], ratios3, 'o-', linewidth=2, color='red')
plt.show()

print(q3PCA.eigenvalues)

#plt.scatter(np.arange(60), q3Proj[0], color="red")
#plt.show()

# Scores plot
x1_3 = q3Proj[:, 0]
x2_3 = q3Proj[:, 1]

plt.scatter(x1_3, x2_3, color="red")
plt.show()


# Scree plot
ratios4 = q4PCA.varianceRatio()
plt.plot(x, ratios4[0:20], 'o-', linewidth=2, color='blue')
plt.show()

# Scores plot
x1_4 = q4Proj[:, 0]
x2_4 = q4Proj[:, 1]

plt.scatter(x1_4, x2_4, color="blue")
plt.show()

# Loadings plot
x1_4 = q4PCA.eigenvectors[0]
x2_4 = q4PCA.eigenvectors[1]

plt.scatter(x1_4, x2_4, color="blue")
plt.show()

