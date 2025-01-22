import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# Franke Function Definition
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75 * np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5 * np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Set parameters
np.random.seed(42)
n = 100  # Number of data points
n_boostraps = 100  # Number of bootstrap samples
maxdegree = 5  # Maximum polynomial degree

# Generate data from Franke function
x = np.random.rand(n, 1)
y = np.random.rand(n, 1)
z = FrankeFunction(x, y) + np.random.normal(0, 0.1, (n, 1))  # Adding noise

# Prepare variables for bias-variance analysis
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(np.c_[x, y], z, test_size=0.2)

# Bias-Variance tradeoff for polynomial degrees
for degree in range(maxdegree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    y_pred = np.empty((y_test.shape[0], n_boostraps))  # Ensure y_pred matches the shape of y_test
    
    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)  # Resample the training data
        y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()  # Predict on the same test data size

    # Store results for error, bias^2, and variance
    polydegree[degree] = degree
    error[degree] = np.mean(np.mean((y_test - y_pred)**2, axis=1))
    bias[degree] = np.mean((y_test - np.mean(y_pred, axis=1))**2)
    variance[degree] = np.mean(np.var(y_pred, axis=1))
    
    # Output results for each polynomial degree
    print(f'Polynomial degree: {degree}')
    print(f'Error: {error[degree]}')
    print(f'Bias^2: {bias[degree]}')
    print(f'Variance: {variance[degree]}')
    print(f'{error[degree]} >= {bias[degree]} + {variance[degree]} = {bias[degree] + variance[degree]}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='Bias^2')
plt.plot(polydegree, variance, label='Variance')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE / Bias^2 / Variance')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.show()
