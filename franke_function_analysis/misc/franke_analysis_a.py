import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Franke Function Definition
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75 * np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5 * np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Generate data
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
z = FrankeFunction(x, y) + np.random.normal(0, 1, n)  # Adding noise

# Prepare the data for train-test split
X = np.c_[x, y]

# Train-Test Split
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

# Lists to store errors and R^2 values
mse_train_list, mse_test_list = [], []
r2_train_list, r2_test_list = [], []
beta_coefficients = []

# Polynomial Regression for degrees 1 to 5
degrees = np.arange(1, 6, 1)
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Linear Regression model
    beta = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ z_train
    
    # Predictions
    z_train_pred = X_train_poly @ beta
    z_test_pred = X_test_poly @ beta
    
    # Calculate MSE and R^2
    mse_train = mean_squared_error(z_train, z_train_pred)
    mse_test = mean_squared_error(z_test, z_test_pred)
    r2_train = r2_score(z_train, z_train_pred)
    r2_test = r2_score(z_test, z_test_pred)
    
    # Append results to the lists
    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    r2_train_list.append(r2_train)
    r2_test_list.append(r2_test)
    beta_coefficients.append(beta)

# Plot MSE and R^2 as a function of the polynomial degree
plt.figure(figsize=(14, 6))

# Plot MSE
plt.subplot(1, 2, 1)
plt.plot(degrees, mse_train_list, label='Train MSE', marker='o')
plt.plot(degrees, mse_test_list, label='Test MSE', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('MSE vs Polynomial Degree')
plt.legend()

# Plot R^2
plt.subplot(1, 2, 2)
plt.plot(degrees, r2_train_list, label='Train R^2', marker='o')
plt.plot(degrees, r2_test_list, label='Test R^2', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('R^2 Score')
plt.title('R^2 Score vs Polynomial Degree')
plt.legend()

plt.show()

# Plot coefficients beta as the polynomial degree increases
plt.figure(figsize=(10, 6))
for i in range(len(beta_coefficients[0])):
    plt.plot(degrees, [beta[i] for beta in beta_coefficients], '-o', label=f'Coefficient {i}')
plt.xlabel('Coefficient Index')
plt.ylabel('Beta Coefficients')
plt.title('Beta Coefficients as Polynomial Degree Increases')
plt.legend()
plt.show()

# Now let's apply scaling and check the effect
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mse_train_list_scaled, mse_test_list_scaled = [], []
r2_train_list_scaled, r2_test_list_scaled = [], []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly_scaled = poly.fit_transform(X_train_scaled)
    X_test_poly_scaled = poly.transform(X_test_scaled)
    
    # Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly_scaled, z_train)
    
    # Predictions
    z_train_pred_scaled = lin_reg.predict(X_train_poly_scaled)
    z_test_pred_scaled = lin_reg.predict(X_test_poly_scaled)
    
    # Calculate MSE and R^2
    mse_train_scaled = mean_squared_error(z_train, z_train_pred_scaled)
    mse_test_scaled = mean_squared_error(z_test, z_test_pred_scaled)
    r2_train_scaled = r2_score(z_train, z_train_pred_scaled)
    r2_test_scaled = r2_score(z_test, z_test_pred_scaled)
    
    # Append results to the lists
    mse_train_list_scaled.append(mse_train_scaled)
    mse_test_list_scaled.append(mse_test_scaled)
    r2_train_list_scaled.append(r2_train_scaled)
    r2_test_list_scaled.append(r2_test_scaled)

# Plot comparison of scaled vs non-scaled MSE and R^2
plt.figure(figsize=(14, 6))

# Plot MSE comparison
plt.subplot(1, 2, 1)
plt.plot(degrees, mse_test_list, label='Test MSE (No Scaling)', marker='o')
plt.plot(degrees, mse_test_list_scaled, label='Test MSE (Scaled)', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('MSE Comparison (Scaled vs Non-Scaled)')
plt.legend()

# Plot R^2 comparison
plt.subplot(1, 2, 2)
plt.plot(degrees, r2_test_list, label='Test R^2 (No Scaling)', marker='o')
plt.plot(degrees, r2_test_list_scaled, label='Test R^2 (Scaled)', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('R^2 Score')
plt.title('R^2 Comparison (Scaled vs Non-Scaled)')
plt.legend()

plt.show()

# Plot Franke Function and the predicted values as scatter plot for 5th degree polynomial

# Generate meshgrid for Franke Function
x_mesh, y_mesh = np.meshgrid(x, y)
z_mesh = FrankeFunction(x_mesh, y_mesh)

# Predictions for 5th degree polynomial
poly = PolynomialFeatures(degree=5)
X_train_poly_scaled = poly.fit_transform(X_train_scaled)
X_test_poly_scaled = poly.transform(X_test_scaled)
beta = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ z_train
z_mesh_pred = X_test_poly_scaled @ beta

# Plot Franke Function
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.3, cmap=cm.coolwarm)
ax.set_title('Franke Function')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Plot Predictions
x_pred = X_test[:, 0]
y_pred = X_test[:, 1]
x_pred_mesh, y_pred_mesh = np.meshgrid(x_pred, y_pred)

plt.legend()
plt.show()