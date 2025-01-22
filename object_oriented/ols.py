import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class OLS:
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        # Closed form solution of OLS: β = (X^T X)^(-1) X^T y
        X_transpose = X.T
        self.beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

    def predict(self, X):
        # Predict using the learned coefficients
        return X @ self.beta

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2