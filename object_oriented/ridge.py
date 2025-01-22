import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Regularization strength
        self.beta = None

    def fit(self, X, y):
        # Closed form solution of Ridge regression: β = (X^T X + αI)^(-1) X^T y
        X_transpose = X.T
        n_features = X.shape[1]
        I = np.eye(n_features)
        self.beta = np.linalg.inv(X_transpose @ X + self.alpha * I) @ X_transpose @ y

    def predict(self, X):
        return X @ self.beta

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2