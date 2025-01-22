from sklearn.linear_model import Lasso
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class LassoRegression:
    def __init__(self, alpha=1.0):
        self.model = Lasso(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = self.model.score(X_test, y_test)
        return mse, r2