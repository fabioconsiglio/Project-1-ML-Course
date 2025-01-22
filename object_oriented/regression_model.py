from lasso import LassoRegression
from ridge import Ridge
from ols import OLS
import numpy as np

class RegressionModel:
    def __init__(self, method='ols'):
        if method == 'ols':
            self.model = OLS()
        elif method == 'ridge':
            self.model = Ridge()
        elif method == 'lasso':
            self.model = LassoRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def fit(self, X_train, y_train):
        # Implement the fitting logic
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
