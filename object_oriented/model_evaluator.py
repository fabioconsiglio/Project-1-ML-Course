from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import KFold

class ModelEvaluator:
    def __init__(self, model, scaling=False):
        self.model = model
        self.scaling = scaling
        self.scaler = None
        if scaling:
            self.scaler = StandardScaler()

    def split_data(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size)

    def scale_data(self, X_train, X_test):
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        return X_train, X_test

    def bootstrap(self, X, y, n_bootstrap=100):
        mse_list = []
        r2_list = []
        n_samples = len(y)

        for _ in range(n_bootstrap):
            indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
            X_resampled, y_resampled = X[indices], y[indices]
            X_train, X_test, y_train, y_test = self.split_data(X_resampled, y_resampled)
            if self.scaling:
                X_train, X_test = self.scale_data(X_train, X_test)
            self.model.fit(X_train, y_train)
            mse, r2 = self.model.evaluate(X_test, y_test)
            mse_list.append(mse)
            r2_list.append(r2)

        return np.mean(mse_list), np.mean(r2_list)

    def cross_validate(self, X, y, k=5):
        kf = KFold(n_splits=k)
        mse_list = []
        r2_list = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if self.scaling:
                X_train, X_test = self.scale_data(X_train, X_test)
            self.model.fit(X_train, y_train)
            mse, r2 = self.model.evaluate(X_test, y_test)
            mse_list.append(mse)
            r2_list.append(r2)

        return np.mean(mse_list), np.mean(r2_list)

    def evaluate(self, X_train, X_test, y_train, y_test):
        if self.scaling:
            X_train, X_test = self.scale_data(X_train, X_test)
        self.model.fit(X_train, y_train)
        return self.model.evaluate(X_test, y_test)