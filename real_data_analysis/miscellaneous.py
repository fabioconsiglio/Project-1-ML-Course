import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score


# Franke Function Definition
def FrankeFunction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Franke function definition.

    params:
     - x (np.ndarray): x-values.
     - y (np.ndarray): y-values.

    returns:
    - np.ndarray: Franke function values.
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


# Design matrix function
def design_matrix(degree: int) -> PolynomialFeatures:
    """
    Create a design matrix for polynomial regression.

    params:
     - degree (int): Polynomial degree.

    returns:
     - PolynomialFeatures: PolynomialFeatures object.
    """
    sklearn_poly = PolynomialFeatures(degree=degree)
    return sklearn_poly


# Train test splitter
def train_test_splitter(
    X: np.ndarray, z: np.ndarray, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and test sets.

    params:
     - X (np.ndarray): Input data.
     - z (np.ndarray): Target data.
     - test_size (float): Test set size.

    returns:
    np.ndarray: X_train.
    np.ndarray: X_test.
    np.ndarray: z_train.
    np.ndarray: z_test.
    """
    return train_test_split(X, z, test_size=test_size)


# Plot Franke function
def plot_franke_function(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, show: bool = True, save: bool = False
) -> None:
    """
    Plot the Franke function.

    params:
    - x (np.ndarray): x-values.
    - y (np.ndarray): y-values.
    - z (np.ndarray): z-values.
    - show (bool): Show the plot.
    - save (bool): Save the plot.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title("Franke Function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if save:
        fig.savefig("franke_function.png")
    if show:
        plt.show()


# OLS function
def ols(
    X_train: np.ndarray,
    z_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ordinary least squares regression.

    params:
    - X_train (np.ndarray): Training input data.
    - z_train (np.ndarray): Training target data.

    returns:
    - z_test_pred (np.ndarray): Predicted test target data.
    - z_train_pred (np.ndarray): Predicted training target data.
    - beta (np.ndarray): Beta coefficients.
    """
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    z_test_pred = (X_test @ beta).ravel()
    z_train_pred = X_train @ beta.ravel()
    return z_test_pred, z_train_pred, beta


def ridge(
    X_train: np.ndarray,
    z_train: np.ndarray,
    X_test: np.ndarray,
    lambda_: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ridge regression.

    params:
    - X_train (np.ndarray): Training input data.
    - z_train (np.ndarray): Training target data.
    - lambda_ (float): Ridge penalty.

    returns:
    - z_test_pred (np.ndarray): Predicted test target data.
    - z_train_pred (np.ndarray): Predicted training target data.
    - beta (np.ndarray): Beta coefficients.
    """
    I = np.identity(X_train.shape[1])
    beta = np.linalg.pinv(X_train.T @ X_train + lambda_ * I) @ X_train.T @ z_train
    z_test_pred = (X_test @ beta).ravel()
    z_train_pred = (X_train @ beta).ravel()
    return z_test_pred, z_train_pred, beta


def lasso(
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lasso regression.

    params:
    - x_train (np.ndarray): Training input data.
    - z_train (np.ndarray): Training target data.
    - alpha (float): Lasso penalty.

    returns:
    - z_test_pred (np.ndarray): Predicted test target data.
    - z_train_pred (np.ndarray): Predicted training target data.
    - beta (np.ndarray): Beta coefficients.
    """
    lasso_model = Lasso(alpha=alpha, fit_intercept=False, precompute=True)
    lasso_model.fit(x_train, z_train)
    beta = lasso_model.coef_
    z_test_pred = lasso_model.predict(x_test)
    z_train_pred = lasso_model.predict(x_train)
    return z_test_pred, z_train_pred, beta


def k_fold_cv(
    X: np.ndarray,
    z: np.ndarray,
    k: int = 5,
    model: str = "ols",
    degree: int = 5,
    lambda_: float = 0.1,
):
    """
    K-fold cross-validation.

    params:
    - X (np.ndarray): Input data.
    - z (np.ndarray): Target data.
    - k (int): Number of folds.
    - model (str): Model type.
    - degree (int): Polynomial degree.
    - lambda_ (float): Regularization parameter.

    returns:
    - mse (float): Mean squared error.
    - r2 (float): R^2 score.
    """
    n = X.shape[0]
    mse, r2 = 0, 0
    for i in range(k):
        start, end = int(i * n / k), int((i + 1) * n / k)
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        z_train = np.concatenate((z[:start], z[end:]), axis=0)
        X_test, z_test = X[start:end], z[start:end]
        poly = design_matrix(degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        if model == "ols":
            z_test_pred, _, _ = ols(X_train_poly, z_train, X_test_poly)
        elif model == "ridge":
            z_test_pred, _, _ = ridge(
                X_train_poly, z_train, X_test_poly, lambda_
            )
        elif model == "lasso":
            z_test_pred, _, _ = lasso(
                X_train_poly, z_train, X_test_poly, lambda_
            )
        mse += mean_squared_error(z_test, z_test_pred)
        r2 += r2_score(z_test, z_test_pred)
    return mse / k, r2 / k


def calc_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error.

    params:
    - y_true (np.ndarray): True target data.
    - y_pred (np.ndarray): Predicted target data.

    returns:
    - float: Mean squared error.
    """
    return mean_squared_error(y_true, y_pred)


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 score.

    params:
    - y_true (np.ndarray): True target data.
    - y_pred (np.ndarray): Predicted target data.

    returns:
    - float: R^2 score.
    """
    return r2_score(y_true, y_pred)

def calc_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Bias^2.

    params:
    - y_true (np.ndarray): True target data.
    - y_pred (np.ndarray): Predicted target data.

    returns:
    - float: Bias^2.
    """
    mean_pred = np.mean(y_pred, axis=1)
    return np.mean((y_true - mean_pred) ** 2)

def calc_variance(y_pred: np.ndarray) -> float:
    """
    Variance.

    params:
    - y_pred (np.ndarray): Predicted target data.

    returns:
    - float: Variance.
    """
    return np.mean(np.var(y_pred, axis=1, keepdims=True))
