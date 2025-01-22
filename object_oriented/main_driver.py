from data_handler import DataHandler
from model_evaluator import ModelEvaluator
from regression_model import RegressionModel

class RegressionAnalysis:
    def __init__(self, data_type='franke', method='ols', scaling=False):
        self.data_handler = DataHandler(data_type)

        self.model = RegressionModel(method)

        self.evaluator = ModelEvaluator(self.model, scaling=scaling)

    def run(self, test_size=0.2):
        X, y = self.data_handler.x, self.data_handler.y
        X_train, X_test, y_train, y_test = self.evaluator.split_data(X, y, test_size)
        mse, r2 = self.evaluator.evaluate(X_train, X_test, y_train, y_test)
        print(f"MSE: {mse}, R^2: {r2}")

    def run_bootstrap(self, n_bootstrap=100):
        X, y = self.data_handler.x, self.data_handler.y
        mse, r2 = self.evaluator.bootstrap(X, y, n_bootstrap)
        print(f"Bootstrap MSE: {mse}, Bootstrap R^2: {r2}")

    def run_cross_validation(self, k=5):
        X, y = self.data_handler.x, self.data_handler.y
        mse, r2 = self.evaluator.cross_validate(X, y, k)
        print(f"Cross-Validation MSE: {mse}, Cross-Validation R^2: {r2}")

if __name__ == "__main__":
    # OLS Regression on Franke function
    analysis = RegressionAnalysis(data_type='franke', method='ols', scaling=False)
    analysis.run()

    # # Ridge Regression on real-world data
    # analysis = RegressionAnalysis(data_type='real', method='ridge', scaling=True)
    # analysis.run()

    # # Lasso Regression on real-world data
    # analysis = RegressionAnalysis(data_type='real', method='lasso', scaling=False)
    # analysis.run()