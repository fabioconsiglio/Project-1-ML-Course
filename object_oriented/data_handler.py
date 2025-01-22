import numpy as np

class DataHandler:
    def __init__(self, data_type='franke'):
        if data_type == 'franke':
            self.x, self.y = self.generate_franke_data()
        else:
            self.x, self.y, self.z = self.load_real_world_data()

    def generate_franke_data(self):
        # Generate x and y arrays and calculate z using the Franke function
        n = 100
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        x, y = np.meshgrid(x, y)
        z = self.FrankeFunction(x, y)
        X = np.vstack((x.ravel(), y.ravel())).T
        z_data = z.ravel()  # Reshape z to match the shape of X
        return X, z_data

    def FrankeFunction(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
    
    def load_real_world_data(self, filepath):
        # Load and preprocess real-world data from file (e.g., GeoTIFF)
        pass

    def add_noise(self, noise_level):
        # Add random noise to the data (for testing regression models)
        pass