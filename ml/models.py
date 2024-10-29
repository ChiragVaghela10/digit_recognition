from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


class RegressorModelParameters(object):
    """
    This class is responsible for defining parameters for a regressor models e.g. Linear Regression,
    Logistic Regression, etc.

    The class stores model weights and cost history associated with it as well to inspect calculated optimum weights and
    how well algorithm learned over each iteration.
    """
    def __init__(self, w_init: np.ndarray = None, b_init: np.ndarray = None, filepath: Path = None):
        self.w_init = w_init
        self.b_init = b_init
        self.weights_filepath = filepath
        self.w_min = None
        self.b_min = None
        self.cost_history = None

    def save_weights(self, weights: np.ndarray, bias: np.ndarray, cost_history: np.ndarray, filepath: Path = None) -> None:
        """
        Saves weights and cost history for model.

        filepath: filepath to save weights and cost history into
        weights: numpy array of trained weights
        bias: numpy array of trained bias
        cost_history: numpy array of cost history
        """
        if filepath:
            self.weights_filepath = filepath

        self.w_min = weights
        self.b_min = bias
        self.cost_history = cost_history

        with h5py.File(str(self.weights_filepath), 'w') as file:
            file.create_dataset('init_weights', data=self.w_init)
            file.create_dataset('init_bias', data=self.b_init)
            file.create_dataset('min_weights', data=weights)
            file.create_dataset('min_bias', data=bias)
            file.create_dataset('cost_history', data=cost_history)
            file.close()

    def load_initial_weights(self, filepath: Path = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads initial weights from file if filepath provided otherwise returns initial weights from the instance.

        filepath: filepath to load initial weights from.
        """
        if filepath:
            self.weights_filepath = filepath

        if self.weights_filepath.exists():
            with h5py.File(str(self.weights_filepath), 'r') as model_file:
                weights = model_file['init_weights'][()]
                bias = model_file['init_bias'][()]

                return weights, bias
        else:
            raise FileNotFoundError

    def load_optimum_weights(self, filepath: Path = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads trained weights and cost history from file if filepath provided.

        filepath: filepath to load weights and cost history from.
        """
        if filepath:
            self.weights_filepath = filepath

        if self.weights_filepath.exists():
            with h5py.File(str(self.weights_filepath), 'r') as model_file:
                self.w_min = model_file['min_weights'][()]
                self.b_min = model_file['min_bias'][()]

                return self.w_min, self.b_min
        else:
            raise FileNotFoundError

    def load_plot_history(self, filepath: Path = None) -> Tuple[np.ndarray]:
        if filepath:
            self.weights_filepath = filepath

        if self.weights_filepath.exists():
            with h5py.File(str(self.weights_filepath), 'r') as model_file:
                self.cost_history = model_file['cost_history'][()]

            return self.cost_history
        else:
            raise FileNotFoundError
