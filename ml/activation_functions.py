from abc import ABC

import numpy as np


class ActivationFunction(ABC):
    def __init__(self):
        self.X = None
        self.w = None
        self.b = None

    @staticmethod
    def compute_f_x(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        pass


class LinearActivationFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_f_x(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(X, w.T) + b


class LogisticActivationFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_f_x(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))

        fx = LinearActivationFunction()
        fx = fx.compute_f_x(X, w, b)
        return sigmoid(fx)


class ReLUActivationFunction(ActivationFunction):
    def compute_f_x(self, X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        pass
