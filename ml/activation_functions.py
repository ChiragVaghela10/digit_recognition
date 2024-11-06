from abc import ABC

import numpy as np


class ActivationFunction(ABC):
    def __init__(self):
        pass

    @staticmethod
    def compute_f_x(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def compute_cost(y: np.ndarray, y_hat: np.array) -> [np.ndarray, np.ndarray]:
        pass

    def apply_threshold(self, y_hat: np.ndarray) -> np.ndarray:
        pass


class LinearActivationFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_f_x(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(X, w.T) + b

    @staticmethod
    def compute_cost(y: np.ndarray, y_hat: np.array) -> [np.ndarray, np.ndarray]:
        samples = y.shape[0]

        cost = (y_hat - y) ** 2
        cost = np.sum(cost, axis=0) / (2 * samples)

        return cost

    def apply_threshold(self, y_hat: np.ndarray) -> np.ndarray:
        return (y_hat == y_hat.max(axis=1, keepdims=1)).astype(int)


class LogisticActivationFunction(ActivationFunction):
    def __init__(self, threshold: float = None):
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def compute_f_x(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))

        fx = LinearActivationFunction()
        fx = fx.compute_f_x(X, w, b)
        return sigmoid(fx)

    @staticmethod
    def compute_cost(y: np.ndarray, y_hat: np.array) -> [np.ndarray, np.ndarray]:
        samples = y.shape[0]

        cost = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        cost = np.sum(cost, axis=0) / (- samples)

        return cost

    def apply_threshold(self, y_hat: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if threshold:
            self.threshold = threshold

        y_hat_binary = np.zeros_like(y_hat)
        y_hat_binary[np.arange(len(y_hat)), y_hat.argmax(1)] = 1
        return y_hat_binary


class ReLUActivationFunction(ActivationFunction):
    def compute_f_x(self, X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        pass
