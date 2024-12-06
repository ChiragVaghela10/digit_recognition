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
        """
        Computers f_sub_w_comma_b(x) = w.T * X + b

        This function is particularly important because it computes the value of hypothesis function f(x).
        """
        # NOTE: The implementation explained in Andrew Ng's course defines X as (n X m) while I used X as (m X n).
        # This different has led to the interchange of variables in the np.dot() function call which is performing
        # matrix multiplication of X (m X n) and w (n X l). where, l is number of nodes. Therefore, result (y_hat)
        # is (m X l).
        return np.dot(X, w.T) + b

    @staticmethod
    def compute_cost(y: np.ndarray, y_hat: np.array) -> [np.ndarray, np.ndarray]:
        samples = y.shape[0]

        cost = (y_hat - y) ** 2
        cost = np.sum(cost, axis=0) / (2 * samples)

        return cost

    def apply_threshold(self, y_hat: np.ndarray) -> np.ndarray:
        y_hat_binary = np.zeros_like(y_hat)
        y_hat_binary[np.arange(len(y_hat)), y_hat.argmax(1)] = 1
        return y_hat_binary


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
        y_hat_binary = np.zeros_like(y_hat)
        y_hat_binary[np.arange(len(y_hat)), y_hat.argmax(1)] = 1
        return y_hat_binary


class ReLUActivationFunction(ActivationFunction):
    def compute_f_x(self, X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        pass
