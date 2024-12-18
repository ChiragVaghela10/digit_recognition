from abc import ABC
from typing import Tuple

import numpy as np

from ml.activation_functions import ActivationFunction
from ml.experiments.experiment import ExpLearningOfPredictedValues


class Optimizer(ABC):
    def __init__(self):
        pass

#     @staticmethod
#     def compute_cost(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         pass
#
#     @staticmethod
#     def compute_gradient(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         pass
#
#     @classmethod
#     def gradient_descent(cls, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         pass
#
#
# class BinaryCrossEntropy(Optimizer):
#     def __init__(self):
#         super().__init__()
#
#     @staticmethod
#     def compute_cost(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         pass
#
#     @staticmethod
#     def compute_gradient(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         pass
#
#     @staticmethod
#     def gradient_descent(self, y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         pass
#

# EXPERIMENTAL CODE
# exp = ExpLearningOfPredictedValues()


class GradientDescent(Optimizer):
    @staticmethod
    def compute_gradient(x: np.ndarray, y: np.ndarray, y_hat: np.array) -> Tuple[np.ndarray, np.ndarray]:
        samples = y.shape[0]

        # NOTE: The implementation explained in Andrew Ng's course defines X as (n X m) while I used X as (m X n).
        # This different has led to the interchange of variables in the np.dot() function call which is performing
        # matrix multiplication of X (m X n) and dz = (y_hat - y) = (a - y) having dimensions (m X l). where, l is
        # number of nodes. Refer comments in compute_fx for explanation. Therefore, result (dw) is (l X n).
        dw = np.dot((y_hat - y).T, x) / samples
        db = (y_hat - y).sum(axis=0) / samples

        return dw, db

    @classmethod
    def gradient_descent(cls, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.array, nodes: int,
                         a: ActivationFunction, alpha: float, iters: int) -> Tuple[np.ndarray, np.array, np.ndarray]:
        cost_list = np.zeros((iters, nodes))

        for i in range(iters):
            print('Starting iteration {} ...'.format(i))

            y_pred = a.compute_f_x(x, w, b)

            # EXPERIMENTAL CODE
            # exp.add_data(y_pred)

            cost = a.compute_cost(y=y, y_hat=y_pred)

            print('Cost after iteration {}: {}'.format(i, cost))
            cost_list[i] = cost
            dw, db = cls.compute_gradient(x=x, y=y, y_hat=y_pred)
            w_tmp = w - alpha * dw
            b_tmp = b - alpha * db
            w = w_tmp
            b = b_tmp

        # EXPERIMENTAL CODE
        # exp.plot_learning()

        return w, b, cost_list
