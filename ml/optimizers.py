from typing import Tuple

import numpy as np

from ml.activation_functions import ActivationFunction


class GradientDescent(object):
    @staticmethod
    def compute_gradient(x: np.ndarray, y: np.ndarray, y_hat: np.array) -> Tuple[np.ndarray, np.ndarray]:
        samples = y.shape[0]

        dw = np.dot((y_hat - y).T, x) / samples
        db = (y_hat - y).sum(axis=0) / samples

        return dw, db

    @staticmethod
    def compute_cost(y: np.ndarray, y_hat: np.array, get_y_pred: bool = False) -> [np.ndarray, np.ndarray]:
        samples = y.shape[0]

        cost = (y_hat - y) ** 2
        cost = np.sum(cost, axis=0) / (2 * samples)

        if get_y_pred:
            return y_hat
        else:
            return cost

    @classmethod
    def gradient_descent(cls, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.array, nodes: int,
                         a: ActivationFunction, alpha: float, iters: int) -> Tuple[np.ndarray, np.array, np.ndarray]:
        cost_list = np.zeros((iters, nodes))

        for i in range(iters):
            print('Starting iteration {} ...'.format(i))

            y_pred = a.compute_f_x(x, w, b)
            cost = cls.compute_cost(y=y, y_hat=y_pred)

            print('Cost after iteration {}: {}'.format(i, cost))
            cost_list[i] = cost
            dw, db = cls.compute_gradient(x=x, y=y, y_hat=y_pred)
            w_tmp = w - alpha * dw
            b_tmp = b - alpha * db
            w = w_tmp
            b = b_tmp

        return w, b, cost_list
