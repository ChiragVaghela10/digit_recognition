from abc import ABC

import numpy as np
from pathlib import Path

from typing import Tuple

from ml.models import RegressorModelParameters


class GradientDescent(object):
    def __init__(self, x: np.ndarray = None, y: np.ndarray = None, w: np.ndarray = None, b: np.ndarray = None,
                 alpha: float = None, iters: int = None):
        self.x = x
        self.y = y
        self.w = w
        self.b = b
        self.alpha = alpha
        self.iters = iters

    def compute_cost(self):
        pass

    def compute_gradient(self):
        pass

    def compute_gradient_descent(self):
        pass


class LinearRegressor(object):
    def __init__(self, nodes=1):
        self.nodes = nodes

    def compute_f_x(self, x: np.ndarray, w: np.ndarray, b: np.array) -> np.ndarray:
        return np.dot(x, w.T) + b

    def compute_gradient(self, x: np.ndarray, y: np.ndarray, y_hat: np.array) -> Tuple[np.ndarray, np.ndarray]:
        samples = y.shape[0]

        dw = np.dot((y_hat - y).T, x) / samples
        db = (y_hat - y).sum(axis=0) / samples

        return dw, db

    def compute_cost(self, y: np.ndarray, y_hat: np.array, get_y_pred: bool = False) -> [np.ndarray, np.ndarray]:
        samples = y.shape[0]

        cost = (y_hat - y) ** 2
        cost = np.sum(cost, axis=0) / (2 * samples)

        if get_y_pred:
            return y_hat
        else:
            return cost

    def gradient_descent(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.array, alpha: float,
                         iters: int) -> Tuple[np.ndarray, np.array, np.ndarray]:
        cost_list = np.zeros((iters, self.nodes))

        for i in range(iters):
            print('Starting iteration {} ...'.format(i))

            y_pred = self.compute_f_x(x, w, b)
            cost = self.compute_cost(y=y, y_hat=y_pred)

            print('Cost after iteration {}: {}'.format(i, cost))
            cost_list[i] = cost
            dw, db = self.compute_gradient(x=x, y=y, y_hat=y_pred)
            w_tmp = w - alpha * dw
            b_tmp = b - alpha * db
            w = w_tmp
            b = b_tmp

        return w, b, cost_list

    def train(self, xTrain: np.ndarray, yTrain: np.ndarray, parameters: RegressorModelParameters,
              learning_rate: float, iterations: int, weights_filepath: Path = None) -> RegressorModelParameters:
        w_init, b_init = parameters.load_initial_weights(filepath=weights_filepath)

        w_min, b_min, cost_list = self.gradient_descent(x=xTrain, y=yTrain, w=w_init, b=b_init, alpha=learning_rate,
                                                        iters=iterations)
        print('Optimum weights and bias are: {}, {}'.format(w_min, b_min))
        parameters.save_weights(weights=w_min, bias=b_min, cost_history=cost_list, filepath=weights_filepath)
        return parameters

    def predict(self, xTest: np.ndarray, yTest: np.ndarray,
                parameters: RegressorModelParameters, weights_filepath: Path = None) -> Tuple[np.array, np.array]:
        print('Predicting...')
        w_min, b_min = parameters.load_optimum_weights(weights_filepath)
        y_pred = self.compute_f_x(x=xTest, w=w_min, b=b_min)
        return self.compute_cost(y=yTest,y_hat=y_pred, get_y_pred=True)
