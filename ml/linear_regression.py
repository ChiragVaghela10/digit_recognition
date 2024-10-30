from abc import ABC

import numpy as np
from pathlib import Path

from typing import Tuple

from ml.models import RegressorModelParameters


class GradientDescent(object):
    def __init__(self):
        pass

    def compute_cost(self):
        pass

    def compute_gradient(self):
        pass

    def compute_gradient_descent(self):
        pass


class LinearRegression(object):
    def __init__(self, nodes=1):
        self.nodes = nodes

    def compute_f_x(self, x: np.ndarray, w: np.ndarray, b: np.array) -> np.ndarray:
        pass

    def compute_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.array) -> Tuple[np.ndarray, np.ndarray]:
        samples = x.shape[0]
        features = x.shape[1]
        dw = np.zeros((features, self.nodes))
        db = np.zeros(self.nodes)

        for node in range(self.nodes):
            y_hat = np.zeros(samples).reshape(-1, 1)
            for sample in range(samples):
                y_hat[sample] = np.dot(x[sample], w[:, node]) + b[node]
                dw[:, node] += (y_hat[sample] - y[sample, node]) * x[sample]
                db[node] += y_hat[sample] - y[sample, node]

            dw[:, node] = dw[:, node] / samples
            db[node] = db[node] / samples
        return dw, db

    def compute_cost(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.array,
                     get_y_pred: bool = False) -> [np.ndarray, np.ndarray]:
        samples = x.shape[0]
        cost = np.zeros(self.nodes)
        y_hat = np.zeros((samples, self.nodes))

        for node in range(self.nodes):
            for sample in range(samples):
                y_hat[sample, node] = np.dot(x[sample], w[:, node]) + b[node]
                cost[node] += (y_hat[sample, node] - y[sample, node]) ** 2

            cost[node] = cost[node] / (2 * samples)
        if get_y_pred:
            return y_hat
        else:
            return cost

    def gradient_descent(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.array, alpha: float,
                         iters: int) -> Tuple[np.ndarray, np.array, np.ndarray]:
        cost_list = np.zeros((iters, self.nodes))

        for i in range(iters):
            print('Starting iteration {} ...'.format(i))
            cost = self.compute_cost(x, y, w, b)
            print('Cost after iteration {}: {}'.format(i, cost))
            cost_list[i] = cost
            dw, db = self.compute_gradient(x, y, w, b)
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
        return self.compute_cost(x=xTest, y=yTest, w=w_min, b=b_min, get_y_pred=True)
