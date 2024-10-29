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

    def compute_gradient(self, xTrain: np.array, yTrain: np.array, w: np.array,
                         b: np.array) -> Tuple[np.array, np.array]:
        samples = xTrain.shape[0]
        features = xTrain.shape[1]
        dw = np.zeros((features, self.nodes))
        db = np.zeros(self.nodes)

        for node in range(self.nodes):
            y_hat = np.zeros(samples).reshape(-1, 1)
            for sample in range(samples):
                y_hat[sample] = np.dot(xTrain[sample], w[:, node]) + b[node]
                dw[:, node] += (y_hat[sample] - yTrain[sample, node]) * xTrain[sample]
                db[node] += y_hat[sample] - yTrain[sample, node]

            dw[:, node] = dw[:, node] / samples
            db[node] = db[node] / samples
        return dw, db

    def compute_cost(self, x: np.array, y: np.array, w: np.array, b: np.array,
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

    def gradient_descent(self, xTrain: np.array, yTrain: np.array, w_init: np.array, b_init: np.array, alpha: float,
                         iters: int) -> Tuple[np.array, np.array, np.array]:
        cost_list = np.zeros((iters, self.nodes))
        w = w_init
        b = b_init

        for i in range(iters):
            print('Starting iteration {} ...'.format(i))
            cost = self.compute_cost(xTrain, yTrain, w, b)
            print('Cost after iteration {}: {}'.format(i, cost))
            cost_list[i] = cost
            dw, db = self.compute_gradient(xTrain, yTrain, w, b)
            w_tmp = w - alpha * dw
            b_tmp = b - alpha * db
            w = w_tmp
            b = b_tmp

        return w, b, cost_list

    def train(self, xTrain: np.array, yTrain: np.array, parameters: RegressorModelParameters,
              learning_rate: float, iterations: int, weights_filepath: Path = None) -> RegressorModelParameters:
        w_init, b_init = parameters.load_initial_weights(filepath=weights_filepath)

        w_min, b_min, cost_list = self.gradient_descent(xTrain, yTrain, w_init, b_init, learning_rate, iterations)
        print('Optimum weights and bias are: {}, {}'.format(w_min, b_min))
        parameters.save_weights(weights=w_min, bias=b_min, cost_history=cost_list, filepath=weights_filepath)
        return parameters

    def predict(self, xTest: np.ndarray, yTest: np.ndarray,
                parameters: RegressorModelParameters, weights_filepath: Path = None) -> Tuple[np.array, np.array]:
        print('Predicting...')
        w_min, b_min = parameters.load_optimum_weights(weights_filepath)
        return self.compute_cost(data=xTest, target=yTest, w=w_min, b=b_min, get_y_pred=True)
