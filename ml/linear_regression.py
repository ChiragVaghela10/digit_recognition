import pickle
from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from contants import colors


class Regressor(ABC):
    w_init = None
    b_init = None
    w_min = None
    b_min = None

    def __init__(self):
        pass

    def save_weights(self, weights: np.ndarray, bias: np.array, cost: np.ndarray):
        pass

    def load_weights(self) -> Tuple[np.ndarray, np.array]:
        pass


class LinearRegression(object):
    def __init__(self, nodes=1):
        self.nodes = nodes
        self.cost_list = None
        self.w_init = None
        self.b_init = None
        self.w_min = None
        self.b_min = None

    @staticmethod
    def save_weights(weights: np.ndarray, bias: np.array, cost: np.ndarray) -> None:
        with open('weights/lr_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)
            pickle.dump(bias, f)
            pickle.dump(cost, f)
            print('Saved model weights')

    @staticmethod
    def load_weights() -> Tuple[np.ndarray, np.array, np.ndarray]:
        with open('weights/lr_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
            bias = pickle.load(f)
            cost = pickle.load(f)
            print('Loaded model weights')
            return weights, bias, cost

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

    def compute_cost(self, data: np.array, target: np.array, w: np.array, b: np.array,
                     get_y_pred: bool = False) -> [np.ndarray, np.ndarray]:
        samples = data.shape[0]
        cost = np.zeros(self.nodes)
        y_hat = np.zeros((samples, self.nodes))

        for node in range(self.nodes):
            for sample in range(samples):
                y_hat[sample, node] = np.dot(data[sample], w[:, node]) + b[node]
                cost[node] += (y_hat[sample, node] - target[sample, node]) ** 2

            cost[node] = cost[node] / (2 * samples)
        if get_y_pred:
            return y_hat
        else:
            return cost

    def gradient_descent(self, xTrain: np.array, yTrain: np.array, w_init: np.array, b_init: np.array, alpha: float,
                         iters: int) -> Tuple[np.array, np.array, np.array]:
        self.cost_list = np.zeros((iters, self.nodes))
        w = w_init
        b = b_init

        for i in range(iters):
            print('Starting iteration {} ...'.format(i))
            cost = self.compute_cost(xTrain, yTrain, w, b)
            print('Cost after iteration {}: {}'.format(i, cost))
            self.cost_list[i] = cost
            dw, db = self.compute_gradient(xTrain, yTrain, w, b)
            w_tmp = w - alpha * dw
            b_tmp = b - alpha * db
            w = w_tmp
            b = b_tmp

        return w, b, self.cost_list

    def train(self, xTrain: np.array, yTrain: np.array, w_init: np.array, b_init: np.array,
              learning_rate: float, iterations: int) -> Tuple[np.array, np.array, np.array]:
        self.w_init = w_init
        self.b_init = b_init

        self.w_min, self.b_min, self.cost_list = self.gradient_descent(xTrain, yTrain, w_init,
                                                                       b_init, learning_rate, iterations)
        print('Optimum weights and bias are: {}, {}'.format(self.w_min, self.b_min))

        return self.w_min, self.b_min, self.cost_list

    def predict(self, xTest: np.array, yTest: np.array, wMin: np.array, bMin: float) -> Tuple[np.array, np.array]:
        print('Predicting...')
        return self.compute_cost(data=xTest, target=yTest, w=wMin, b=bMin, get_y_pred=True)
