import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List

from ml.models import Model


class LinearRegression(Model):
    def __init__(self):
        super().__init__()
        self.w_init = None
        self.b_init = None
        self.w_min = None
        self.b_min = None
        self.alpha = None
        self.iters = None

    def compute_gradient(self, xTrain: np.array, yTrain: np.array, w: np.array, b: np.array) -> Tuple[np.array, np.array]:
        m = xTrain.shape[0]
        y_hat = np.zeros(m).reshape(-1, 1)

        dw, db = 0, 0
        for i in range(m):
            y_hat[i] = np.dot(xTrain[i], w) + b
            dw += (y_hat[i] - yTrain[i]) * xTrain[i]
            db += y_hat[i] - yTrain[i]

        dw = dw / m
        db = db / m
        return dw, db

    def compute_cost(self, data: np.array, target: np.array, w: np.array, b: np.array,
                     get_y_pred: bool = False) -> [float, Tuple[np.ndarray, float]]:
        m = data.shape[0]
        y_hat = np.zeros(m).reshape(-1, 1)
        cost = 0

        for i in range(m):
            y_hat[i] = np.dot(data[i], w) + b
            cost += (y_hat[i] - target[i]) ** 2

        cost = cost / (2 * m)
        if get_y_pred:
            return y_hat, cost
        else:
            return cost

    def plot_cost(self, cost_data: List[float]):
        plt.plot(cost_data[:10])
        plt.show()

    def gradient_descent(self, xTrain: np.array, yTrain: np.array, w_init: np.array, b_init: np.array, alpha: float,
                         iters: int) -> Tuple[np.array, np.array]:
        w = w_init
        b = b_init
        cost_list = []
        for i in range(iters):
            print('Starting iteration {} ...'.format(i))
            cost = self.compute_cost(xTrain, yTrain, w, b)
            print('Cost after iteration {}: {}'.format(i, cost))
            cost_list.append(cost)
            dw, db = self.compute_gradient(xTrain, yTrain, w, b)
            w_tmp = w - alpha * dw
            b_tmp = b - alpha * db
            w = w_tmp
            b = b_tmp

        self.plot_cost(cost_list)
        return w, b

    def train(self, xTrain: np.array, yTrain: np.array, w_init: np.array, b_init: np.array,
              learning_rate: float, iterations: int) -> Tuple[np.array, np.array]:
        self.w_init = w_init
        self.b_init = b_init
        self.alpha = learning_rate
        self.iters = iterations

        self.w_min, self.b_min = self.gradient_descent(xTrain, yTrain, w_init, b_init, learning_rate, iterations)
        print('Optimum weights and bias are: {}, {}'.format(self.w_min, self.b_min))
        return self.w_min, self.b_min

    def predict(self, xTest: np.array, yTest: np.array, wMin: np.array, bMin: float) -> np.array:
        print('Predicting...')
        y_pred, rmse = self.compute_cost(data=xTest, target=yTest, w=wMin, b=bMin, get_y_pred=True)
        print("Prediction : {}".format(y_pred))
        print('RMSE: {}'.format(rmse))


class Normalizer(object):
    def __init__(self):
        self.mu = None
        self.sigma = None

    def normalize(self, X) -> np.array:
        if self.mu is None:
            self.mu = np.mean(X, axis=0)
        if self.sigma is None:
            self.sigma = np.std(X, axis=0)
        print("mu and sigma are: {}, {}".format(self.mu, self.sigma))
        X_norm = (X - self.mu) / self.sigma
        return X_norm


X_train = np.arange(1, 11).reshape(-1, 1)
y_train = X_train ** 2
X_test = np.arange(11, 16).reshape(-1, 1)
y_test = X_test ** 2

# Feature Engineering
X_train = np.c_[X_train, X_train ** 2, X_train ** 3]
X_test = np.c_[X_test, X_test ** 2, X_test ** 3]

# Normalization
normalizer = Normalizer()
X_train = normalizer.normalize(X_train)
X_test = normalizer.normalize(X_test)

W_init = np.array(np.zeros(X_train.shape[1]))
b_init = np.array(np.zeros(1))

my_model = LinearRegression()
w_min, b_min = my_model.train(xTrain=X_train, yTrain=y_train, w_init=W_init, b_init=b_init, learning_rate=5e-1, iterations=15000)
my_model.predict(xTest=X_test, yTest=y_test, wMin=w_min, bMin=b_min)

