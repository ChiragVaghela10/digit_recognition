import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List

from ml.models import Model


class LinearRegression(Model):
    def __init__(self, xTrain: np.array, yTrain: np.array, xTest: np.array, yTest: np.array):
        super().__init__(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest)
        self.w_init = None
        self.b_init = None
        self.w_min = None
        self.b_min = None
        self.alpha = None
        self.iters = None

    def compute_gradient(self, w_init: np.array, b_init: np.array) -> Tuple[np.array, np.array]:
        w = w_init
        b = b_init
        m = self.X_train.shape[0]
        y_hat = np.zeros(m).reshape(-1, 1)

        dw, db = 0, 0
        for i in range(m):
            y_hat[i] = np.dot(self.X_train[i], w) + b
            dw += (y_hat[i] - self.y_train[i]) * self.X_train[i]
            db += y_hat[i] - self.y_train[i]

        dw = dw / m
        db = db / m
        return dw, db

    def compute_cost(self, x: np.array, w: np.array, b: np.array) -> float:
        m = x.shape[0]
        y_hat = np.zeros(m).reshape(-1, 1)
        cost = 0

        for i in range(m):
            y_hat[i] = np.dot(x[i], w) + b
            cost += (y_hat[i] - self.y_train[i]) ** 2

        cost = cost / (2 * m)
        return cost

    def plot_cost(self, cost_data: List[float]):
        plt.plot(cost_data[:10])
        plt.show()

    def gradient_descent(self, w_init: np.array, b_init: np.array, alpha: float,
                         iters: int) -> Tuple[np.array, np.array]:
        w = w_init
        b = b_init
        cost_list = []
        for i in range(iters):
            print('Starting iteration {} ...'.format(i))
            cost = self.compute_cost(self.X_train, w, b)
            print('Cost after iteration {}: {}'.format(i, cost))
            cost_list.append(cost)
            dw, db = self.compute_gradient(w, b)
            w_tmp = w - alpha * dw
            b_tmp = b - alpha * db
            w = w_tmp
            b = b_tmp

        self.plot_cost(cost_list)
        return (w, b)

    def train(self, w_init: np.array, b_init: np.array, learning_rate: float, iterations: int) -> Tuple[np.array, np.array]:
        self.w_init = w_init
        self.b_init = b_init
        self.alpha = learning_rate
        self.iters = iterations

        self.w_min, self.b_min = self.gradient_descent(w_init, b_init, learning_rate, iterations)
        print('Optimum weights and bias are: {}, {}'.format(self.w_min, self.b_min))
        return (self.w_min, self.b_min)

    def predict(self, x:np.array, w: np.array, b: float) -> np.array:
        rmse = self.compute_cost(x, w, b)
        print('RMSE: {}'.format(rmse))

    def eval(self, x: np.array, y: np.array) -> np.array:
        pass


def normalize(X) -> tuple:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    print(mu, sigma)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


X_train = np.arange(1, 11).reshape(-1, 1)
y_train = X_train ** 2
X_test = np.array([np.arange(11, 16)])
y_test = np.array([X_test ** 2])

# Feature Engineering
X_train = np.c_[X_train, X_train ** 2, X_train ** 3]
X_train, mu, sigma = normalize(X_train)

W_init = np.array(np.zeros(X_train.shape[1]))
b_init = np.array(np.zeros(1))

my_model = LinearRegression(xTrain=X_train, yTrain=y_train, xTest=X_test, yTest=y_test)
w_min, b_min = my_model.train(w_init=W_init, b_init=b_init, learning_rate=5e-1, iterations=15000)
#my_model.predict()

