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
        self.alpha = None
        self.iters = None

    def compute_gradient(self, w_init: np.array, b_init: np.array) -> Tuple[np.array, np.array]:
        w = w_init
        b = b_init
        m = self.X_train.shape[0]
        y_hat = np.zeros(m)

        dw, db = 0, 0
        for i in range(m):
            y_hat[i] = np.dot(self.X_train[i], w) + b
            dw += (y_hat[i] - self.y_train[i]) * self.X_train[i]
            db += y_hat[i] - self.y_train[i]

        dw = dw / m
        db = db / m
        return dw, db

    def compute_cost(self, w: np.array, b: np.array) -> float:
        m = self.X_train.shape[0]
        y_hat = np.zeros(m)
        cost = 0

        for i in range(m):
            y_hat[i] = np.dot(self.X_train[i], w) + b
            cost += (y_hat[i] - self.y_train[i]) ** 2

        cost = cost / (2 * m)
        return cost

    def plot_cost(self, cost_data: List[float]):
        plt.plot(cost_data)
        plt.show()

    def gradient_descent(self, w_init: np.array, b_init: np.array, alpha: float,
                         iters: int) -> Tuple[np.array, np.array]:
        w = w_init
        b = b_init
        cost_list = []
        for i in range(iters):
            print('Starting iteration {} ...'.format(i))
            cost = self.compute_cost(w, b)
            print('Cost after iteration {}: {}'.format(i, cost))
            cost_list.append(cost)
            dw, db = self.compute_gradient(w_init, b_init)
            w_tmp = w - alpha * dw
            b_tmp = b - alpha * db
            w = w_tmp
            b = b_tmp

        self.plot_cost(cost_list)
        return (w, b)

    def train(self, w_init: np.array, b_init: np.array, learning_rate: float, iterations: int):
        self.w_init = w_init
        self.b_init = b_init
        self.alpha = learning_rate
        self.iters = iterations

        w_min, b_min = self.gradient_descent(w_init, b_init, learning_rate, iterations)
        print('final output: {}, {}'.format(w_min, b_min))

    def predict(self, a: int):
        print('test')

    def eval(self, x: np.array, y: np.array) -> np.array:
        pass


X_train = np.arange(1, 11) #.reshape(-1, 1)
y_train = X_train ** 2
X_test = np.array([np.arange(11, 16)])
y_test = np.array([X_test ** 2])

# Feature Engineering
X_train = np.c_[X_train, X_train**2, X_train**3]

W_init = np.array(np.zeros(X_train.shape[1]))
b_init = np.array(np.zeros(1))
my_model = LinearRegression(xTrain=X_train, yTrain=y_train, xTest=X_test, yTest=y_test)
my_model.train(w_init=W_init, b_init=b_init, learning_rate=1e-2, iterations=100)
#my_model.predict(10)
