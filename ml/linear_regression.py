import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List

from contants import total_digits, digits_repetition, image_size, trainRatio
from ml.models import Model


class LinearRegression(Model):
    def __init__(self):
        super().__init__()
        self.w_init = None
        self.b_init = None
        self.w_min = None
        self.b_min = None
        self.cost_list = []

    def compute_gradient(self, xTrain: np.array, yTrain: np.array, w: np.array, b: np.array) -> Tuple[np.array, np.array]:
        m = xTrain.shape[0]
        n = xTrain.shape[1]
        y_hat = np.zeros(m).reshape(-1, 1)

        dw, db = np.zeros((n, 10)), np.zeros(10)
        for i in range(m):
            y_hat[i] = np.dot(xTrain[i], w[:, 0]) + b[0]
            dw[:, 0] += (y_hat[i] - yTrain[i, 0]) * xTrain[i]
            db[0] += y_hat[i] - yTrain[i, 0]

        dw[:, 0] = dw[:, 0] / m
        db[0] = db[0] / m
        return dw[:, 0], db[0]

    def compute_cost(self, data: np.array, target: np.array, w: np.array, b: np.array,
                     get_y_pred: bool = False) -> [float, Tuple[np.ndarray, float]]:
        m = data.shape[0]
        cost = np.zeros(10)
        #for j in range(n):
        y_hat = np.zeros(m).reshape(-1, 1)

        for i in range(m):
            y_hat[i] = np.dot(data[i], w[:, 0]) + b[0]
            cost[0] += (y_hat[i] - target[i, 0]) ** 2

        cost[0] = cost[0] / (2 * m)
        if get_y_pred:
            return y_hat, cost[0]
        else:
            return cost[0]

    def plot_cost(self, cost_data: List[float]):
        plt.plot(cost_data)
        plt.show()

    def plot_result(self, pred: np.array, target: np.array, error: np.array):
        samples = len(target)
        pred = np.round(pred)
        correct_pred = sum(pred[:, 0] == target[:, 0])
        print('Correct Prediction: {},  Total:{}, Accuracy: {}'.format(correct_pred,
                                                                       samples, correct_pred / samples))
        plt.plot(pred, "-r", label='prediction')
        plt.plot(target[:, 0], "-b", label='target')
        plt.legend(loc="upper right")
        plt.show()
        pass

    def gradient_descent(self, xTrain: np.array, yTrain: np.array, w_init: np.array, b_init: np.array, alpha: float,
                         iters: int) -> Tuple[np.array, np.array]:
        w = w_init
        b = b_init
        for i in range(iters):
            print('Starting iteration {} ...'.format(i))
            cost = self.compute_cost(xTrain, yTrain, w, b)
            print('Cost after iteration {}: {}'.format(i, cost))
            self.cost_list.append(cost)
            dw, db = self.compute_gradient(xTrain, yTrain, w, b)
            w_tmp = w[:, 0] - alpha * dw
            b_tmp = b[0] - alpha * db
            w[:, 0] = w_tmp
            b[0] = b_tmp

        self.plot_cost(self.cost_list)
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
        self.plot_result(y_pred, yTest, rmse)


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
        x_norm = (X - self.mu) / self.sigma
        return x_norm


class OneHotEncoder(object):
    def __init__(self):
        pass

    def encode(self, X: np.array, categories: int = None) -> np.array:
        if categories is None:
            categories = np.unique(X).shape[0]
        res = np.eye(categories)[np.array(X).reshape(-1)]
        return res.reshape(list(X.shape) + [categories])

    def decode(self, X):
        pass


with open('img_data.txt', 'r', encoding='ascii') as dataFile:
    mfeat_pix = pd.read_table(dataFile, sep='  ', header=None, engine='python').values
    img_data = mfeat_pix.reshape(total_digits, digits_repetition, image_size)

# Get X_train with (10, 160, 240) shape
X_train = img_data[:, :int(digits_repetition * trainRatio), :]
# Convert X_train to (1600, 240) shape
X_train = X_train.reshape(np.round(total_digits * digits_repetition * trainRatio).astype(int), image_size)
# Get X_test with (10, 40 ,240) shape
X_test = img_data[:, int(digits_repetition * trainRatio):, :]
# Convert X_test to (400, 240) shape
X_test = X_test.reshape(np.round(total_digits * digits_repetition * (1 - trainRatio)).astype(int), image_size)

# y_train_new = y_train.reshape(int(total_digits * digits_repetition * trainRatio), total_digits) # also encodes

encoder = OneHotEncoder()

# Create y_train with 160 repetitions of each digit in order
y_train = np.repeat(np.arange(total_digits), np.round(digits_repetition * trainRatio))
# One hot encode y_train
y_train = encoder.encode(y_train)
# Create y_train with 40 repetitions of each digit in order
y_test = np.repeat(np.arange(total_digits), np.round(digits_repetition * (1- trainRatio)).astype(int))
# One hot encode y_test
y_test = encoder.encode(y_test)

# Normalization
normalizer = Normalizer()
X_train = normalizer.normalize(X_train)
X_test = normalizer.normalize(X_test)

W_init = np.zeros((X_train.shape[1], total_digits))
b_init = np.zeros(total_digits)

linear_regressor = LinearRegression()
w_min, b_min = linear_regressor.train(xTrain=X_train, yTrain=y_train, w_init=W_init, b_init=b_init,
                                      learning_rate=1e-2, iterations=1000)
linear_regressor.predict(xTest=X_test, yTest=y_test, wMin=w_min, bMin=b_min)


#X_train = np.arange(1, 11).reshape(-1, 1)
#y_train = X_train ** 2
#X_test = np.arange(11, 16).reshape(-1, 1)
#y_test = X_test ** 2

# Feature Engineering
#X_train = np.c_[X_train, X_train ** 2, X_train ** 3]
#X_test = np.c_[X_test, X_test ** 2, X_test ** 3]