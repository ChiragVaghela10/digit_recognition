from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Model(ABC):
    @abstractmethod
    def __init__(self, xTrain: pd.DataFrame, yTrain: pd.DataFrame, xTest: pd.DataFrame, yTest: pd.DataFrame,
                 iters: int):
        self.X_train = xTrain
        self.y_train = yTrain
        self.X_test = xTest
        self.y_test = yTest
        self.iterations = iters

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def eval(self) -> pd.DataFrame:
        pass


class LinearRegression(Model):
    def __init__(self, xTrain: pd.DataFrame, yTrain: pd.DataFrame, xTest: pd.DataFrame, yTest: pd.DataFrame,
                 w_init: pd.DataFrame, b_init: pd.DataFrame, iterations: int, alpha: float):
        super().__init__(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest, iters=iterations)
        self.w_init = w_init
        self.b_init = b_init
        self.alpha = alpha
        self.train()

    def compute_gradient(self):
        pass

    def gradient_descent(self):
        pass

    def train(self):
        pass

    def predict(self, a: int):
        print('test')

    def eval(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        pass


X_train = pd.DataFrame(np.arange(1, 11))
y_train = X_train ** 2
X_test = pd.DataFrame(np.arange(11, 16))
y_test = X_test ** 2
W_init = pd.DataFrame(np.zeros(X_train.shape[1]))
b_init = pd.DataFrame(np.zeros(1))
my_model = LinearRegression(xTrain=X_train, yTrain=y_train, xTest=X_test, yTest=y_test, w_init=W_init,
                            b_init=b_init, alpha=1e-2, iterations=100)
my_model.predict(10)

