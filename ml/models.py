from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Model(ABC):
    def __init__(self, xTrain: np.array, yTrain: np.array, xTest: np.array, yTest: np.array) -> None:
        self.X_train = xTrain
        self.y_train = yTrain
        self.X_test = xTest
        self.y_test = yTest

    @abstractmethod
    def train(self, iterations: int) -> None:
        pass

    @abstractmethod
    def predict(self) -> np.array:
        pass

    @abstractmethod
    def eval(self) -> np.array:
        pass
