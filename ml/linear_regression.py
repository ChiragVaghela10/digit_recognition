from abc import ABC

import numpy as np
from pathlib import Path

from ml.activation_functions import LinearActivationFunction, ActivationFunction, LogisticActivationFunction
from ml.models import RegressorModelParameters
from ml.optimizers import GradientDescent


class Algorithms(ABC):
    def __init__(self):
        pass

    @staticmethod
    def train(self, xTrain: np.ndarray, yTrain: np.ndarray, parameters: RegressorModelParameters,
              optimizer: GradientDescent, activation: ActivationFunction,
              learning_rate: float, iterations: int, nodes=1, weights_filepath: Path = None) -> RegressorModelParameters:
        pass

    @staticmethod
    def predict(self, xTest: np.ndarray, parameters: RegressorModelParameters,
                activation: ActivationFunction, weights_filepath: Path = None) -> np.ndarray:
        pass


class Regressor(Algorithms):
    def train(self, xTrain: np.ndarray, yTrain: np.ndarray, parameters: RegressorModelParameters,
              optimizer: GradientDescent, activation: ActivationFunction,
              learning_rate: float, iterations: int, nodes=1, weights_filepath: Path = None) -> RegressorModelParameters:
        w_init, b_init = parameters.load_initial_weights(filepath=weights_filepath)

        w_min, b_min, cost_list = optimizer.gradient_descent(x=xTrain, y=yTrain, w=w_init, b=b_init,
                                                             nodes=nodes, a=activation, alpha=learning_rate,
                                                             iters=iterations)

        print('Optimum weights and bias are: {}, {}'.format(w_min, b_min))
        parameters.save_weights(weights=w_min, bias=b_min, cost_history=cost_list, filepath=weights_filepath)

        return parameters

    def predict(self, xTest: np.ndarray, parameters: RegressorModelParameters,
                activation: ActivationFunction, weights_filepath: Path = None) -> np.array:
        print('Predicting...')
        w_min, b_min = parameters.load_optimum_weights(weights_filepath)
        y_pred = activation.compute_f_x(X=xTest, w=w_min, b=b_min)
        y_pred = activation.apply_threshold(y_hat=y_pred)
        return y_pred


# class LogisticRegressor(Regressor):
#     def train(self, xTrain: np.ndarray, yTrain: np.ndarray, parameters: RegressorModelParameters,
#               optimizer: GradientDescent, activation: LogisticActivationFunction,
#               learning_rate: float, iterations: int, nodes=1, weights_filepath: Path = None) -> RegressorModelParameters:
#         w_init, b_init = parameters.load_initial_weights(filepath=weights_filepath)
#
#         w_min, b_min, cost_list = optimizer.gradient_descent(x=xTrain, y=yTrain, w=w_init, b=b_init,
#                                                              nodes=nodes, a=activation, alpha=learning_rate,
#                                                              iters=iterations)
#
#         print('Optimum weights and bias are: {}, {}'.format(w_min, b_min))
#         parameters.save_weights(weights=w_min, bias=b_min, cost_history=cost_list, filepath=weights_filepath)
#
#         return parameters
#
#     def predict(self, xTest: np.ndarray, parameters: RegressorModelParameters,
#                 activation: LogisticActivationFunction, weights_filepath: Path = None) -> np.array:
#         print('Predicting...')
#         w_min, b_min = parameters.load_optimum_weights(weights_filepath)
#         y_pred = activation.compute_f_x(X=xTest, w=w_min, b=b_min)
#         y_pred = (y_pred == y_pred.max(axis=1, keepdims=1)).astype(int)
#         return y_pred
