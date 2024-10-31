import numpy as np
from pathlib import Path

from ml.activation_functions import LinearActivationFunction
from ml.models import RegressorModelParameters
from ml.optimizers import GradientDescent


class LinearRegressor(object):
    def __init__(self):
        pass

    @staticmethod
    def train(xTrain: np.ndarray, yTrain: np.ndarray, parameters: RegressorModelParameters,
              optimizer: GradientDescent, activation: LinearActivationFunction,
              learning_rate: float, iterations: int, nodes=1, weights_filepath: Path = None) -> RegressorModelParameters:
        w_init, b_init = parameters.load_initial_weights(filepath=weights_filepath)

        w_min, b_min, cost_list = optimizer.gradient_descent(x=xTrain, y=yTrain, w=w_init, b=b_init,
                                                             nodes=nodes, a=activation, alpha=learning_rate,
                                                             iters=iterations)

        print('Optimum weights and bias are: {}, {}'.format(w_min, b_min))
        parameters.save_weights(weights=w_min, bias=b_min, cost_history=cost_list, filepath=weights_filepath)

        return parameters

    @staticmethod
    def predict(xTest: np.ndarray, parameters: RegressorModelParameters,
                activation: LinearActivationFunction, weights_filepath: Path = None) -> np.array:
        print('Predicting...')
        w_min, b_min = parameters.load_optimum_weights(weights_filepath)
        y_pred = activation.compute_f_x(X=xTest, w=w_min, b=b_min)
        y_pred = (y_pred == y_pred.max(axis=1, keepdims=1)).astype(int)
        return y_pred
