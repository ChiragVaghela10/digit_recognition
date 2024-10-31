from pathlib import Path

import numpy as np

from contants import total_digits, image_size
from ml.linear_regression import LinearRegressor, GradientDescent, LinearActivationFunction
from ml.models import RegressorModelParameters
from ml.plots import plot_cost, plot_result, plot_digit
from ml.statistics import Normalizer, OneHotEncoder
from preprocessing import ImageDataSet

ROOT_DIR = Path(__file__).parent
WEIGHTS_DIR = ROOT_DIR / Path('ml/weights/lr_weights.hp5')

img_data = ImageDataSet(ROOT_DIR / Path('data/mfeat-pix.txt'))
img_data.load()
X_train, X_test, y_train, y_test = img_data.preprocess()

# Normalization
normalizer = Normalizer()
X_train = normalizer.normalize(X_train)
X_test = normalizer.normalize(X_test)

# One Hot Encoding
encoder = OneHotEncoder()
y_train = encoder.encode(y_train)
y_test = encoder.encode(y_test)

# Initial Weights and Bias
W_init = np.zeros((total_digits, X_train.shape[1]))
b_init = np.zeros(total_digits)

# Performing Linear Regression
linear_regressor = LinearRegressor()
gradient_descent = GradientDescent()
activation_function = LinearActivationFunction()
initial_model_paras = RegressorModelParameters(
    w_init=W_init,
    b_init=b_init,
    filepath=WEIGHTS_DIR
)
trained_model_paras = linear_regressor.train(
    xTrain=X_train,
    yTrain=y_train,
    nodes=total_digits,
    parameters=initial_model_paras,
    optimizer=gradient_descent,
    activation=activation_function,
    learning_rate=1e-2,
    iterations=100
)

# Load saved weights to avoid training
# trained_model_paras = RegressorModelParameters()
# trained_model_paras.load_optimum_weights(filepath=WEIGHTS_DIR)

plot_cost(filepath=ROOT_DIR / Path('results/lr_cost.png'), cost=trained_model_paras.load_plot_history())
y_pred = linear_regressor.predict(xTest=X_test, parameters=trained_model_paras,
                                  activation=activation_function)
plot_result(filepath=ROOT_DIR / Path('results/lr_result.png'), pred=y_pred, target=y_test)

# Prediction of random sample
# random_digit = np.random.randint(0, 10)
# random_index = np.random.randint(0, 40)
# random_sample = X_test[(random_digit * 40) + random_index].reshape(-1, image_size)
# plot_digit(random_sample)
# test_target = np.array([np.zeros(10),])
# test_target[0, random_digit] = 1        # Providing target y
# y_pred = linear_regressor.predict(xTest=random_sample, parameters=trained_model_paras, activation=activation_function)
# predicted_digit = np.argmax(np.round(y_pred).astype(int))
# print("Predicted digit: {} and actual digit: {}".format(predicted_digit, random_digit))
