from pathlib import Path

import numpy as np

from contants import total_digits
from ml.linear_regression import LinearRegression
from ml.models import RegressorModelParameters
from ml.plots import plot_cost, plot_result
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
W_init = np.zeros((X_train.shape[1], total_digits))
b_init = np.zeros(total_digits)

# Performing Linear Regression
linear_regressor = LinearRegression(total_digits)
# model_paras = RegressorModelParameters(w_init=W_init, b_init=b_init, filepath=WEIGHTS_DIR)
# model_paras = linear_regressor.train(xTrain=X_train, yTrain=y_train, parameters=model_paras, learning_rate=1e-2,
#                                      iterations=100)

# Load Weights from file instead of training every time
model_paras = RegressorModelParameters()
model_paras.load_optimum_weights(filepath=WEIGHTS_DIR)

plot_cost(filepath=ROOT_DIR / Path('results/lr_cost.png'), cost=model_paras.load_plot_history())
y_pred = linear_regressor.predict(xTest=X_test, yTest=y_test, parameters=model_paras)
plot_result(filepath=ROOT_DIR / Path('results/lr_result.png'), pred=y_pred, target=y_test)

# Prediction of random sample
# random_digit = np.random.randint(0, 10, size=1)
# random_index = np.random.randint(0, 200, size=1)
# random_sample = img_data[random_digit, random_index, :].reshape(-1, image_size)
# linear_regressor.plot_digit(random_sample)
# random_sample = normalizer.normalize(random_sample)
# test_target = np.array([np.zeros(10),])
# test_target[0, random_digit] = 1        # Providing target y
# predicted_digit = linear_regressor.predict(xTest=random_sample, yTest=test_target, wMin=w_min, bMin=b_min)
# predicted_digit = np.argmax(np.round(predicted_digit).astype(int))
# print("Predicted digit: {} and actual digit: {}".format(predicted_digit, random_digit))
