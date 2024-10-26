import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from contants import total_digits, digits_repetition, image_size, trainRatio, colors
from ml.helpers import OneHotEncoder, Normalizer


class LinearRegression(object):
    def __init__(self, classes=1):
        super().__init__()
        self.classes = classes
        self.cost_list = None
        self.w_init = None
        self.b_init = None
        self.w_min = None
        self.b_min = None

    @staticmethod
    def plot_digit(img_data: np.array) -> None:
        plt.imshow(img_data.reshape(16, 15))
        plt.show()

    @staticmethod
    def plot_cost(cost: np.array):
        classes = cost.shape[1]
        for cls in range(classes):
            plt.plot(cost[:, cls], "{}".format(colors[cls]), label="Class: {}".format(cls))
        plt.legend(loc="upper right")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost vs Iterations")
        plt.savefig("results/lr_cost.png")
        plt.close()

    @staticmethod
    def plot_result(pred: np.array, target: np.array):
        samples = len(target)
        pred = np.round(pred).astype(int)
        table_data = np.zeros((10, 3))
        for cls in range(pred.shape[1]):
            correct_pred = sum(target[:, cls] == pred[:, cls])
            accuracy = correct_pred / samples
            print("Class: {}, Correct Predictions: {} / 400, Accuracy: {}".format(cls, correct_pred, accuracy))
            table_data[cls] = np.array([cls, correct_pred, accuracy])
        cols = ["Class", "Correct Predictions", "Accuracy"]
        rows = ['Class: %d' % x for x in (range(10))]

        # hides background axes
        plt.axis('off')

        plt.table(cellText=table_data, rowLabels=rows, colLabels=cols, loc='center')
        plt.savefig("results/lr_result.png")
        plt.close()

    @staticmethod
    def save_weights(weights: np.ndarray, bias: np.array, cost: np.ndarray) -> None:
        with open('weights/lr_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)
            pickle.dump(bias, f)
            pickle.dump(cost, f)
            print('Saved model weights')

    @staticmethod
    def load_weights() -> Tuple[np.ndarray, np.array, np.ndarray]:
        with open('weights/lr_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
            bias = pickle.load(f)
            cost = pickle.load(f)
            print('Loaded model weights')
            return weights, bias, cost

    def compute_gradient(self, xTrain: np.array, yTrain: np.array, w: np.array,
                         b: np.array) -> Tuple[np.array, np.array]:
        samples = xTrain.shape[0]
        features = xTrain.shape[1]
        dw = np.zeros((features, self.classes))
        db = np.zeros(self.classes)

        for cls in range(self.classes):
            y_hat = np.zeros(samples).reshape(-1, 1)
            for sample in range(samples):
                y_hat[sample] = np.dot(xTrain[sample], w[:, cls]) + b[cls]
                dw[:, cls] += (y_hat[sample] - yTrain[sample, cls]) * xTrain[sample]
                db[cls] += y_hat[sample] - yTrain[sample, cls]

            dw[:, cls] = dw[:, cls] / samples
            db[cls] = db[cls] / samples
        return dw, db

    def compute_cost(self, data: np.array, target: np.array, w: np.array, b: np.array,
                     get_y_pred: bool = False) -> [np.ndarray, np.ndarray]:
        samples = data.shape[0]
        cost = np.zeros(self.classes)
        y_hat = np.zeros((samples, self.classes))

        for cls in range(self.classes):
            for sample in range(samples):
                y_hat[sample, cls] = np.dot(data[sample], w[:, cls]) + b[cls]
                cost[cls] += (y_hat[sample, cls] - target[sample, cls]) ** 2

            cost[cls] = cost[cls] / (2 * samples)
        if get_y_pred:
            return y_hat
        else:
            return cost

    def gradient_descent(self, xTrain: np.array, yTrain: np.array, w_init: np.array, b_init: np.array, alpha: float,
                         iters: int) -> Tuple[np.array, np.array, np.array]:
        self.cost_list = np.zeros((iters, self.classes))
        w = w_init
        b = b_init

        for i in range(iters):
            print('Starting iteration {} ...'.format(i))
            cost = self.compute_cost(xTrain, yTrain, w, b)
            print('Cost after iteration {}: {}'.format(i, cost))
            self.cost_list[i] = cost
            dw, db = self.compute_gradient(xTrain, yTrain, w, b)
            w_tmp = w - alpha * dw
            b_tmp = b - alpha * db
            w = w_tmp
            b = b_tmp

        return w, b, self.cost_list

    def train(self, xTrain: np.array, yTrain: np.array, w_init: np.array, b_init: np.array,
              learning_rate: float, iterations: int) -> Tuple[np.array, np.array, np.array]:
        self.w_init = w_init
        self.b_init = b_init

        self.w_min, self.b_min, self.cost_list = self.gradient_descent(xTrain, yTrain, w_init, b_init, learning_rate, iterations)
        print('Optimum weights and bias are: {}, {}'.format(self.w_min, self.b_min))

        return self.w_min, self.b_min, self.cost_list

    def predict(self, xTest: np.array, yTest: np.array, wMin: np.array, bMin: float) -> Tuple[np.array, np.array]:
        print('Predicting...')
        return self.compute_cost(data=xTest, target=yTest, w=wMin, b=bMin, get_y_pred=True)


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
y_test = np.repeat(np.arange(total_digits), np.round(digits_repetition * (1 - trainRatio)).astype(int))
# One hot encode y_test
y_test = encoder.encode(y_test)

# Normalization
normalizer = Normalizer()
X_train = normalizer.normalize(X_train)
X_test = normalizer.normalize(X_test)

W_init = np.zeros((X_train.shape[1], total_digits))
b_init = np.zeros(total_digits)

linear_regressor = LinearRegression(total_digits)
w_min, b_min, cost_data = linear_regressor.train(xTrain=X_train, yTrain=y_train,
                                                 w_init=W_init, b_init=b_init, learning_rate=1e-2, iterations=1000)
linear_regressor.save_weights(weights=w_min, bias=b_min, cost=cost_data)

w_min, b_min, cost_data = linear_regressor.load_weights()
linear_regressor.plot_cost(cost=cost_data)
y_pred = linear_regressor.predict(xTest=X_test, yTest=y_test, wMin=w_min, bMin=b_min)
linear_regressor.plot_result(pred=y_pred, target=y_test)

digit = 7
random_sample = normalizer.normalize(img_data[digit, 129, :].reshape(-1, 240))
linear_regressor.plot_digit(random_sample)
test_target = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],])
predicted_digit = linear_regressor.predict(xTest=random_sample, yTest=test_target, wMin=w_min, bMin=b_min)
predicted_digit = np.argmax(np.round(predicted_digit).astype(int))
print("Predicted digit: {} and actual digit: {}".format(predicted_digit, digit))
