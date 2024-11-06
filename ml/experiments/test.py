import numpy as np
from matplotlib import pyplot as plt


class ExpLearningOfPredictedValues:
    def __init__(self):
        self.data = np.zeros((1600, 100))

    def add_data(self, y_hat: np.array):
        self.data = np.c_[self.data, y_hat[:, 0]]

    def plot_learning(self) -> None:
        for i in range(100, 200):
            if i % 10 == 0:
                plt.plot(self.data[:500, i], label=f'iteration: {i-100}')
                plt.legend(loc="upper right")
        plt.title("Prediction of samples for digit '0' in training set vs iterations")
        plt.xlabel("Samples (first 500 out of 1600 samples)")
        plt.ylabel("Prediction (value representing probability of digit being '0')")
        plt.savefig(f'ml/experiments/plots/learning_plot.png')
        plt.close()
