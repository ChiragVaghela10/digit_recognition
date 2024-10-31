from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from contants import colors


def plot_digit(img_data: np.array) -> None:
    plt.title("Sample Digit")
    plt.imshow(img_data.reshape(16, 15))
    plt.show()


def plot_cost(filepath: Path, cost: np.array):
    classes = cost.shape[1]
    for node in range(classes):
        plt.plot(cost[:, node], "{}".format(colors[node]), label="Class: {}".format(node))
    plt.legend(loc="upper right")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.savefig(filepath)
    plt.close()


def plot_result(filepath: Path, pred: np.array, target: np.array):
    samples = len(target)
    table_data = np.zeros((10, 3))

    for node in range(pred.shape[1]):
        correct_pred = sum(target[:, node] == pred[:, node])
        accuracy = correct_pred / samples

        table_data[node] = np.array([node, correct_pred, accuracy])
        print("Class: {}, Correct Predictions: {} / 400, Accuracy: {}".format(node, correct_pred, accuracy))

    plt.title("Digit Prediction Accuracy")
    cols = ["Class", "Correct Predictions (out of 400)", "Accuracy"]
    rows = ['Class: %d' % x for x in (range(10))]

    # hides background axes
    plt.axis('off')

    plt.table(cellText=table_data, rowLabels=rows, colLabels=cols, loc='center')
    plt.savefig(filepath)
    plt.close()
