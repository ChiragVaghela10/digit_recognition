import numpy as np
import matplotlib.pyplot as plt

from constants import *


def shuffle_dataset(data: np.array, labels: np.array) -> (np.array, np.array):
    """
    Shuffles the dataset and labels.

    data: dataset to be shuffled.
    labels: labels to be shuffled.
    """
    np.random.seed(42)  # Ensuring reproducibility
    shuffled_indices = np.random.permutation(len(data))
    data = data[shuffled_indices]
    labels = labels[shuffled_indices]
    return data, labels


def split_dataset(data: np.array, labels: np.array, split_ratio: float):
    """
    Splits dataset into train and test sets.

    data: dataset to be split.
    labels: labels to be split.
    split_ratio: proportion of the dataset to be split.
    """
    train_size = int(len(data) * split_ratio)
    train_data, test_data = data[:train_size], data[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]
    return train_data, train_labels, test_data, test_labels


def plot_samples(data: np.array, labels: np.array, num_per_class: int) -> None:
    """Plots sample images from the dataset."""
    fig, axes = plt.subplots(num_per_class, total_digits, figsize=(10, 10))
    for digit in range(total_digits):
        for j in range(num_per_class):
            pic = data[labels == digit][j].reshape(16, 15)
            axes[j, digit].imshow(-pic, cmap='gray')
            axes[j, digit].axis('off')
    plt.show()

