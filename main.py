import numpy as np
import matplotlib.pyplot as plt

from constants import *
from kNN.algorithms import knn_accuracies
from kNN.utilities import plot_samples, split_dataset, shuffle_dataset, plot_clusters, plot_accuracies

with open('data/mfeat-pix.txt', 'r', encoding='ascii') as dataFile:
    mfeat_pix = np.loadtxt('data/mfeat-pix.txt', dtype=int)

labels = np.repeat(np.arange(total_digits), digits_per_class)

# Shuffle the dataset
dataset, labels = shuffle_dataset(mfeat_pix, labels)

# Dataset split
train_data, train_labels, test_data, test_labels = split_dataset(data=dataset, labels=labels, split_ratio=train_ratio)

plot_clusters(train_data, train_labels)
plot_samples(data=train_data, labels=train_labels, num_per_class=10)

k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
accuracies, max_acc, optimal_k = knn_accuracies(train_data=train_data, train_labels=train_labels, test_data=test_data,
                                                test_labels=test_labels, k_values=k_values)

plot_accuracies(accuracies, max_acc, optimal_k)
