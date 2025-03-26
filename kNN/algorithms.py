import numpy as np
from decorators.decorators import timer


def eu_distance(pointA, pointB):
    """Calculates the Euclidian distance between two points."""
    return np.sqrt(np.sum((pointA - pointB) ** 2))


@timer
def kNN(train_samples, train_labels, test_samples, k: int = 3):
    """
    This function assign labels to test_samples according to the k nearest neighbors present in train_samples.

    train_samples: numpy array of shape (n_samples, n_features)
    train_labels: numpy array of shape (n_samples,)
    test_samples: numpy array of shape (n_samples, n_features)
    k: int
    """
    predicted_labels = []
    for test_sample in test_samples:
        distances = np.array([eu_distance(test_sample, train_sample) for train_sample in train_samples])

        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = train_labels[k_nearest_indices]

        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predicted_labels.append(predicted_label)
    return np.array(predicted_labels)


def knn_accuracies(train_data, train_labels, test_data, test_labels, k_values):
    best_k, best_accuracy = 0, 0
    accuracy_list = []
    for k in k_values:
        predictions = kNN(train_data, train_labels, test_data, k)

        accuracy = (np.sum(predictions == test_labels) / len(test_labels)) * 100
        accuracy_list.append((k, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    return accuracy_list, best_accuracy, best_k
