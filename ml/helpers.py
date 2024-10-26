import numpy as np


class Normalizer(object):
    def __init__(self):
        self.mu = None
        self.sigma = None

    def normalize(self, X) -> np.array:
        if self.mu is None:
            self.mu = np.mean(X, axis=0)
        if self.sigma is None:
            self.sigma = np.std(X, axis=0)
        print("mu and sigma are: {}, {}".format(self.mu, self.sigma))
        x_norm = (X - self.mu) / self.sigma
        return x_norm


class OneHotEncoder(object):
    def __init__(self, categories: int = None):
        self.categories = categories

    def encode(self, X: np.array, categories: int = None) -> np.array:
        if categories:
            self.categories = categories

        if self.categories is None:
            categories = np.unique(X).shape[0]
        res = np.eye(categories)[np.array(X).reshape(-1)]

        return res.reshape(list(X.shape) + [categories])

    def decode(self, X):
        pass
