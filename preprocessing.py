from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from contants import total_digits, digits_repetition, image_size, trainRatio


class ImageDataSet(object):
    def __init__(self, filename: Path) -> None:
        self.data = None
        self.filename = filename

    def load(self) -> np.ndarray:
        with open(self.filename, 'r', encoding='ascii') as dataFile:
            self.data = pd.read_table(dataFile, sep='  ', header=None, engine='python').values
            return self.data

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.data = self.data.reshape(total_digits, digits_repetition, image_size)

        # Get X_train with (10, 160, 240) shape
        x_train = self.data[:, :int(digits_repetition * trainRatio), :]
        # Convert X_train to (1600, 240) shape
        x_train = x_train.reshape(np.round(total_digits * digits_repetition * trainRatio).astype(int), image_size)
        # Get X_test with (10, 40 ,240) shape
        x_test = self.data[:, int(digits_repetition * trainRatio):, :]
        # Convert X_test to (400, 240) shape
        x_test = x_test.reshape(np.round(total_digits * digits_repetition * (1 - trainRatio)).astype(int), image_size)
        # Create y_train with 160 repetitions of each digit in order
        y_train = np.repeat(np.arange(total_digits), np.round(digits_repetition * trainRatio))
        # Create y_train with 40 repetitions of each digit in order
        y_test = np.repeat(np.arange(total_digits), np.round(digits_repetition * (1 - trainRatio)).astype(int))

        return x_train, x_test, y_train, y_test

