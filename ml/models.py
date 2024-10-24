from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Model(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self) -> np.array:
        pass
