import abc
import typing
import numpy as np


class FeatureDataReducer(abc.ABC):
    """
    Base interface for a feature reduction processor. The purpose of such objects is to take in a series of feature
    data vectors and reduce them into a single feature vector.
    """
    @abc.abstractmethod
    def reduce(self, data: typing.Iterable[np.ndarray]) -> np.ndarray:
        """
        Reduces the given series of feature vectors to a single feature vector.

        :param data: The data to reduce.
        :return: The reduced feature vector.
        """
        pass
