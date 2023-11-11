import abc
import typing

import pandas as pd
import numpy as np


PandasData = typing.Union[pd.Series, pd.DataFrame]
NumberResult = typing.Union[int, float]


class FeatureExtractor(abc.ABC):
    """
    Base for feature extractor classes to inherit from. Defines the common interface for interacting with
    the feature extractors.
    """
    @abc.abstractmethod
    def extract(self, frame_channel_data: pd.DataFrame) -> np.ndarray:
        """
        Extracts features as a flat numpy array.

        :param frame_channel_data: a Pandas dataframe representing the EEG data, wherein each column corresponds to data
                                   from a specific channel.
        :return: the numpy array of features extracted from the data.
        """
        pass

    @staticmethod
    def flatten_feature_data(data: typing.List[np.ndarray]) -> np.ndarray:
        """
        Helper method which flattens lists of extracted features. This is helpful for when the features extracted
        may not be of the same length, as this will raise errors in numpy if the numpy flatten() method is used.

        :param data: the data to flatten.
        :return: a flattened feature vector.
        """
        flattened_data = []

        for feature_vector in data:
            for element in feature_vector:
                flattened_data.append(element)

        return np.array(flattened_data)
