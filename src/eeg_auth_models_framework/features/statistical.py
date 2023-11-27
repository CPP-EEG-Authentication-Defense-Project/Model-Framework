import enum
import functools
import typing

import pandas as pd
import numpy as np

from . import base


class StatisticalFeature(enum.Enum):
    MIN = enum.auto()
    MAX = enum.auto()
    MEAN = enum.auto()
    STD_DEV = enum.auto()
    ZERO_CROSSING_RATE = enum.auto()


class StatisticalFeatureExtractor(base.FeatureExtractor):
    """
    Extracts statistical features from EEG data and generates feature vectors.
    """
    def __init__(self, features: typing.List[StatisticalFeature]):
        self.features = frozenset(features)

    @functools.cached_property
    def feature_extractor_callables(self) -> typing.List[typing.Callable[[base.PandasData], base.NumberResult]]:
        """
        Property which calculates a list of callables which can be used to extract features from Pandas format data.

        :return: the list of callables.
        """
        extractors = []
        if StatisticalFeature.MIN in self.features:
            extractors.append(extract_min)
        if StatisticalFeature.MAX in self.features:
            extractors.append(extract_max)
        if StatisticalFeature.MEAN in self.features:
            extractors.append(extract_mean)
        if StatisticalFeature.STD_DEV in self.features:
            extractors.append(extract_std_dev)
        if StatisticalFeature.ZERO_CROSSING_RATE in self.features:
            extractors.append(extract_zero_crossing_rate)
        return extractors

    def extract(self, frame_channel_data: pd.DataFrame) -> np.ndarray:
        extracted_feature_chunks = []

        for frame_column in frame_channel_data:
            frame_data_column = frame_channel_data[frame_column]
            features = [extractor(frame_data_column) for extractor in self.feature_extractor_callables]
            feature_vector_chunk = np.array(
                features
            )
            extracted_feature_chunks.append(feature_vector_chunk)

        return np.array(extracted_feature_chunks).flatten()


def extract_min(data: base.PandasData) -> base.NumberResult:
    """
    Retrieves the statistical minimum from the given data.

    :param data: the data to retrieve the minimum from.
    :return: the minimum value.
    """
    return data.min()


def extract_max(data: base.PandasData) -> base.NumberResult:
    """
    Retrieves the statistical maximum from the given data.

    :param data: the data to retrieve the maximum from.
    :return: the maximum value.
    """
    return data.max()


def extract_mean(data: base.PandasData) -> base.NumberResult:
    """
    Retrieves the statistical mean from the given data.

    :param data: the data to retrieve the mean from.
    :return: the mean value.
    """
    return data.mean()


def extract_std_dev(data: base.PandasData) -> base.NumberResult:
    """
    Retrieves the statistical standard deviation from the given data.

    :param data: the data to retrieve the standard deviation from.
    :return: the standard deviation.
    """
    return data.std()


def extract_zero_crossing_rate(data: base.PandasData) -> base.NumberResult:
    """
    Retrieves the zero crossing rate from the given data.

    :param data: the data to retrieve the zero crossing rate from.
    :return: the zero crossing rate value.
    """
    return data.agg(_calculate_zero_crossing_rate)


def _calculate_zero_crossing_rate(data_to_process: pd.Series) -> float:
    """
    Helper function which calculates a Zero Crossing Rate (ZCR) for the given
    Pandas Series.

    :param data_to_process: The series to retrieve the ZCR for.
    :return: The ZCR.
    """
    row_array = data_to_process.to_numpy()
    zero_crossings = _count_zero_crossings(row_array)
    return zero_crossings / len(row_array)


def _count_zero_crossings(target_array: np.ndarray) -> int:
    """
    Helper function which counts the number of zero crossings in a given array.

    see: https://stackoverflow.com/a/30281079/13261549

    :param target_array: The array to count zero crossings from.
    :return: The number of zero crossings in the array.
    """
    return ((target_array[:-1] * target_array[1:]) < 0).sum()
