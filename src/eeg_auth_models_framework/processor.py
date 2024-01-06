import typing
import logging
import itertools
import pandas as pd
import numpy as np

from .pre_process import PreProcessingPipeline
from .features import FeatureExtractPipeline
from .normalization import NormalizationPipeline
from .utils.logging_helpers import LOGGER_NAME


_logger = logging.getLogger(LOGGER_NAME)


class DataProcessor:
    """
    This class manages data processing pipelines, abstracting their execution into a single interface.
    """
    def __init__(self,
                 pre_process: PreProcessingPipeline,
                 feature_extraction: FeatureExtractPipeline,
                 normalization: NormalizationPipeline = None):
        self.pre_process_steps = pre_process
        self.feature_extraction_steps = feature_extraction
        self.normalization_steps = normalization

    def process(self, dataframes: typing.List[pd.DataFrame]) -> typing.List[np.ndarray]:
        """
        Utility method which combines the application of pre-processing and feature extraction into
        one callable.

        :param dataframes: The raw Dataframes to be processed.
        :return: The processed data features.
        """
        pre_processed_data = self.apply_pre_process_steps(dataframes)
        feature_data = self.apply_feature_extraction_steps(pre_processed_data)
        if self.normalization_steps:
            feature_data = self.apply_normalization_steps(feature_data)
        return feature_data

    def apply_pre_process_steps(self, dataframes: typing.List[pd.DataFrame]) -> typing.List[pd.DataFrame]:
        """
        Applies pre-processing steps to the given list of DataFrames, returning a flat
        list of DataFrames as its result.

        :param dataframes: the list of DataFrames to apply pre-processing to.
        :return: the list of DataFrames that have been pre-processed.
        """
        pre_process_results = []

        for frame in dataframes:
            pre_process_results.append(self.pre_process_steps.run(frame))

        return list(itertools.chain.from_iterable(pre_process_results))

    def apply_feature_extraction_steps(self, dataframes: typing.List[pd.DataFrame]) -> typing.List[np.ndarray]:
        """
        Applies feature extraction steps to the given list of DataFrames, returning a flat
        list of numpy arrays as its result.

        :param dataframes: the list of DataFrames to extract feature data from.
        :return: a list of numpy arrays containing feature data.
        """
        return [self.feature_extraction_steps.run(frame) for frame in dataframes]

    def apply_normalization_steps(self, data: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
        """
        Applies normalization steps to the given list of feature vectors, applying a new list of normalized
        feature vectors.

        :param data: The feature data to normalize.
        :return: The normalized feature data.
        """
        if not self.normalization_steps:
            raise ValueError('No normalization steps defined!')
        return [self.normalization_steps.run(features) for features in data]
