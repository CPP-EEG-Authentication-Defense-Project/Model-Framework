import typing
import logging
import itertools
import pandas as pd
import numpy as np

from scipy import stats
from .pre_process import PreProcessingPipeline
from .features import FeatureExtractPipeline
from .normalization import NormalizationPipeline, FeatureMetaDataIndex, FeatureMetaData, NormalizationStep
from .reduction import FeatureReduction
from .utils.logging_helpers import LOGGER_NAME, PrefixedLoggingAdapter


_logger = logging.getLogger(LOGGER_NAME)


class DataProcessor:
    """
    This class manages data processing pipelines, abstracting their execution into a single interface.
    """
    def __init__(self,
                 pre_process: PreProcessingPipeline,
                 feature_extraction: FeatureExtractPipeline,
                 normalization: NormalizationPipeline = None,
                 reducer: FeatureReduction = None):
        self.pre_process_steps = pre_process
        self.feature_extraction_steps = feature_extraction
        self.normalization_steps = normalization
        self.reducer = reducer
        self.prefixed_logger = PrefixedLoggingAdapter('[Data Processor]', _logger)

    @property
    def is_metadata_required(self):
        """
        Property which determines whether metadata is required for any potential normalization in data processing.

        :return: A flag indicating whether metadata is required.
        """
        if self.normalization_steps is None:
            return False
        return any(norm_step.metadata_required for norm_step in self.normalization_steps)

    def process(self,
                dataframes: typing.List[pd.DataFrame],
                metadata: FeatureMetaDataIndex = None) -> typing.List[np.ndarray]:
        """
        Utility method which combines the application of pre-processing and feature extraction into
        one callable.

        :param dataframes: The raw Dataframes to be processed.
        :param metadata: Optional metadata to use for normalization.
        :return: The processed data features.
        """
        pre_processed_data = self.apply_pre_process_steps(dataframes)
        feature_data = self.apply_feature_extraction_steps(pre_processed_data)
        if self.normalization_steps:
            if self.is_metadata_required and metadata is None:
                raise ValueError('Metadata is required for normalization.')
            feature_data = self.apply_normalization_steps(feature_data, **{NormalizationStep.METADATA_KEY: metadata})
        if self.reducer:
            feature_data = self.apply_reduction(feature_data)
        return feature_data

    def extract_metadata(self, dataframes: typing.List[pd.DataFrame]) -> FeatureMetaDataIndex:
        """
        Processes the given Dataframes (without any normalization or reduction)
        and extracts metadata from the generated series of feature vectors.

        :param dataframes: The raw Dataframes to be processed.
        :return: The metadata extracted.
        """
        pre_processed_data = self.apply_pre_process_steps(dataframes)
        feature_data = self.apply_feature_extraction_steps(pre_processed_data)
        features_dataframe = self._convert_features_to_dataframe(feature_data)
        return self._get_metadata_index(features_dataframe)

    def apply_pre_process_steps(self, dataframes: typing.List[pd.DataFrame]) -> typing.List[pd.DataFrame]:
        """
        Applies pre-processing steps to the given list of DataFrames, returning a flat
        list of DataFrames as its result.

        :param dataframes: the list of DataFrames to apply pre-processing to.
        :return: the list of DataFrames that have been pre-processed.
        """
        self.prefixed_logger.info(
            f'Applying {len(self.pre_process_steps)} pre-processing steps to {len(dataframes)} frames'
        )
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
        self.prefixed_logger.info(
            f'Applying {len(self.feature_extraction_steps)} feature extraction steps to {len(dataframes)} frames'
        )
        return [self.feature_extraction_steps.run(frame) for frame in dataframes]

    def apply_normalization_steps(self, data: typing.List[np.ndarray], **kwargs) -> typing.List[np.ndarray]:
        """
        Applies normalization steps to the given list of feature vectors, applying a new list of normalized
        feature vectors.

        :param data: The feature data to normalize.
        :return: The normalized feature data.
        """
        if not self.normalization_steps:
            raise ValueError('No normalization steps defined!')
        self.prefixed_logger.info(
            f'Applying {len(self.normalization_steps)} normalization steps to {len(data)} frames'
        )
        return [self.normalization_steps.run(features, **kwargs) for features in data]

    def apply_reduction(self, data: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
        """
        Applies feature reduction to the given list of feature vectors, reducing the vectors down to a single
        vector.

        :param data: The feature vector data to reduce.
        :return: The reduced data.
        """
        if not self.reducer:
            raise ValueError('No reducer defined!')
        self.prefixed_logger.info(f'Applying feature reduction to {len(data)} frames')
        return self.reducer.reduce(data)

    @staticmethod
    def _convert_features_to_dataframe(data: typing.List[np.ndarray]) -> pd.DataFrame:
        """
        Simple utility method wrapping the conversion of the given list of feature vectors to a pandas DataFrame.

        :param data: The feature vectors to convert to a Dataframe.
        :return: The generated Dataframe.
        """
        return pd.DataFrame(
            np.vstack(data)
        )

    def _get_metadata_index(self, features: pd.DataFrame) -> FeatureMetaDataIndex:
        """
        Helper method which generates a feature metadata index from the given feature data,
        transformed into a Dataframe.

        :param features: The feature data to use to generate the index.
        :return: The feature metadata index.
        """
        self.prefixed_logger.info(
            f'Generating metadata for {len(features.axes[0])}x{len(features.axes[1])} features matrix'
        )
        meta_data_index = FeatureMetaDataIndex()

        for feature_idx in features.columns:
            meta_data_index.append(
                self._get_metadata_from_series(
                    features[feature_idx]
                )
            )

        return meta_data_index

    @staticmethod
    def _get_metadata_from_series(features: pd.Series) -> FeatureMetaData:
        """
        Helper method which generates a feature metadata object from the given series data.

        :param features: The feature series data to use to generate the metadata object.
        :return: The metadata object.
        """
        series_data = features.to_numpy()
        return FeatureMetaData(
            std_dev=np.std(series_data),
            median_abs_dev=stats.median_abs_deviation(series_data),
            mean=np.mean(series_data),
            median=np.median(series_data),
            min=np.min(series_data),
            max=np.max(series_data)
        )
