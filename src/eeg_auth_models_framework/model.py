import abc
import typing
import itertools
import pandas as pd
import numpy as np

from .pre_process.base import PreProcessStep
from .features.base import FeatureExtractor
from .data.base import DatasetDownloader, DatasetReader
from .training.base import DataLabeller, StratifiedSubjectData
from .training.labelling import SubjectDataStratificationHandler


M = typing.TypeVar('M')


class ModelBuilder(abc.ABC, typing.Generic[M]):
    """
    Base model builder class used to abstract the process of constructing the pre-processing and feature extraction
    necessary to construct an EEG authentication model.
    """
    def __init__(self,
                 pre_process: typing.List[PreProcessStep] = None,
                 feature_extraction: typing.List[FeatureExtractor] = None):
        self.pre_process_steps = pre_process or []
        self.feature_extraction_steps = feature_extraction or []

    @property
    @abc.abstractmethod
    def data_downloader(self) -> DatasetDownloader:
        """
        Property representing the training dataset downloader to be used for initiating model training.

        :return: the downloader instance.
        """
        pass

    @property
    @abc.abstractmethod
    def data_reader(self) -> DatasetReader:
        """
        Property representing the training dataset reader to be used to format training data.

        :return: the reader instance.
        """
        pass

    @property
    @abc.abstractmethod
    def labeller(self) -> DataLabeller:
        pass

    @abc.abstractmethod
    def run_training(self, labelled_data: typing.Dict[str, typing.List[StratifiedSubjectData]]) -> M:
        """
        Executes model training, returning the final model results.

        :param labelled_data: the labelled training data to use for training.
        :return: the trained model results.
        """
        pass

    def pre_process(self, data: pd.DataFrame) -> typing.List[pd.DataFrame]:
        """
        Executes pre-processing steps on the given DataFrame, returning a list of DataFrames generated from
        processing.

        :param data: the data to apply pre-processing steps to.
        :return: the list of processed DataFrame instances.
        """
        dataframes = [data]

        for step in self.pre_process_steps:
            dataframes = step.apply(dataframes)

        return dataframes

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Executes feature extraction steps on the given DataFrame, returning a numpy array of feature data.

        :param data: the DataFrame to extract features from.
        :return: a numpy array of feature data.
        """
        feature_components = [extractor.extract(data) for extractor in self.feature_extraction_steps]
        return np.array(list(itertools.chain.from_iterable(feature_components)))

    def train(self, k_folds=10) -> M:
        """
        Initiates the process of training an authentication model using the current configuration.

        :return: the trained model results.
        """
        data_path = self.data_downloader.retrieve()
        training_data = self.data_reader.format_data(data_path)
        for subject, data in training_data.items():
            training_data[subject] = self._apply_pre_process_steps(data)
        for subject, data in training_data.items():
            training_data[subject] = self._apply_feature_extraction_steps(data)
        labelled_data = self.labeller.label_data(training_data)
        stratification_handler = SubjectDataStratificationHandler(k_folds)
        stratified_data = stratification_handler.get_k_folds_data(labelled_data)
        return self.run_training(stratified_data)

    def _apply_pre_process_steps(self, dataframes: typing.List[pd.DataFrame]) -> typing.List[pd.DataFrame]:
        """
        Helper method which applies pre-processing steps to the given list of DataFrames, returning a flat
        list of DataFrames as its result.

        :param dataframes: the list of DataFrames to apply pre-processing to.
        :return: the list of DataFrames that have been pre-processed.
        """
        pre_process_results = []

        for frame in dataframes:
            pre_process_results.append(self.pre_process(frame))

        return list(itertools.chain.from_iterable(pre_process_results))

    def _apply_feature_extraction_steps(self, dataframes: typing.List[pd.DataFrame]) -> typing.List[np.ndarray]:
        """
        Helper method which applies feature extraction steps to the given list of DataFrames, returning a flat
        list of numpy arrays as its result.

        :param dataframes: the list of DataFrames to extract feature data from.
        :return: a list of numpy arrays containing feature data.
        """
        return [self.extract_features(frame) for frame in dataframes]
