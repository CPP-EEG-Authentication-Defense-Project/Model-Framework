import abc
import typing
import itertools
import logging
import pandas as pd
import numpy as np
import numpy.typing as np_types

from .pre_process.base import PreProcessStep
from .features.base import FeatureExtractor
from .data.base import DatasetDownloader, DatasetReader
from .training.base import DataLabeller, StratifiedSubjectData
from .training.labelling import SubjectDataStratificationHandler
from .training.results import TrainingResult
from .utils.logging_helpers import PrefixedLoggingAdapter


LOGGER_NAME = 'eeg-auth-models:model'
_logger = logging.getLogger(LOGGER_NAME)
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
    def create_classifier(self) -> M:
        """
        Creates a fresh classifier instance, to be used for training models.

        :return: the model instance.
        """
        pass

    @abc.abstractmethod
    def train_classifier(self, model: M, x_data: np_types.ArrayLike, y_data: np_types.ArrayLike):
        """
        Executes the training routine on a given model with the given data.

        :param model: the model to train.
        :param x_data: the X input data.
        :param y_data: the Y expected output data.
        """
        pass

    @abc.abstractmethod
    def score_classifier(self, model: M, x_data: np_types.ArrayLike, y_data: np_types.ArrayLike) -> float:
        """
        Executes the scoring routine on a given model with the given data.

        :param model: the model to train.
        :param x_data: the X input data.
        :param y_data: the Y expected output data.
        :returns: the overall score.
        """
        pass

    def run_training(self, labelled_data: typing.Dict[str, typing.List[StratifiedSubjectData]]) -> TrainingResult[M]:
        """
        Executes model training, returning the final model results.

        :param labelled_data: the labelled training data to use for training.
        :return: the trained model results.
        """
        subject_models: typing.Dict[str, M] = {
            subject: self.create_classifier()
            for subject in labelled_data
        }
        training_scores: typing.Dict[str, typing.List[float]] = {
            subject: []
            for subject in labelled_data
        }
        for subject in labelled_data:
            subject_logger = PrefixedLoggingAdapter(f'[subject: {subject}]', _logger)
            subject_logger.info(f'[Subject: {subject}] Training model...')
            stratified_data = labelled_data[subject]
            model = subject_models[subject]
            iteration_count = 1
            for segment in stratified_data:
                subject_logger.info(f'[Subject: {subject}] Running training fold {iteration_count}')
                self.train_classifier(model, segment.train.x, segment.train.y)
                training_scores[subject].append(
                    self.score_classifier(model, segment.test.x, segment.test.y)
                )
                iteration_count += 1
            subject_logger.info(f'[Subject: {subject}] Training complete')
        return TrainingResult(
            subject_models,
            training_scores
        )

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
        training_logger = PrefixedLoggingAdapter('[Model Training]', _logger)
        training_logger.info('Executing training routine')
        training_logger.info('Downloading data')
        data_path = self.data_downloader.retrieve()
        training_logger.info('Formatting data')
        training_data = self.data_reader.format_data(data_path)
        training_logger.info('Running pre-processing steps')
        for subject, data in training_data.items():
            training_data[subject] = self._apply_pre_process_steps(data)
        training_logger.info('Running feature extraction steps')
        for subject, data in training_data.items():
            training_data[subject] = self._apply_feature_extraction_steps(data)
        training_logger.info('Labelling data')
        labelled_data = self.labeller.label_data(training_data)
        training_logger.info('Applying stratification')
        stratification_handler = SubjectDataStratificationHandler(k_folds)
        stratified_data = stratification_handler.get_k_folds_data(labelled_data)
        training_logger.info('Training model(s)')
        results = self.run_training(stratified_data)
        training_logger.info('Training complete')
        return results

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
