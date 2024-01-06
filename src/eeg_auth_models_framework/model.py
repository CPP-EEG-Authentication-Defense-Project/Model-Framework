import abc
import typing
import logging
import numpy.typing as np_types

from .data.base import DatasetDownloader, DatasetReader
from .training.base import DataLabeller, StratifiedSubjectData
from .training.labelling import SubjectDataStratificationHandler
from .training.results import TrainingResult
from .processor import DataProcessor
from .utils.logging_helpers import PrefixedLoggingAdapter, LOGGER_NAME

_logger = logging.getLogger(LOGGER_NAME)
M = typing.TypeVar('M')


class ModelBuilder(abc.ABC, typing.Generic[M]):
    """
    Base model builder class used to abstract the process of constructing the pre-processing and feature extraction
    necessary to construct an EEG authentication model.
    """
    def __init__(self,
                 data_downloader: DatasetDownloader,
                 data_reader: DatasetReader,
                 data_labeller: DataLabeller,
                 data_processor: DataProcessor):
        self.data_downloader = data_downloader
        self.data_reader = data_reader
        self.data_labeller = data_labeller
        self.data_processor = data_processor

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

    def train(self, k_folds=10) -> TrainingResult[M]:
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
        training_logger.info('Processing data')
        for subject, data in training_data.items():
            training_data[subject] = self.data_processor.process(data)
        training_logger.info('Labelling data')
        labelled_data = self.data_labeller.label_data(training_data)
        training_logger.info('Applying stratification')
        stratification_handler = SubjectDataStratificationHandler(k_folds)
        stratified_data = stratification_handler.get_k_folds_data(labelled_data)
        training_logger.info('Training model(s)')
        results = self.run_training(stratified_data)
        training_logger.info('Training complete')
        return results
