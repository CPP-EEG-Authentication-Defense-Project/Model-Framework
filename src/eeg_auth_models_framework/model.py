import abc
import typing
import logging
import time
import numpy.typing as np_types

from .data.base import DatasetDownloader, DatasetReader
from .training.base import DataLabeller, LabelledSubjectData
from .training.labelling import SubjectDataPreparer, StratifiedSubjectData
from .training.results import TrainingResult, TrainingStatistics
from .processor import DataProcessor
from .utils.logging_helpers import PrefixedLoggingAdapter, LOGGER_NAME

_logger = logging.getLogger(LOGGER_NAME)
M = typing.TypeVar('M')
D = typing.TypeVar('D')


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

    # noinspection PyMethodMayBeStatic
    def train_classifier(self, model: M, x_data: np_types.ArrayLike, y_data: np_types.ArrayLike):
        """
        Executes the training routine on a given model with the given data.

        :param model: the model to train.
        :param x_data: the X input data.
        :param y_data: the Y expected output data.
        """
        model.fit(x_data, y_data)

    # noinspection PyMethodMayBeStatic
    def score_classifier(self, model: M, x_data: np_types.ArrayLike, y_data: np_types.ArrayLike) -> float:
        """
        Executes the scoring routine on a given model with the given data.

        :param model: the model to train.
        :param x_data: the X input data.
        :param y_data: the Y expected output data.
        :returns: the overall score.
        """
        return model.score(x_data, y_data)

    def run_training(self, labelled_data: typing.Dict[str, LabelledSubjectData[D]], k_folds: int) -> TrainingResult[M]:
        """
        Executes model training, returning the final model results.

        :param labelled_data: the labelled training data to use for training.
        :param k_folds: the number of k-folds to use during training.
        :return: the trained model results.
        """
        subject_models: typing.Dict[str, M] = {
            subject: self.create_classifier()
            for subject in labelled_data
        }
        training_stats: typing.Dict[str, TrainingStatistics] = {
            subject: TrainingStatistics()
            for subject in labelled_data
        }
        data_preparer = SubjectDataPreparer(k_folds)
        for subject in labelled_data:
            subject_logger = PrefixedLoggingAdapter(f'[subject: {subject}]', _logger)
            subject_logger.info('Starting training process')
            subject_logger.info('Generating stratified dataset')
            training_data = data_preparer.get_data(labelled_data, subject)
            model = subject_models[subject]
            subject_logger.info('Training model...')
            iteration_count = 1
            training_stats[subject].train_start = time.time()
            for segment in training_data.stratified_training_data:
                subject_logger.info(f'Running training fold {iteration_count}')
                self.train_classifier(model, segment.train.x, segment.train.y)
                # TODO: add hook for calibration, to get estimate data that can be used for ROC curve
                #       https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
                #       https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
                training_stats[subject].scores.append(
                    self.score_classifier(model, segment.test.x, segment.test.y)
                )
                iteration_count += 1
            training_stats[subject].train_end = time.time()
            subject_logger.info('Training complete')
        return TrainingResult(
            subject_models,
            training_stats
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
        training_logger.info('Training model(s)')
        results = self.run_training(labelled_data, k_folds)
        training_logger.info('Training complete')
        return results
