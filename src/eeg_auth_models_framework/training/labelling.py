import typing
import numpy as np
import numpy.typing as np_types
from sklearn.model_selection import StratifiedKFold, train_test_split
from .base import DataLabeller, LabelledSubjectData, StratifiedSubjectData, TrainingDataPair, SubjectModelTrainingData


D = typing.TypeVar('D')


class SubjectDataLabeller(DataLabeller):
    """
    Labeller which generates a labelled dataset tailored to each subject in a list.
    """
    def label_data(self, data):
        labelled_data = {}

        for key in data:
            labelled_data[key] = self._generate_labels_for_subject(data, key)

        return labelled_data

    @staticmethod
    def _generate_labels_for_subject(data: typing.Dict[str, typing.List[D]], subject: str) -> LabelledSubjectData[D]:
        """
        Helper method which generates labelled data for a specific subject.

        :param data: the original dataset to use as reference.
        :param subject: the subject to generate labelled data for.
        :return: the labelled data.
        """
        if subject not in data:
            raise KeyError(f'Subject "{subject}" not found in data map!')
        label_translation_map = {}
        samples_list = []
        labels_list = []

        for key in data:
            label_id = 1 if key == subject else 0
            label_translation_map[key] = label_id
            for subject_frame_sample in data[key]:
                samples_list.append(subject_frame_sample)
                labels_list.append(label_id)

        return LabelledSubjectData(samples_list, labels_list)


class SubjectDataPreparer(typing.Generic[D]):
    """
    Utility class which helps to produce training/test/validation data.
    """
    def __init__(self, folds: int, validation_set_size: float = 0.2, random_state: typing.Union[int, float] = 42):
        if validation_set_size < 0 or validation_set_size > 1:
            raise ValueError(f'Validation set size must be between 0 and 1 (got {validation_set_size})')
        self.splitter = StratifiedKFold(folds)
        self.validation_set_size = validation_set_size
        self.random_state = random_state

    def get_data(self,
                 subject_data: typing.Dict[str, LabelledSubjectData[D]],
                 target_subject: str) -> SubjectModelTrainingData:
        """
        Generates training/test/validation data to be used for training a specific subject model. The training/test
        data will be split into stratified k-folds, while the validation data will be separated out from the training
        and test data.

        :param subject_data: the labelled subject data map to use to produce the k-folds.
        :param target_subject: the subject to use to generate the stratified k-folds data.
        :return: an object wrapping the training data that was assembled.
        """
        labelled_data = subject_data[target_subject]
        x_data = np.array(labelled_data.data)
        y_data = np.array(labelled_data.labels)
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=self.validation_set_size, random_state=self.random_state
        )
        k_folds_data = self._generate_subject_splits(x_train, y_train)

        return SubjectModelTrainingData(
            stratified_training_data=k_folds_data,
            validation_data=TrainingDataPair(
                x=x_test,
                y=y_test
            )
        )

    def _generate_subject_splits(self,
                                 x_data: np_types.ArrayLike,
                                 y_data: np_types.ArrayLike) -> typing.List[StratifiedSubjectData]:
        """
        Utility method which generates a list of stratified data for the given x-y combination.

        :param x_data: the x data points to use for the splits.
        :param y_data: the y data points to use for the splits.
        :return: a list of data point splits.
        """
        stratified_data = []

        for train, test in self.splitter.split(x_data, y_data):
            train_pair = TrainingDataPair(
                x=x_data[train],
                y=y_data[train]
            )
            test_pair = TrainingDataPair(
                x=x_data[test],
                y=y_data[test]
            )
            stratified_data.append(
                StratifiedSubjectData(train_pair, test_pair)
            )

        return stratified_data
