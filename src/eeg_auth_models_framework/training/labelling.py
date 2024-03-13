import typing
import numpy as np
import numpy.typing as np_types
from sklearn.model_selection import StratifiedKFold
from .base import DataLabeller, LabelledSubjectData, StratifiedSubjectData, StratifiedPair


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


class SubjectDataStratificationHandler(typing.Generic[D]):
    """
    Utility class which helps to produce stratified k-fold data.
    """
    def __init__(self, folds: int):
        self.splitter = StratifiedKFold(folds)

    def get_k_folds_data(self,
                         subject_data: typing.Dict[str, LabelledSubjectData[D]],
                         target_subject: str) -> typing.List[StratifiedSubjectData]:
        """
        Generates a list of stratified k-fold data for the given subject, based on the labelled subject data map.

        :param subject_data: the labelled subject data map to use to produce the k-folds.
        :param target_subject: the subject to use to generate the stratified k-folds data.
        :return: a list containing the subject's k-folds data.
        """
        labelled_data = subject_data[target_subject]
        x_data = np.array(labelled_data.data)
        y_data = np.array(labelled_data.labels)
        k_folds_data = self._generate_subject_splits(x_data, y_data)

        return k_folds_data

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
            train_pair = StratifiedPair(
                x_data[train],
                y_data[train]
            )
            test_pair = StratifiedPair(
                x_data[test],
                y_data[test]
            )
            stratified_data.append(
                StratifiedSubjectData(train_pair, test_pair)
            )

        return stratified_data
