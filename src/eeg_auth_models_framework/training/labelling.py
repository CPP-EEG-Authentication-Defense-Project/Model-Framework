import dataclasses
import typing


D = typing.TypeVar('D')


@dataclasses.dataclass
class LabelledSubjectData(typing.Generic[D]):
    """
    Simple container for labelled data. The data samples are stored in a list and the corresponding labels are stored
    in a separate list.
    """
    data: typing.List[D]
    labels: typing.List[int]


class SubjectDataLabeller(typing.Generic[D]):
    """
    Labeller which generates a labelled dataset tailored to each subject in a list.
    """
    def __init__(self, subjects: typing.List[str]):
        self.subjects = subjects

    def label_data(self, data: typing.Dict[str, typing.List[D]]) -> typing.Dict[str, LabelledSubjectData[D]]:
        """
        Perform labelling on the given data. This will generate a new map with data containers wrapping the labelled
        data.

        :param data: the data to label.
        :return: the labelled dataset.
        """
        labelled_data = {}

        for key in labelled_data:
            labelled_data[key] = self._generate_labels_for_subject(data, key)

        return labelled_data

    @staticmethod
    def _generate_labels_for_subject(data: typing.Dict[str, typing.List[D]], subject: str) -> LabelledSubjectData:
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
