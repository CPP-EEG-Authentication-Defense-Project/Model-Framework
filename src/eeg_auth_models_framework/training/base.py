import abc
import dataclasses
import typing
import numpy.typing


D = typing.TypeVar('D')


@dataclasses.dataclass
class LabelledSubjectData(typing.Generic[D]):
    """
    Simple container for labelled data. The data samples are stored in a list and the corresponding labels are stored
    in a separate list.
    """
    data: typing.List[D]
    labels: typing.List[int]


@dataclasses.dataclass
class TrainingDataPair:
    """
    Container for x-y related data generated from stratification.
    """
    x: typing.List[numpy.typing.ArrayLike]
    y: numpy.typing.ArrayLike


@dataclasses.dataclass
class StratifiedSubjectData:
    """
    Container for train and test data generated from stratification of subject data.
    """
    train: TrainingDataPair
    test: TrainingDataPair


class DataLabeller(abc.ABC, typing.Generic[D]):
    """
    Base class defining the common interface for data labeller implementations.
    """
    @abc.abstractmethod
    def label_data(self, data: typing.Dict[str, typing.List[D]]) -> typing.Dict[str, LabelledSubjectData[D]]:
        """
        Perform labelling on the given data. This will generate a new map with data containers wrapping the labelled
        data.

        :param data: the data to label.
        :return: the labelled dataset.
        """
        pass
