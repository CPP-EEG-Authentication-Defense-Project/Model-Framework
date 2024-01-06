import abc
import dataclasses
import functools
import typing
import numpy as np


@dataclasses.dataclass
class FeatureMetaData:
    """
    Simple container class for meta data about a feature element. This meta data is used to facilitate processing,
    such as quantization.
    """
    std_dev: float
    median_abs_dev: float
    mean: float
    median: float
    min: float
    max: float


AttributeOption = typing.Literal['std_dev', 'median_abs_dev', 'mean', 'median', 'min', 'max']


class FeatureMetaDataIndex(list[FeatureMetaData]):
    """
    Specialized list type used to create an index of feature vector metadata.
    """
    @functools.lru_cache(maxsize=4)
    def get_metadata_vector(self, attr: AttributeOption) -> np.ndarray:
        """
        Generates a 1-D vector of metadata based on the given metadata attribute.

        :param attr: the metadata attribute to create the vector from.
        :return: the vector.
        """
        return np.array([getattr(el, attr) for el in self])


class NormalizationStep(abc.ABC):
    """
    Base class defining the interface for all normalization steps.
    """
    def __init__(self, metadata: FeatureMetaDataIndex):
        self.metadata = metadata

    @abc.abstractmethod
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Applies normalization to the given feature vector.

        :param data: the original feature vector to normalize.
        :return: the normalized feature vector.
        """
        pass
