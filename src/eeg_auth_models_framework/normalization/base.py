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


class FeatureMetaDataIndex(typing.List[FeatureMetaData]):
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
    METADATA_KEY = 'metadata'

    def apply_normalization(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Executes normalization procedure, with the additional step of validating whether metadata was passed in
        (if it is required).

        :param data: The data to normalize.
        :return: The normalized data.
        """
        if self.metadata_required and self.METADATA_KEY not in kwargs:
            raise ValueError(
                f'{self.__class__.__name__} requires metadata to be passed in under the name "{self.METADATA_KEY}" to'
                f'perform normalization.'
            )
        return self.normalize(data, **kwargs)

    @abc.abstractmethod
    def normalize(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies normalization to the given feature vector.

        :param data: the original feature vector to normalize.
        :return: the normalized feature vector.
        """
        pass

    @property
    @abc.abstractmethod
    def metadata_required(self) -> bool:
        """
        Property indicating whether metadata is needed to use this normalization step. If this property is true, and
        metadata is not provided a runtime, then this will trigger errors to be thrown.

        :return: A flag indicating whether metadata is required.
        """
        pass
