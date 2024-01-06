import numpy as np

from .base import NormalizationStep
from ..utils import pipeline


class NormalizationPipeline(pipeline.DataPipeline[NormalizationStep, np.ndarray, np.ndarray]):
    """
    Specialized list of normalization steps to be run on input data.
    """
    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Executes a sequence of normalization steps on a given data array, returning
        a normalized version of the original array.

        :param data: The array to normalize using all current normalization steps.
        :return: The normalized result.
        """
        normalized_data = data
        for step in self:
            normalized_data = step.normalize(normalized_data)
        return normalized_data
