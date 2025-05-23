import itertools
import pandas as pd
import numpy as np

from .base import FeatureExtractor
from ..utils import pipeline


class FeatureExtractPipeline(pipeline.DataPipeline[FeatureExtractor, pd.DataFrame, np.ndarray]):
    """
    Specialized list which manages applying feature extraction steps to a DataFrame.
    """
    def run(self, data: pd.DataFrame) -> np.ndarray:
        """
        Executes feature extraction steps on the given DataFrame, returning a numpy array of feature data.

        :param data: the DataFrame to extract features from.
        :return: a numpy array of feature data.
        """
        feature_components = [extractor.extract(data) for extractor in self]
        return np.array(list(itertools.chain.from_iterable(feature_components)))
