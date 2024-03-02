import typing
import pandas as pd
import numpy as np

from . import base


class AverageFeatureReduction(base.FeatureReduction):
    """
    Reduces feature data vectors by averaging all features across each feature index (i.e., each column in the matrix).
    Optionally, a window size can be specified so that the entire set of vectors is not reduced to only one result.
    """
    def __init__(self, window_size: int = None):
        self.window_size = window_size

    def reduce(self, data: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
        if self.window_size is None:
            return [self._perform_average_reduction(data)]
        window_to_reduce = []
        results_set = []
        for vector in data:
            window_to_reduce.append(vector)
            if len(window_to_reduce) >= self.window_size:
                results_set.append(self._perform_average_reduction(window_to_reduce))
                window_to_reduce = []
        if len(window_to_reduce) > 0:
            results_set.append(self._perform_average_reduction(window_to_reduce))
        return results_set

    @staticmethod
    def _perform_average_reduction(data: typing.List[np.ndarray]) -> np.ndarray:
        """
        Reduces the given set of vectors to a single feature vector, using the average of each "column" or index
        in the vector.

        :param data: The set of vectors to reduce.
        :return: The reduced vector.
        """
        dataframe = pd.DataFrame(np.vstack(data))
        average_data = dataframe.mean()
        return average_data.to_numpy()
