import typing
import pandas as pd
import numpy as np

from . import base


class AverageFeatureReduction(base.FeatureReduction):
    """
    Reduces feature data vectors by averaging all features across each feature index (i.e., each column in the matrix).
    """
    def reduce(self, data: typing.List[np.ndarray]) -> np.ndarray:
        dataframe = pd.DataFrame(np.vstack(data))
        average_data = dataframe.mean()
        return average_data.to_numpy()
