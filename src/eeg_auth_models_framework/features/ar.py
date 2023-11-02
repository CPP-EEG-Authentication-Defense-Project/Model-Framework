import pandas as pd
import numpy as np

from statsmodels.tsa import ar_model
from . import base


class ARFeatureExtractor(base.BaseFeatureExtractor):
    """
    Trains AR models on EEG channel data and returns the coefficients of those models as features in a vector.
    """
    def __init__(self, ar_model_config: dict = None):
        self.ar_lag_config = ar_model_config or {}

    def extract(self, frame_channel_data: pd.DataFrame) -> np.ndarray:
        extracted_feature_chunks = []

        for frame_column in frame_channel_data:
            frame_data_column = frame_channel_data[frame_column]
            features = self._get_ar_coefficients(frame_data_column)
            extracted_feature_chunks.append(features)

        return self.flatten_feature_data(extracted_feature_chunks)

    def _get_ar_coefficients(self, data: base.PandasData) -> np.ndarray:
        """
        Helper method which will take the given data series, train an AR model on the data, and then return
        the estimated coefficients.

        :param data: the data to use to generate coefficients.
        :return: the coefficients.
        """
        model = ar_model.AutoReg(data.to_numpy(), **self.ar_lag_config)
        fit_results = model.fit()
        return fit_results.params
