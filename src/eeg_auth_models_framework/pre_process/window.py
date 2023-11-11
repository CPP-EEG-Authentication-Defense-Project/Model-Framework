import typing
import pandas as pd

from .base import PreProcessStep


class DataWindowStep(PreProcessStep):
    """
    Utility class used for generating windowed data.
    """
    def __init__(self, window_size: int, overlap: float = 0, use_original_data=False):
        super().__init__(use_original_data)
        self.window_size = window_size
        self.overlap = overlap

    def apply(self, data: typing.List[pd.DataFrame]) -> typing.List[pd.DataFrame]:
        """
        Expands the given DataFrames into a larger list of frames based on configured window size.

        :param data: The DataFrames to convert,
        :return: The list of frames.
        """
        windowed_frames = []

        for dataframe in data:
            windowed_frames.extend(self._create_windows(dataframe))

        return windowed_frames

    def _create_windows(self, dataframe: pd.DataFrame) -> typing.List[pd.DataFrame]:
        """
        Generates windowed data from the given DataFrame.

        :param dataframe: the DataFrame to create windows from.
        :return: the window frames.
        """
        windowed_data = []
        start = 0
        end = self.window_size

        while end <= len(dataframe):
            window = dataframe[start:end]
            windowed_data.append(window)

            start += int(self.window_size * (1 - self.overlap))
            end += int(self.window_size * (1 - self.overlap))

        return windowed_data
