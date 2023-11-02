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

    def apply(self, data: pd.DataFrame) -> typing.List[pd.DataFrame]:
        """
        Converts the given DataFrame into a list of frames.

        :param data: The DataFrame to convert,
        :return: The list of frames.
        """
        windowed_data = []
        start = 0
        end = self.window_size

        while end <= len(data):
            window = data[start:end]
            windowed_data.append(window)

            start += int(self.window_size * (1 - self.overlap))
            end += int(self.window_size * (1 - self.overlap))

        return windowed_data
