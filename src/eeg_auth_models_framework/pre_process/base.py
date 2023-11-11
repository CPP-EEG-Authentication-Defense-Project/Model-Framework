import abc
import typing
import pandas as pd


class PreProcessStep(abc.ABC):
    def __init__(self, use_original_data=False):
        self.use_original_data = use_original_data

    @abc.abstractmethod
    def apply(self, data: typing.List[pd.DataFrame]) -> typing.List[pd.DataFrame]:
        pass
