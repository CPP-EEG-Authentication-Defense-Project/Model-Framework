import abc
import typing
import pandas as pd


class PreProcessStep(abc.ABC):
    @abc.abstractmethod
    def apply(self, data: typing.List[pd.DataFrame]) -> typing.List[pd.DataFrame]:
        pass
