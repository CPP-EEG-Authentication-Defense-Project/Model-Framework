import abc
import pathlib
import typing
import pandas as pd


SubjectDataMap = typing.Dict[str, pd.DataFrame]


class TrainingDataFormatter(abc.ABC):
    @abc.abstractmethod
    def format_data(self, dataset_path: pathlib.Path) -> SubjectDataMap:
        pass
