import abc
import typing
import itertools
import pandas as pd
import numpy as np

from .pre_process.base import PreProcessStep
from .features.base import FeatureExtractor
from .data.base import DatasetDownloader, DatasetReader


class BaseModel(abc.ABC):
    def __init__(self,
                 pre_process: typing.List[PreProcessStep] = None,
                 feature_extraction: typing.List[FeatureExtractor] = None):
        self._pre_process_steps = pre_process or []
        self._feature_extraction_steps = feature_extraction or []

    def pre_process(self, data: pd.DataFrame) -> typing.List[pd.DataFrame]:
        dataframes = [data]

        for step in self._pre_process_steps:
            dataframes = step.apply(dataframes)

        return dataframes

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        feature_components = [extractor.extract(data) for extractor in self._feature_extraction_steps]
        return np.array(itertools.chain.from_iterable(feature_components))

    def train(self):
        data_path = self.data_downloader.retrieve()
        training_data = self.data_reader.format_data(data_path)
        # TODO: prepare data and perform training (actual model definition/execution will be abstract)

    @property
    @abc.abstractmethod
    def data_downloader(self) -> DatasetDownloader:
        pass

    @property
    @abc.abstractmethod
    def data_reader(self) -> DatasetReader:
        pass
