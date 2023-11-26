import typing
import pandas as pd

from .base import PreProcessStep


class PreProcessingPipeline(list[PreProcessStep]):
    """
    Specialized list which manages applying pre-processing steps to a DataFrame.
    """
    def pre_process(self, data: pd.DataFrame) -> typing.List[pd.DataFrame]:
        """
        Executes pre-processing steps on the given DataFrame, returning a list of DataFrames generated from
        processing.

        :param data: the data to apply pre-processing steps to.
        :return: the list of processed DataFrame instances.
        """
        dataframes = [data]

        for step in self:
            dataframes = step.apply(dataframes)

        return dataframes
