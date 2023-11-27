import typing
import mne
import mne.io
import pandas as pd


class MNEDataFrameConverter:
    """
    Utility class which is used to create an object capable of repeated conversions from Pandas DataFrame format
    to MNE data format.
    """
    def __init__(self, sample_frequency: int, channels: typing.List[str] = None):
        self.channels = channels
        self.sample_frequency = sample_frequency

    def convert(self, dataframe: pd.DataFrame) -> mne.io.RawArray:
        """
        Converts the given dataframe to MNE data format.

        :param dataframe: the Pandas DataFrame to convert.
        :return: the MNE format data.
        """
        channels = self.channels or dataframe.columns.values.tolist()
        transposed_dataframe = dataframe.transpose(copy=True)
        data_info = mne.create_info(channels, self.sample_frequency, ch_types='eeg')
        return mne.io.RawArray(transposed_dataframe.to_numpy(), data_info)
