import typing
import mne
import mne.io
import pandas as pd


class MNEDataFrameConverter:
    """
    Utility class which is used to create an object capable of repeated conversions from Pandas DataFrame format
    to MNE data format.
    """
    def __init__(self, channels: typing.List[str], sample_frequency: int):
        self.channels = channels
        self.sample_frequency = sample_frequency

    def convert(self, dataframe: pd.DataFrame) -> mne.io.RawArray:
        """
        Converts the given dataframe to MNE data format.

        :param dataframe: the Pandas DataFrame to convert.
        :return: the MNE format data.
        """
        transposed_dataframe = dataframe.transpose(copy=True)
        data_info = mne.create_info(self.channels, self.sample_frequency, ch_types='eeg')
        return mne.io.RawArray(transposed_dataframe.to_numpy(), data_info)
