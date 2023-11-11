import copy
import dataclasses
import typing
import mne.io
import pandas as pd
import numpy as np

from .base import PreProcessStep


RawMNE = typing.Union[mne.io.Raw, mne.io.RawArray]
RawFrequencyDataMap = typing.Dict[str, RawMNE]


@dataclasses.dataclass
class FrequencyBand:
    """
    Dataclass representing a target EEG data frequency band.
    """
    lower: typing.Optional[float]
    upper: typing.Optional[float]
    label: str


class EEGBandpassFilterStep(PreProcessStep):
    """
    Bandpass filter, implemented using MNE in the backend.
    """
    def __init__(self,
                 frequencies: typing.List[FrequencyBand],
                 channels: typing.List[str],
                 sample_frequency: int,
                 use_original_data=False):
        super().__init__(use_original_data)
        self.bands = frequencies
        self.channels = channels
        self.sample_frequency = sample_frequency

    def apply(self, data: typing.List[pd.DataFrame]) -> typing.List[pd.DataFrame]:
        """
        Applies filtration to the given EEG data in Pandas DataFrames, generating the output as new DataFrames.

        :param data: the EEG data to apply filtration to.
        :return: a list of Pandas DataFrame representing the filtered EEG data.
        """
        return [self._run_filtration(frame) for frame in data]

    def _run_filtration(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Executes filtration on a particular DataFrame, generating a new DataFrame with the filtered data.

        :param dataframe: the DataFrame to filter.
        :return: the filtered DataFrame.
        """
        mne_data = self._convert_dataframe_to_mne(dataframe)
        frequencies_map: RawFrequencyDataMap = {}

        for frequency in self.bands:
            filtered_data: RawMNE = copy.deepcopy(mne_data)
            if frequency.upper is not None or frequency.lower is not None:
                filtered_data: mne.io.Raw = filtered_data.filter(
                    l_freq=frequency.lower,
                    h_freq=frequency.upper,
                    verbose=False,
                    l_trans_bandwidth=1,
                    h_trans_bandwidth=1
                )
            frequencies_map[frequency.label] = filtered_data

        return self._map_channels(frequencies_map, self.channels)

    def _convert_dataframe_to_mne(self, dataframe: pd.DataFrame) -> mne.io.RawArray:
        """
        Helper method which converts the given dataframe into an MNE format array.

        :param dataframe: the dataframe to convert.
        :return: an MNE format array of data.
        """
        transposed_dataframe = dataframe.transpose(copy=True)
        data_info = mne.create_info(self.channels, self.sample_frequency, ch_types='eeg')
        return mne.io.RawArray(transposed_dataframe.to_numpy(), data_info)

    @staticmethod
    def _map_channels(frequencies_map: RawFrequencyDataMap, channels: typing.List[str]) -> pd.DataFrame:
        """
        Generates a single DataFrame wherein each column is a channel / frequency band combination.
        For example, if there was only one channel (e.g., T7) and two frequency bands (e.g., Alpha, Beta) then
        the columns would be:
            "T7.Alpha", "T7.Beta"

        :param frequencies_map: The raw frequency data in a map,
                                where each key is a frequency type and each value is raw MNE format frequency data.
        :return: The DataFrame of channel / frequency band data.
        """
        frequency_data: typing.Dict[str, np.ndarray] = {}

        for channel in channels:
            for frequency_type in frequencies_map:
                channel_data: pd.DataFrame = frequencies_map[frequency_type].to_data_frame(picks=channel)
                frequency_data[f'{channel}.{frequency_type}'] = channel_data[channel].to_numpy()

        return pd.DataFrame(frequency_data)
