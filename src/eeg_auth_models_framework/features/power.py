import pandas as pd
import numpy as np

from mne.time_frequency import Spectrum
from . import base
from ..utils import conversion


class PSDExtractor(base.FeatureExtractor):
    """
    Feature extractor which retrieves power spectrum density values from the given data.
    """
    def __init__(self, converter: conversion.MNEDataFrameConverter, **kwargs):
        self.converter = converter
        self.psd_kwargs = kwargs

    def extract(self, frame_channel_data: pd.DataFrame) -> np.ndarray:
        frame_data_as_mne = self.converter.convert(frame_channel_data)
        spectrum_data: Spectrum = frame_data_as_mne.compute_psd(**self.psd_kwargs)
        power_spectrum: np.ndarray
        frequencies: np.ndarray
        power_spectrum, frequencies = spectrum_data.get_data(return_freqs=True)
        return power_spectrum
