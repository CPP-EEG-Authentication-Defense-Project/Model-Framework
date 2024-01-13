import numpy as np

from . import base


class MedianNormalizationStep(base.NormalizationStep):
    """
    Applies the median normalization technique, as described Lee Friedman and Oleg V. Komogortsev in their paper on
    biometric feature normalization techniques (DOI: 10.1109/TIFS.2019.2904844).

    Essentially, this normalization step subtracts the median from the given data and then divides the result by the
    median absolute deviation.
    """
    metadata_required = True

    def normalize(self, data: np.ndarray) -> np.ndarray:
        median_data = self.metadata.get_metadata_vector('median')
        normalized_data = data - median_data
        median_data = self.metadata.get_metadata_vector('median_abs_dev')
        normalized_data = normalized_data / median_data
        return normalized_data
