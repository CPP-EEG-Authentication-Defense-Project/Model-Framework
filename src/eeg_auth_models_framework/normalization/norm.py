import numpy as np

from . import base


class L1NormalizationStep(base.NormalizationStep):
    """
    L1 normalization approach, based on the description from Lee Friedman and Oleg V. Komogortsev
    in their paper on biometric feature normalization techniques (DOI: 10.1109/TIFS.2019.2904844).
    """
    def normalize(self, data: np.ndarray) -> np.ndarray:
        mean_centered_data = data - self.metadata.get_metadata_vector('mean')
        normalized_vector = mean_centered_data / np.linalg.norm(mean_centered_data, ord=1)
        return normalized_vector
