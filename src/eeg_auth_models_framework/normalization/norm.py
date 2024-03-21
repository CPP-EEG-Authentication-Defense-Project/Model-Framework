import numpy as np

from . import base


class L1NormalizationStep(base.NormalizationStep):
    """
    L1 normalization approach, based on the description from Lee Friedman and Oleg V. Komogortsev
    in their paper on biometric feature normalization techniques (DOI: 10.1109/TIFS.2019.2904844).
    """
    metadata_required = True

    def normalize(self, data: np.ndarray, **kwargs) -> np.ndarray:
        metadata: base.FeatureMetaDataIndex = kwargs[self.METADATA_KEY]
        mean_centered_data = data - metadata.get_metadata_vector('mean')
        normalized_vector = mean_centered_data / np.linalg.norm(mean_centered_data, ord=1)
        return normalized_vector
