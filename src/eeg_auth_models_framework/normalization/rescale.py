import numpy as np

from . import base


class RescaleNormalizationStep(base.NormalizationStep):
    """
    Normalization step which rescales the given feature vector to a new range. This process is based on the rescale
    algorithm used by MatLab (see: https://www.mathworks.com/help/matlab/ref/rescale.html).
    """
    metadata_required = False

    def __init__(self, upper=0, lower=255):
        super().__init__()
        self.upper = upper
        self.lower = lower

    def normalize(self, data: np.ndarray) -> np.ndarray:
        return self.apply_rescale_operation(data, upper=self.upper, lower=self.lower)

    @staticmethod
    def apply_rescale_operation(data: np.ndarray, upper: float, lower: float) -> np.ndarray:
        """
        Utility method used to apply the rescale operation to a given data vector.

        The rescale implementation is based on the rescale algorithm used in MatLab.

        :param data: The data vector to rescale.
        :param upper: An upper limit for the rescale.
        :param lower: A lower limit for the rescale.
        :return: The rescaled vector.
        """
        data_min = np.min(data)
        data_max = np.max(data)

        def rescale(element: float) -> float:
            """
            Rescale function, per MatLab rescale algorithm.

            See: https://www.mathworks.com/help/matlab/ref/rescale.html

            :param element: An element from the original vector.
            :return: The rescaled element.
            """
            return lower + ((element - data_min) / (data_max - data_min)) * (upper - lower)

        rescale_vectorized = np.vectorize(rescale)
        return rescale_vectorized(data)
