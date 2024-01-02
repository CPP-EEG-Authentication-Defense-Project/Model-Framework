import numpy
import numpy as np

from . import base


class HistogramEqualizationStep(base.NormalizationStep):
    """
    Implements histogram equalization, as described Lee Friedman and Oleg V. Komogortsev in their paper on
    biometric feature normalization techniques (DOI: 10.1109/TIFS.2019.2904844).

    NOTE: this step assumes that the data to be normalized has already been rescaled to the configured lower and upper
          limit.
    """
    def __init__(self, metadata: base.FeatureMetaDataIndex, lower=0, upper=255):
        super().__init__(metadata)
        self.lower = lower
        self.upper = upper

    def normalize(self, data: np.ndarray) -> np.ndarray:
        data_size = len(data)
        cumulative_histogram = self._calculate_equalized_cumulative_histogram(data)
        normalized_output = numpy.zeros(data_size)
        for i in range(data_size):
            if data[i] in range(self.lower, self.upper):
                normalized_output[i] = cumulative_histogram[data[i]]
        return normalized_output

    def _calculate_equalized_cumulative_histogram(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates an equalized cumulative histogram, to be used to transform the data into its normalized form.

        :param data: The data to use to generate the histogram.
        :return: The histogram.
        """
        cumulative_probability = self._get_cumulative_probability(data)
        cumulative_histogram = numpy.floor(cumulative_probability * self.upper)
        return cumulative_histogram

    def _get_cumulative_probability(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates a cumulative probability vector based on the given data.

        :param data: The data to use to calculate the probability vector.
        :return: The probability vector.
        """
        cumulative_density = self._get_cumulative_density(data)
        return cumulative_density / len(data)

    def _get_cumulative_density(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates the cumulative density function of the given data.

        :param data: The data to retrieve the cumulative density function for.
        :return: The cumulative density function result vector.
        """
        histogram = self._get_histogram(data)
        cumulative_density = np.zeros(histogram.shape)
        cumulative_density[0] = histogram[0]
        for i in range(1, histogram.shape[0]):
            cumulative_density[i] = cumulative_density[i - 1] + histogram[i]
        return cumulative_density

    def _get_histogram(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates histogram counts of the given data, based on the currently configured lower and upper limits.

        :param data: The data to use to calculate the histogram.
        :return: The histogram counts.
        """
        bins = list(range(self.lower, self.upper + 1))
        return np.histogram(data, bins=bins)[0]
