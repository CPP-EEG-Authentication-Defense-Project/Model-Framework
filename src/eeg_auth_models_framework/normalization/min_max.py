from . import base, rescale


class MinMaxNormalizationStep(rescale.RescaleNormalizationStep):
    """
    Simple alias for rescaling data to a minimum and maximum of 0 and 1, respectively.
    """
    def __init__(self, metadata: base.FeatureMetaDataIndex):
        super().__init__(metadata, lower=0, upper=1.0)
