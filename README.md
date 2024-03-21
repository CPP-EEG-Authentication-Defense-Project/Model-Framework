# EEG Authentication Models Framework

This package contains abstractions used to simplify the process of training machine learning models
for the purpose of authentication with EEG data.

## Contents

The highest level abstractions available in this package are the `ModelBuilder` and `DataProcessor` classes. The
`ModelBuilder` class is intended to be inherited from to construct the training process for a type of authentication
model. On the other hand, the `DataProcessor` class wraps common processing steps, such as pre-processing, 
normalization, feature extraction, etc.

### Data

The `data` sub-package contains modules related to downloading and formatting EEG datasets. This is
done using classes that inherit from the `DatasetDownloader` class and the `DatasetReader` class. The 
`DatasetDownloader` class is used to retrieve relevant files for a particular dataset. On the other hand, the 
`DatasetReader` reads retrieved files and formats them into expected data structures for the rest of the framework
interfaces to handle.

### Features

The `features` sub-package contains modules abstracting different approaches to extracting features from EEG
data. Additionally, a `DataPipeline` interface is used to allow for multiple feature extraction techniques to be applied
in series. Each feature extraction technique inherits from the `FeatureExtractor` class and returns standard types,
so that the data produced is compatible with other abstractions.

### Normalization

The `normalization` sub-package contains modules abstracting different data normalization techniques that could be
applied to EEG data. A `DataPipeline` interface is used to allow multiple normalization steps to be applied in series.
Each normalization technique inherits from the `NormalizationStep` class and returns standard types, so that the 
normalized data continues to work with other interfaces. In addition, `NormalizationStep` may require a 
`FeatureMetaDataIndex` (which is a series of `FeatureMetaData` for the feature vectors the normalization is intended to apply to) 
to be provided at runtime, to support their normalization approach. 

### Pre-Process

The `pre_process` sub-package contains modules abstracting pre-processing techniques that can be applied to EEG data.
Similar to other major data processing interfaces, a `DataPipeline` interface is used to allow multiple pre-processing
steps to be run in series. Each pre-processing step inherits from `PreProcessStep` and returns standard types.

### Reduction

The `reduction` sub-package contains modules abstracting different approaches to reducing multiple feature vector frames
into a single feature vector. Each technique inherits from the `FeatureReduction` class and returns standard types.

### Training

The `training` sub-package contains modules abstracting steps related to training models for authentication. This 
includes common tasks such as labelling and packaging training results. This framework primarily supports a
one-model-per-user approach to authentication, so standard training results are expected to contain multiple models
for each user trained against.

### Utilities

The `utils` sub-package contains common utility modules used throughout the framework. For example, logging 
abstractions and the standard `DataPipeline` interface used by the other sub-packages.

## References

This framework leverages the following resources:

1. Auditory-evoked EEG dataset: https://doi.org/10.13026/ps31-fc50
   1. Via Physionet (https://doi.org/10.1161/01.cir.101.23.e215)
2. Biometric normalization techniques: https://doi.org/10.1109/TIFS.2019.2904844
