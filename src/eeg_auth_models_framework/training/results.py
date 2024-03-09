import dataclasses
import typing
import statistics


M = typing.TypeVar('M')


@dataclasses.dataclass
class TrainingStatistics:
    """
    Wrapper for data on training results.
    """
    train_start: float
    train_end: float
    scores: typing.List[float]

    @property
    def training_duration(self) -> float:
        """
        The total duration of the training in seconds.

        :return: A duration in seconds.
        """
        return self.train_end - self.train_start

    @property
    def average_score(self) -> float:
        """
        The average score from the training results.

        :return: A score average.
        """
        return statistics.mean(self.scores)


@dataclasses.dataclass
class TrainingResult(typing.Generic[M]):
    """
    Simple wrapper for maps of trained models and scores.
    """
    models: typing.Dict[str, M]
    training_statistics: typing.Dict[str, TrainingStatistics]

    def get_model(self, key: str) -> typing.Optional[M]:
        """
        Retrieves a trained model from the result.

        :param key: the key to use to search for the model.
        :return: the model, or None if it does not exist.
        """
        return self.models.get(key)

    def get_average_score(self, key: str) -> typing.Optional[float]:
        """
        Retrieves an average score from the result.

        :param key: the key to use to search for scores.
        :return: the average score for the given key, or None if there is no average.
        """
        statistical_data = self.training_statistics.get(key)
        if not statistical_data:
            return None
        return statistical_data.average_score

    @property
    def global_average(self) -> float:
        """
        Retrieves a "global" average, which is an average of all the average training scores in the training results
        set.

        :return: The global average value.
        """
        averages = []
        for statistical_data in self.training_statistics.values():
            averages.append(statistical_data.average_score)
        return statistics.mean(averages)

    def iter_subject_average_scores(self) -> typing.Iterator[typing.Tuple[str, float]]:
        """
        Utility method for iterating over all the average scores associated with each subject in the results set.

        :return: The iterator over the average scores associated with each subject.
        """
        for subject, statistical_data in self.training_statistics.items():
            yield subject, statistical_data.average_score
