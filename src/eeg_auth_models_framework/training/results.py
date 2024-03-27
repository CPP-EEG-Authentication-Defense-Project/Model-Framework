import dataclasses
import functools
import typing
import statistics


M = typing.TypeVar('M')


@dataclasses.dataclass
class TrainingStatistics:
    """
    Wrapper for data on training results.
    """
    train_start: float = 0
    train_end: float = 0
    true_positives: typing.List[typing.SupportsInt] = dataclasses.field(default_factory=list)
    false_positives: typing.List[typing.SupportsInt] = dataclasses.field(default_factory=list)
    false_negatives: typing.List[typing.SupportsInt] = dataclasses.field(default_factory=list)
    true_negatives: typing.List[typing.SupportsInt] = dataclasses.field(default_factory=list)
    scores: typing.List[float] = dataclasses.field(default_factory=list)

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

    @functools.cached_property
    def false_accept_rates(self) -> typing.List[float]:
        """
        Calculates the false acceptance rates from the training results.
        Note: the false acceptance rates are cached.

        :return: The false acceptance rates.
        """
        rates = []

        for fp, tn in zip(self.false_positives, self.true_negatives):
            rates.append(
                fp / (fp + tn)
            )

        return rates

    @functools.cached_property
    def false_reject_rates(self) -> typing.List[float]:
        """
        Calculates the false rejection rates from the training results.
        Note: the false rejection rates are cached.

        :return: The false rejection rates.
        """
        rates = []

        for fn, tp in zip(self.false_negatives, self.true_positives):
            rates.append(
                fn / (fn + tp)
            )

        return rates


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
    def global_average_score(self) -> float:
        """
        Retrieves a "global" average, which is an average of all the average training scores in the training results
        set.

        :return: The global average value.
        """
        averages = []
        for statistical_data in self.training_statistics.values():
            averages.append(statistical_data.average_score)
        return statistics.mean(averages)

    @property
    def global_average_time(self) -> float:
        """
        Retrieves the average training time for a model, in seconds.

        :return: The average training time.
        """
        training_times = [stats.training_duration for stats in self.training_statistics.values()]
        return statistics.mean(training_times)

    def iter_subject_average_scores(self) -> typing.Iterator[typing.Tuple[str, float]]:
        """
        Utility method for iterating over all the average scores associated with each subject in the results set.

        :return: The iterator over the average scores associated with each subject.
        """
        for subject, statistical_data in self.training_statistics.items():
            yield subject, statistical_data.average_score
