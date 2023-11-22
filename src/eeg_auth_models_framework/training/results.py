import dataclasses
import typing


M = typing.TypeVar('M')


@dataclasses.dataclass
class TrainingResult(typing.Generic[M]):
    """
    Simple wrapper for maps of trained models and scores.
    """
    models: typing.Dict[str, M]
    scores: typing.Dict[str, typing.List[float]]

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
        scores_found = self.get_scores(key)
        if not scores_found:
            return None
        return sum(scores_found) / len(scores_found)

    def get_scores(self, key: str) -> typing.Optional[typing.List[float]]:
        """
        Retrieves a list of scores from the result.

        :param key: the key to use to search for scores.
        :return: the scores for the given key, or None if they do not exist.
        """
        return self.scores.get(key)
