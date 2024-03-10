import abc
import typing


Step = typing.TypeVar('Step')
Input = typing.TypeVar('Input')
Output = typing.TypeVar('Output')


class DataPipeline(typing.List[Step], abc.ABC, typing.Generic[Step, Input, Output]):
    """
    Abstract base class defining the interface for creating a data pipeline class.
    A data pipeline is essentially a list of steps that can be executed on some
    input to produce some output.
    """
    @abc.abstractmethod
    def run(self, data: Input) -> Output:
        """
        Applies the steps of the current pipeline to the input data.

        :param data: The data to apply the current steps to.
        :return: The processed data.
        """
        pass
