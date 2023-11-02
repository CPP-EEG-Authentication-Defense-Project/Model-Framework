import abc
import typing


I = typing.TypeVar('I')
O = typing.TypeVar('O')


class PreProcessStep(abc.ABC, typing.Generic[I, O]):
    def __init__(self, use_original_data=False):
        self.use_original_data = use_original_data

    @abc.abstractmethod
    def apply(self, data: I) -> O:
        pass
