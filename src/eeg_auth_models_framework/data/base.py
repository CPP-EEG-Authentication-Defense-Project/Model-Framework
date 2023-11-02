import abc
import pathlib
import typing
import platformdirs
import shutil
import logging


T = typing.TypeVar('T')
APP_NAME = 'eeg-auth-defense-models'
AUTHOR = 'auth-defense-proj'
_logger = logging.getLogger('eeg-auth-defense-models')


class DatasetDownloader(abc.ABC):
    """
    Base class for a downloader which will retrieve a dataset and cache the result for future calls.
    """
    label = 'data'

    def retrieve(self, bust_cache=False) -> pathlib.Path:
        """
        Retrieves the dataset, automatically checking the locally cached data first.

        :param bust_cache: a flag indicating whether to bust the cache or not.
        :return: the path to the root directory of the downloaded data.
        """
        cache_directory = self.get_data_cache_directory()
        _logger.debug(f'Retrieve data cache directory: {cache_directory}')
        if bust_cache and cache_directory.exists():
            _logger.debug('Busting data cache')
            shutil.rmtree(cache_directory)
        if not cache_directory.exists():
            _logger.debug('Downloading data')
            cache_directory.mkdir(parents=True)
            self.download_data_to_path(cache_directory)
        return cache_directory

    @abc.abstractmethod
    def download_data_to_path(self, target: pathlib.Path):
        """
        Downloads dataset information into the given directory path.

        :param target: the target directory to download data to.
        """
        pass

    def get_data_cache_directory(self) -> pathlib.Path:
        """
        Helper method which retrieves the directory to use to store all cache data.

        :return: the directory path.
        """
        root_path = pathlib.Path(platformdirs.user_cache_dir(APP_NAME, AUTHOR))
        return root_path / self.label


class DatasetReader(abc.ABC, typing.Generic[T]):
    @abc.abstractmethod
    def format_data(self, dataset_path: pathlib.Path) -> typing.Dict[str, typing.List[T]]:
        pass
