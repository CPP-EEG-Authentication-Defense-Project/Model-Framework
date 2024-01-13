import logging
import re
import urllib.parse
import pandas as pd
import requests
import pathlib
import bs4

from . import DatasetReader
from .base import DatasetDownloader


_logger = logging.getLogger('eeg-auth-defense-models')


class AuditoryDataDownloader(DatasetDownloader):
    dataset_url = 'https://physionet.org/files/auditory-eeg/1.0.0/Segmented_Data/'
    experiment_1_data_pattern = re.compile(r's\d{2}_ex05\.csv')
    label = 'auditory'

    def download_data_to_path(self, target: pathlib.Path):
        _logger.debug(f'Sending listing request to: {self.dataset_url}')
        with requests.get(self.dataset_url) as listing_page:
            listing_soup = bs4.BeautifulSoup(
                listing_page.content,
                features='html.parser'
            )
            self._download_files_in_listing(target, listing_soup)

    def _download_files_in_listing(self, target: pathlib.Path, listing_soup: bs4.BeautifulSoup):
        """
        Helper method which iterates over all file links in the given BeautifulSoup object
        and downloads each file into the target path's directory.

        :param target: The target path directory.
        :param listing_soup: The BeautifulSoup object to use to find download links.
        """
        for file_link in listing_soup.find_all('a'):
            file_href = file_link.get('href')
            if file_href and self.experiment_1_data_pattern.match(file_href):
                file_path = target / file_href
                file_url = urllib.parse.urljoin(self.dataset_url, file_href)
                _logger.debug(f'Downloading file at: {file_url}')
                self._download_url_to_file(file_path, file_url)

    @staticmethod
    def _download_url_to_file(file_path: pathlib.Path, url: str):
        """
        Downloads the given URLs remote content to the given file path.

        :param file_path: The file path to download to.
        :param url: The URL to download from.
        """
        with requests.get(url) as response:
            with open(file_path, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=1024):
                    out_file.write(chunk)


class AuditoryDataReader(DatasetReader[pd.DataFrame]):
    """
    Utility class which reads the auditory dataset and generates a map
    of subject data.
    """
    def __init__(self):
        self.identifier_pattern = re.compile(r'(?P<identifier>s\d{2})')

    def format_data(self, dataset_path):
        """
        Reads all data files in the given directory path and generates a structure of dataframes.

        :param dataset_path: The target directory path.
        :return: A map of dataframes, where the key is an identifier for the file and the value is the dataframe.
        """
        loaded_data_map = {}

        for data_file in dataset_path.iterdir():
            if data_file.suffix == '.csv':
                subject_identifier = self._get_subject_identifier(data_file.name)
                dataframe = pd.read_csv(data_file, index_col=0, header=0)
                loaded_data_map[subject_identifier] = [dataframe]

        return loaded_data_map

    def _get_subject_identifier(self, file_name: str) -> str:
        """
        Helper method which parses a subject identifier from a data file name.

        :param file_name: The file name to parse.
        :return: A subject identifier.
        :raises ValueError: If the identifier could not be parsed.
        """
        search_result = re.search(self.identifier_pattern, file_name)
        identifier = search_result.group('identifier')
        if not identifier:
            raise ValueError(f'Unable to parse subject identifier from file: "{file_name}"')
        return identifier.upper()
