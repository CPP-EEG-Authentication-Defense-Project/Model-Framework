import logging
import re
import urllib.parse
import requests
import pathlib
import bs4

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
