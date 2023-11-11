import re

import pandas as pd

from .base import DatasetReader


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
