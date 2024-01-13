import unittest
import urllib.parse
import responses
import tempfile
import pathlib

from eeg_auth_models_framework import data


class AuditoryDataDownloaderUnitTest(unittest.TestCase):
    _fake_file_a = 's01_ex05.csv'
    _fake_file_b = 's02_ex05.csv'
    _fake_index_html = f"""<html>
    <head><title>Index of /static/published-projects/auditory-eeg/1.0.0/Segmented_Data/</title></head>
    <body>
    <h1>Index of /static/published-projects/auditory-eeg/1.0.0/Segmented_Data/</h1>
    <hr>
    <pre>
    <a href="../">../</a>
    <a href="RECORDS.txt">RECORDS.txt</a>                                        24-Oct-2021 17:13                3840
    <a href="{_fake_file_a}">{_fake_file_a}</a>                                  24-Oct-2021 17:09             1915024
    <a href="{_fake_file_b}">{_fake_file_b}</a>                                  24-Oct-2021 17:09             1921710
    </pre>
    <hr>
    </body>
    </html>"""
    _fake_file_url_a = urllib.parse.urljoin(data.AuditoryDataDownloader.dataset_url, _fake_file_a)
    _fake_file_url_b = urllib.parse.urljoin(data.AuditoryDataDownloader.dataset_url, _fake_file_b)
    _fake_file_content_a = 'Fake file A.'
    _fake_file_content_b = 'Fake file B.'

    @responses.activate
    def test_download_from_links(self):
        self._hook_mock_responses()
        expected_content = {self._fake_file_content_a, self._fake_file_content_b}
        downloader = data.AuditoryDataDownloader()

        with tempfile.TemporaryDirectory() as temp_directory:
            temp_directory_path = pathlib.Path(temp_directory)
            downloader.download_data_to_path(temp_directory_path)
            files = temp_directory_path.glob('*')
            files_count = 0
            content_seen = set()
            for downloaded_file in files:
                files_count += 1
                with open(downloaded_file, mode='r') as data_file:
                    content_seen.add(data_file.read())

        self.assertEqual(len(expected_content), files_count)
        self.assertEqual(expected_content, content_seen)

    def _hook_mock_responses(self):
        """
        Helper method which wraps the registration of several fake HTTP responses needed for testing.
        """
        responses.get(
            data.AuditoryDataDownloader.dataset_url,
            body=self._fake_index_html,
            status=200,
            content_type='text/html'
        )
        responses.get(
            self._fake_file_url_a,
            body=self._fake_file_content_a,
            status=200,
            content_type='text/plain'
        )
        responses.get(
            self._fake_file_url_b,
            body=self._fake_file_content_b,
            status=200,
            content_type='text/plain'
        )
