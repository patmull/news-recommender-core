import multiprocessing
import time
from unittest import mock, TestCase

import psycopg2

from src.prefillers.prefiller import prefilling_job_content_based

LOGGING_FILE_PATH = 'tests/logs/logging_testing.txt'


def mocked_f2(**kargs):
    return 'Hey'


class TestPrefiller(TestCase):

    @mock.patch('src.prefillers.prefiller.fill_recommended_content_based', side_effect=psycopg2.OperationalError)
    def test_prefilling_job_content_based(self, capsys):

        prefilling_job_content_based(method="test", full_text=False, test_call=True)

        with open(LOGGING_FILE_PATH) as f:
            log_lines = f.readlines()

        assert 'DB operational error' in ''.join(log_lines)

    # TODO: def test_fill_recommended_content_based(self): Priority: MEDIUM



