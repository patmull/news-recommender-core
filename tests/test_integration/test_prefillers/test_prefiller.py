import logging
from unittest import mock, TestCase

import psycopg2

from src.prefillers.prefiller import prefilling_job_content_based, fill_recommended_content_based
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods

LOGGING_FILE_PATH = 'tests/logs/logging_testing.txt'


def mocked_f2(**kargs):
    return 'Hey'


LOGGER = logging.getLogger(__name__)


class TestPrefiller(TestCase):

    @mock.patch('src.prefillers.prefiller.fill_recommended_content_based', side_effect=psycopg2.OperationalError)
    def test_prefilling_job_content_based(self, capsys):

        prefilling_job_content_based(method="test", full_text=False, test_call=True)

        with open(LOGGING_FILE_PATH) as f:
            log_lines = f.readlines()

        assert 'DB operational error' in ''.join(log_lines)

    # TODO: def test_fill_recommended_content_based(self): Priority: MEDIUM-HIGH
    """
    def test_fill_recommended_content_based(self, caplog):
        
        Testing by checking the logging output
        @return:

        database = DatabaseMethods()
        LOGGER.info('Testing logger now.')
        fill_recommended_content_based(skip_already_filled=True,
                                       full_text=False,
                                       random_order=False,
                                       reversed_order=False, method="test_prefilled_all")

        assert 'Found 0 not prefilled posts in test_prefilled_all' in caplog.text

        fill_recommended_content_based(skip_already_filled=False,
                                       full_text=False,
                                       random_order=False,
                                       reversed_order=False, method="test_prefilled_all")

        database.null_post_test_prefilled_record()
        fill_recommended_content_based(skip_already_filled=False,
                                       full_text=False,
                                       random_order=False,
                                       reversed_order=False, method="test_prefilled_all")
        assert 'Found 1 not prefilled posts in test_prefilled_all' in caplog.text

        # Here it fixes the problem
        fill_recommended_content_based(skip_already_filled=False,
                                       full_text=False,
                                       random_order=False,
                                       reversed_order=False, method="test_prefilled_all")
        # When running again, there should be 0 prefilled posts
        fill_recommended_content_based(skip_already_filled=False,
                                       full_text=False,
                                       random_order=False,
                                       reversed_order=False, method="test_prefilled_all")
        assert 'Found 0 not prefilled posts in test_prefilled_all' in caplog.text
    """
