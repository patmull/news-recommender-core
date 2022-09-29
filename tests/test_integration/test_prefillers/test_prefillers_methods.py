import os
import random
from unittest import TestCase
from unittest.mock import call

from unittest.mock import patch

import pytest

from src.custom_exceptions.exceptions import TestRunException
from src.prefillers.prefiller import UserBased
from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods

database = DatabaseMethods()
method_options = ["tfidf", "word2vec", "doc2vec", "lda"]
full_text_options = [True, False]
random_reverse_options = [True, False]


class TestConnection:
    # python -m pytest .\tests\test_prefillers_methods.py::ConnectionTest::test_db_connection
    @pytest.mark.integtest
    @patch("psycopg2.connect")
    def test_db_connection(self, mockconnect):
        # noinspection PyPep8Naming
        DB_USER = os.environ.get('DB_RECOMMENDER_USER')
        # noinspection PyPep8Naming
        DB_PASSWORD = os.environ.get('DB_RECOMMENDER_PASSWORD')
        # noinspection PyPep8Naming
        DB_HOST = os.environ.get('DB_RECOMMENDER_HOST')
        # noinspection PyPep8Naming
        DB_NAME = os.environ.get('DB_RECOMMENDER_NAME')

        assert type(DB_USER) is str
        assert type(DB_PASSWORD) is str
        assert type(DB_HOST) is str
        assert type(DB_NAME) is str

        assert bool(DB_USER) is True  # not empty
        assert bool(DB_PASSWORD) is True
        assert bool(DB_HOST) is True
        assert bool(DB_NAME) is True

        database.connect()
        mockconnect.assert_called()
        assert 1 == mockconnect.call_count
        print(mockconnect.call_args_list[0])
        assert mockconnect.call_args_list[0] == call(user=DB_USER, password=DB_PASSWORD,
                                                     host=DB_HOST, dbname=DB_NAME,
                                                     keepalives=1, keepalives_idle=30, keepalives_interval=5,
                                                     keepalives_count=5)


# python -m pytest .\tests\test_prefillers_methods.py::test_not_prefilled_retriaval
@pytest.mark.integtest
def not_prefilled_retriaval(method, full_text):
    database_methods = DatabaseMethods()
    database_methods.connect()
    not_prefilled_posts = database_methods.get_not_prefilled_posts(method=method, full_text=full_text)
    database_methods.disconnect()
    return type(not_prefilled_posts) == list


@pytest.mark.integtest
class TestPrefillers:
    @pytest.mark.integtest
    def test_prefillers(self):
        for i in range(20):
            random_method_choice = random.choice(method_options)
            random_full_text_choice = random.choice(full_text_options)
            assert not_prefilled_retriaval(method=random_method_choice, full_text=random_full_text_choice) \
                   is True


@pytest.mark.integtest
class TestUserPrefillers(TestCase):

    def test_user_preferences_prefiller(self):
        with pytest.raises(TestRunException):
            print("What the heck is going on...")
            print(run_prefilling_collaborative(test_run=True))

    @patch.object(UserBased, "prefilling_job_user_based", autospec=UserBased)
    def test_prefilling_job_user_based_not_called(self, mock_prefilling_job_user_based):
        methods = ['svd', 'user_keywords', 'best_rated']  # last value is BS value
        with pytest.raises(ValueError):
            run_prefilling_collaborative(methods)
        mock_prefilling_job_user_based.assert_not_called()
