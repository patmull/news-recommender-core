import os
from unittest.mock import patch, call

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
import random

method_options = ["tfidf", "word2vec", "doc2vec", "lda"]
full_text_options = [True, False]
random_reverse_options = [True, False]

database = DatabaseMethods()


# python -m pytest .\tests\test_prefillers_methods.py::ConnectionTest::test_db_connection
class TestConnection:
    @patch("psycopg2.connect")
    def test_db_connection(self, mockconnect):
        DB_USER = os.environ.get('DB_RECOMMENDER_USER')
        DB_PASSWORD = os.environ.get('DB_RECOMMENDER_PASSWORD')
        DB_HOST = os.environ.get('DB_RECOMMENDER_HOST')
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
        assert mockconnect.call_args_list[0] == call(user=DB_USER, password=DB_PASSWORD,
                                                     host=DB_HOST, dbname=DB_NAME)


"""
def test_recommendation_prefiller(database, method, full_text, reverse, random_order):
    prepare_and_run(database, method, full_text, reverse, random_order)
"""


# python -m pytest .\tests\test_prefillers_methods.py::test_not_prefilled_retriaval
def not_prefilled_retriaval(method, full_text):
    database = DatabaseMethods()
    database.connect()
    not_prefilled_posts = database.get_not_prefilled_posts(method=method, full_text=full_text)
    database.disconnect()
    return type(not_prefilled_posts) == list


class TestPrefillers:

    def test_prefillers(self):
        for i in range(20):
            random_method_choice = random.choice(method_options)
            random_full_text_choice = random.choice(full_text_options)
            assert not_prefilled_retriaval(method=random_method_choice, full_text=random_full_text_choice) \
                   is True

