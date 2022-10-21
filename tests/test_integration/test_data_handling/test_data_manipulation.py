import os

import pandas as pd
import pytest

# python -m pytest .\tests\test_data_handling\test_data_queries.py
from src.recommender_core.data_handling.data_manipulation import get_redis_connection
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods

TEST_CACHED_PICKLE_PATH = 'tests/testing_files/cached_posts_dataframe_test.pkl'


@pytest.mark.integtest
def test_insert_posts_dataframe_to_cache():
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache(TEST_CACHED_PICKLE_PATH)
    assert os.path.exists(TEST_CACHED_PICKLE_PATH)
