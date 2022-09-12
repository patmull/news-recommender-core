from datetime import datetime
import os

import pandas as pd
import pytest


# python -m pytest .\tests\test_data_handling\test_data_queries.py
from src.recommender_core.data_handling.data_manipulation import get_redis_connection
from src.recommender_core.data_handling.data_queries import RecommenderMethods

TEST_CACHED_PICKLE_PATH = 'db_cache/cached_posts_dataframe_test.pkl'
CRITICAL_COLUMNS_POSTS = ['slug', 'all_features_preprocessed', 'body_preprocessed', 'trigrams_full_text']
CRITICAL_COLUMNS_USERS = ['name', 'slug']
CRITICAL_COLUMNS_RATINGS = ['value', 'user_id', 'post_id']
CRITICAL_COLUMNS_CATEGORIES = ['title']
CRITICAL_COLUMNS_EVALUATION_RESULTS = ['searched_id', 'query_slug', 'results_part_1', 'results_part_2', 'user_id',
                                       'model_name', 'model_variant', 'created_at']


def test_posts_dataframe_good_day():

    recommender_methods = RecommenderMethods()
    # Scenario 1: Good Day
    print('DB_RECOMMENDER_HOST')
    print(os.environ.get('DB_RECOMMENDER_HOST'))
    posts_df = recommender_methods.get_posts_dataframe()
    assert posts_df[posts_df.columns[0]].count() > 1
    common_asserts_for_dataframes(posts_df, CRITICAL_COLUMNS_POSTS)


def test_force_update_posts_cache():
    recommender_methods = RecommenderMethods()
    # Scenario 1: Good Day
    posts_df = recommender_methods.get_posts_dataframe(force_update=True)
    assert os.path.isfile(recommender_methods.cached_file_path)
    common_asserts_for_dataframes(posts_df, CRITICAL_COLUMNS_POSTS)


def test_get_df_from_sql_meanwhile_insert_cache():
    recommender_methods = RecommenderMethods()
    posts_df = recommender_methods.get_df_from_sql_meanwhile_insert_cache()
    assert os.path.isfile(recommender_methods.cached_file_path)
    common_asserts_for_dataframes(posts_df, CRITICAL_COLUMNS_POSTS)


def common_asserts_for_dataframes(df, critical_columns):
    assert isinstance(df, pd.DataFrame)
    assert len(df.index) > 1
    assert set(critical_columns).issubset(df.columns)


def test_redis():
    r = get_redis_connection()
    now = datetime.now()
    test_value = 'test_' + str(now.strftime("%m/%d/%Y %H:%M:%S"))
    r.set('test_pair', test_value)
    assert r.get('test_pair').decode() == test_value


@pytest.fixture(scope='session', autouse=True)
def teardown():
    if os.path.exists(TEST_CACHED_PICKLE_PATH):
        os.remove(TEST_CACHED_PICKLE_PATH)

