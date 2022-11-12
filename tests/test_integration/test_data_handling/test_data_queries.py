import os

import pandas as pd
import pandas.core.indexes.base
import pytest

# python -m pytest .\tests\test_data_handling\test_data_queries.py
from src.recommender_core.data_handling.data_manipulation import get_redis_connection, DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods

TEST_CACHED_PICKLE_PATH = 'db_cache/cached_posts_dataframe_test.pkl'
CRITICAL_COLUMNS_POSTS = ['slug', 'all_features_preprocessed', 'body_preprocessed', 'trigrams_full_text']
CRITICAL_COLUMNS_USERS = ['name', 'slug']
CRITICAL_COLUMNS_RATINGS = ['value', 'user_id', 'post_id']
CRITICAL_COLUMNS_CATEGORIES = ['title']
CRITICAL_COLUMNS_EVALUATION_RESULTS = ['id', 'query_slug', 'results_part_1', 'results_part_2', 'user_id',
                                       'model_name', 'model_variant', 'created_at']


# pytest.mark.integration
def test_posts_dataframe_good_day():
    recommender_methods = RecommenderMethods()
    # Scenario 1: Good Day
    print('DB_RECOMMENDER_HOST')
    print(os.environ.get('DB_RECOMMENDER_HOST'))
    posts_df = recommender_methods.get_posts_dataframe(from_cache=False)
    assert posts_df[posts_df.columns[0]].count() > 1
    common_asserts_for_dataframes(posts_df, CRITICAL_COLUMNS_POSTS)

"""
** HERE WAS REDIS TESTS test_redis_values() and test_redis() BUT FOR SOME WEIRD REASON DO NOT WORK ON TOX, Python 3.10
** MOVED TO ONLY LOCAL TESTS 
"""

# pytest.mark.integration
def test_get_df_from_sql():
    recommender_methods = RecommenderMethods()
    posts_df = recommender_methods.database.get_posts_dataframe_from_sql()
    common_asserts_for_dataframes(posts_df, CRITICAL_COLUMNS_POSTS)


# RUN WITH: pytest tests/test_integration/test_data_handling/test_data_queries.py::test_results_dataframe
# pytest.mark.integration
def common_asserts_for_dataframes(df, critical_columns):
    assert isinstance(df, pd.DataFrame)
    assert len(df.index) > 1
    print("critical_columns and df.columns")
    print(critical_columns)
    print(df.columns.tolist())
    assert set(critical_columns).issubset(df.columns.tolist())


# pytest.mark.integration
def test_find_post_by_slug():
    recommender_methods = RecommenderMethods()
    posts_df = recommender_methods.get_posts_dataframe(from_cache=False)
    random_df_row = posts_df.sample(1)
    random_slug = random_df_row['slug']
    found_df = recommender_methods.find_post_by_slug(random_slug.iloc[0], from_cache=False)
    assert isinstance(found_df, pd.DataFrame)
    assert len(found_df.index) == 1
    assert set(CRITICAL_COLUMNS_POSTS).issubset(found_df.columns)
    assert found_df['slug'].iloc[0] == random_df_row['slug'].iloc[0]


# py.test tests/test_data_handling/test_data_queries.py -k 'test_find_post_by_slug_bad_input'
@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None
])
# pytest.mark.integration
def test_find_post_by_slug_bad_input(tested_input):
    with pytest.raises(ValueError):
        recommender_methods = RecommenderMethods()
        recommender_methods.find_post_by_slug(tested_input, from_cache=False)


# pytest.mark.integration
def test_posts_dataframe_file_missing():
    recommender_methods = RecommenderMethods()
    # Scenario 2: File does not exists
    recommender_methods.cached_file_path = TEST_CACHED_PICKLE_PATH
    posts_df = recommender_methods.get_posts_dataframe(force_update=True, from_cache=False)
    common_asserts_for_dataframes(posts_df, CRITICAL_COLUMNS_POSTS)


# pytest.mark.integration
def test_users_dataframe():
    user_methods = UserMethods()
    users_df = user_methods.get_users_dataframe()
    common_asserts_for_dataframes(users_df, CRITICAL_COLUMNS_USERS)


# pytest.mark.integration
def test_ratings_dataframe():
    recommender_methods = RecommenderMethods()
    ratings_df = recommender_methods.get_ratings_dataframe()
    common_asserts_for_dataframes(ratings_df, CRITICAL_COLUMNS_RATINGS)


# pytest.mark.integration
def test_categories_dataframe():
    recommender_methods = RecommenderMethods()
    categories_df = recommender_methods.get_categories_dataframe()
    common_asserts_for_dataframes(categories_df, CRITICAL_COLUMNS_CATEGORIES)


# pytest.mark.integration
def test_results_dataframe():
    recommender_methods = RecommenderMethods()
    evaluation_results_df = recommender_methods.get_ranking_evaluation_results_dataframe()
    common_asserts_for_dataframes(evaluation_results_df, CRITICAL_COLUMNS_EVALUATION_RESULTS)


# INIT METHODS FROM app.py
def test_get_sql_columns():
    recommender_methods = RecommenderMethods()
    sql_columns = recommender_methods.get_sql_columns()
    assert isinstance(sql_columns, pandas.core.indexes.base.Index)
    assert type(len(sql_columns)) is int


def test_get_sql_rows():
    recommender_methods = RecommenderMethods()
    num_of_rows = recommender_methods.get_sql_num_of_rows()
    assert type(num_of_rows) is int
    assert num_of_rows > 0


def test_get_user_read_history_with_posts():
    test_user_id = 431
    recommender_methods = RecommenderMethods()
    user_posts_history = recommender_methods.get_user_read_history_with_posts(test_user_id)
    assert isinstance(user_posts_history, pandas.DataFrame)
    assert type(len(user_posts_history)) is int


def test_get_posts_with_not_prefilled_ngrams_text():
    recommender_methods = RecommenderMethods()
    database_methods = DatabaseMethods()
    database_methods.null_prefilled_record(db_columns=['trigrams_full_text'])
    assert type(recommender_methods.get_posts_with_not_prefilled_ngrams_text()) is list


def test_get_posts_categories_dataframe():
    recommender_methods = RecommenderMethods()
    posts_categories_df = recommender_methods.get_posts_categories_dataframe()
    posts = recommender_methods.get_posts_dataframe()
    assert len(posts_categories_df) == len(posts)
