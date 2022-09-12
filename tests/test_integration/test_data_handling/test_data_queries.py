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


@pytest.mark.integtest
def test_posts_dataframe_good_day():
    recommender_methods = RecommenderMethods()
    # Scenario 1: Good Day
    print('DB_RECOMMENDER_HOST')
    print(os.environ.get('DB_RECOMMENDER_HOST'))
    posts_df = recommender_methods.get_posts_dataframe(from_cache=False)
    assert posts_df[posts_df.columns[0]].count() > 1
    common_asserts_for_dataframes(posts_df, CRITICAL_COLUMNS_POSTS)


@pytest.mark.integtest
def test_get_df_from_sql():
    recommender_methods = RecommenderMethods()
    posts_df = recommender_methods.database.get_posts_dataframe_from_sql()
    common_asserts_for_dataframes(posts_df, CRITICAL_COLUMNS_POSTS)


@pytest.mark.integtest
def common_asserts_for_dataframes(df, critical_columns):
    assert isinstance(df, pd.DataFrame)
    assert len(df.index) > 1
    assert set(critical_columns).issubset(df.columns)


@pytest.mark.integtest
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
@pytest.mark.integtest
def test_find_post_by_slug_bad_input(tested_input):
    with pytest.raises(ValueError):
        recommender_methods = RecommenderMethods()
        recommender_methods.find_post_by_slug(tested_input, from_cache=False)


@pytest.mark.integtest
def test_posts_dataframe_file_missing():
    recommender_methods = RecommenderMethods()
    # Scenario 2: File does not exists
    recommender_methods.cached_file_path = TEST_CACHED_PICKLE_PATH
    posts_df = recommender_methods.get_posts_dataframe(force_update=True, from_cache=False)
    common_asserts_for_dataframes(posts_df, CRITICAL_COLUMNS_POSTS)


@pytest.mark.integtest
def test_users_dataframe():
    recommender_methods = RecommenderMethods()
    users_df = recommender_methods.get_users_dataframe()
    common_asserts_for_dataframes(users_df, CRITICAL_COLUMNS_USERS)


@pytest.mark.integtest
def test_ratings_dataframe():
    recommender_methods = RecommenderMethods()
    ratings_df = recommender_methods.get_ratings_dataframe()
    common_asserts_for_dataframes(ratings_df, CRITICAL_COLUMNS_RATINGS)


@pytest.mark.integtest
def test_categories_dataframe():
    recommender_methods = RecommenderMethods()
    categories_df = recommender_methods.get_categories_dataframe()
    common_asserts_for_dataframes(categories_df, CRITICAL_COLUMNS_CATEGORIES)


@pytest.mark.integtest
def test_results_dataframe():
    recommender_methods = RecommenderMethods()
    evaluation_results_df = recommender_methods.get_ranking_evaluation_results_dataframe()
    common_asserts_for_dataframes(evaluation_results_df, CRITICAL_COLUMNS_EVALUATION_RESULTS)


@pytest.mark.integtest
def test_redis():
    r = get_redis_connection()
    now = datetime.now()
    test_value = 'test_' + str(now.strftime("%m/%d/%Y %H:%M:%S"))
    r.set('test_pair', test_value)
    assert r.get('test_pair').decode() == test_value


# RUN WITH: tests/test_integration/test_data_handling/test_data_queries.py::test_redis_values
@pytest.mark.integtest
def test_redis_values():
    r = get_redis_connection()
    now = datetime.now()
    test_value = 'test_' + str(now.strftime("%m/%d/%Y %H:%M:%S"))
    r.delete(test_value)
    r.set('test_pair', test_value)
    assert r.get('test_pair').decode() == test_value
    r.delete("posts_by_pred_ratings_user_371")
    test_value = "vytvorili-prvni-rib-eye-steak-ze-zkumavky-chutna-jako-prave-maso"
    # TODO: Restrict PgSQL from ever deleting this user
    test_user_key = "posts_by_pred_ratings_user_371"
    r.delete(test_user_key)
    for key in test_user_key: r.delete(key)
    r.sadd(test_user_key, '')
    print("r.smembers(test_value):")
    print(r.smembers(test_user_key))
    res = r.sadd(test_user_key, test_value)
    print(res)
    print(r.smembers(test_user_key))
    assert res == 1
    assert type(r.smembers(test_user_key)) == set


@pytest.fixture(scope='session', autouse=True)
@pytest.mark.integtest
def teardown():
    if os.path.exists(TEST_CACHED_PICKLE_PATH):
        os.remove(TEST_CACHED_PICKLE_PATH)
