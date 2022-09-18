import json
import random

import pandas as pd
import pytest


# Run with:
# python -m pytest .\tests\test_user_preferences_methods.py::test_user_keywords -rP
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.user_based_algorithms\
    .user_keywords_recommendation import UserBasedMethods


# TODO:
# pytest tests\test_integration\test_recommender_methods\test_user_preferences_methods.py::test_user_categories
@pytest.mark.integtest
def test_user_categories():
    user_based_recommendation = UserBasedMethods()
    recommender_methods = RecommenderMethods()
    # TODO: Repair Error
    users = recommender_methods.get_users_dataframe()
    print("users:")
    print(users.columns)
    list_of_user_ids = users['id'].to_list()
    random_position = random.randrange(len(list_of_user_ids))
    random_id = list_of_user_ids[random_position]
    num_of_recommended_posts = 5
    recommendations = user_based_recommendation.load_recommended_posts_for_user(random_id, num_of_recommended_posts)
    print("Recommendations:")
    print(recommendations)
    assert type(recommendations) is dict
    assert len(recommendations) > 0
    assert type(recommendations['columns']) is list


@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None
])
@pytest.mark.integtest
def test_user_keyword_bad_input(tested_input):

    with pytest.raises(ValueError):
        tfidf = TfIdf()
        tfidf.keyword_based_comparison(tested_input)


def test_insert_recommended_json_user_based():
    # TODO: Insert user from the start
    recommended_methods = RecommenderMethods()
    test_dict = {'test_key': 'test_value'}
    test_json = json.dumps(test_dict)
    test_user_id = 999999
    db = 'pgsql'
    methods = ['svd', 'user_keywords']
    recommended_methods.remove_test_user_prefilled_records(test_user_id)

    database_methods = DatabaseMethods()
    sql = """SELECT recommended_by_svd, recommended_by_user_keywords FROM users WHERE id = {};"""
    # NOTICE: Connection is ok here. Need to stay here due to calling from function that's executing thread
    # operation
    sql = sql.format(test_user_id)
    database_methods.connect()
    # LOAD INTO A DATAFRAME
    df = pd.read_sql_query(sql, database_methods.get_cnx())
    database_methods.disconnect()

    assert df['recommended_by_svd'].iloc[0] is None
    assert df['recommended_by_user_keywords'].iloc[0] is None
    for method in methods:
        recommended_methods.insert_recommended_json_user_based(recommended_json=test_json, user_id=test_user_id,
                                                               db=db, method=method)

    database_methods = DatabaseMethods()
    sql = """SELECT recommended_by_svd, recommended_by_user_keywords FROM users WHERE id = {};"""
    # NOTICE: Connection is ok here. Need to stay here due to calling from function that's executing thread
    # operation
    sql = sql.format(test_user_id)
    database_methods.connect()
    # LOAD INTO A DATAFRAME
    df = pd.read_sql_query(sql, database_methods.get_cnx())
    database_methods.disconnect()

    assert df['recommended_by_svd'].iloc[0] is not None
    assert df['recommended_by_user_keywords'].iloc[0] is not None
    assert type(df['recommended_by_svd'].iloc[0]) is str
    assert type(df['recommended_by_user_keywords'].iloc[0]) is str

    recommended_methods.remove_test_user_prefilled_records(test_user_id)
