import json
import random

import pandas as pd
import pytest


# Run with:
# python -m pytest .\tests\test_integration\test_recommender_methods\test_user_preferences_methods.py
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.user_based_algorithms\
    .user_keywords_recommendation import UserBasedMethods


# TODO:
# pytest tests\test_integration\test_recommender_methods\test_user_preferences_methods.py::test_user_categories
@pytest.mark.integtest
def test_user_categories():
    user_based_recommendation = UserBasedMethods()
    user_methods = UserMethods()
    # TODO: Repair Error
    users = user_methods.get_users_dataframe()
    print("users:")
    print(users.columns)
    list_of_user_ids = users['id'].to_list()
    random_position = random.randrange(len(list_of_user_ids))
    random_id = list_of_user_ids[random_position]
    num_of_recommended_posts = 5
    recommendations = user_based_recommendation.load_best_rated_by_others_in_user_categories(random_id,
                                                                                             num_of_recommended_posts)
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


@pytest.fixture()
def methods():
    methods = ['svd', 'user_keywords', 'best_rated_by_others_in_user_categories']
    return methods


@pytest.fixture()
def dict_for_test():
    test_dict = {'test_key': 'test_value'}
    return test_dict


@pytest.fixture()
def user_id_for_test():
    test_user_id = 999999
    return test_user_id


@pytest.fixture()
def json_for_test(dict_for_test):
    test_json = json.dumps(dict_for_test)
    return test_json


@pytest.fixture()
def db_columns(methods):
    db_column_appendix = 'recommended_by_'
    db_columns = [db_column_appendix + s for s in methods]
    return db_columns


def test_insert_recommended_json_user_based(methods, dict_for_test, db_columns, user_id_for_test, json_for_test):
    # TODO: Insert user from the start
    recommended_methods = RecommenderMethods()
    db = 'pgsql'

    recommended_methods.remove_test_user_prefilled_records(user_id_for_test, db_columns=db_columns)
    df = get_users_df(db_columns, user_id_for_test)

    for method in db_columns:
        assert df[method].iloc[0] is None

    insert_user_recommende_to_db(methods, json_for_test, user_id_for_test, db)
    df = get_users_df(db_columns, user_id_for_test)

    for method in db_columns:
        assert df[method].iloc[0] is not None
        assert type(df[method].iloc[0]) is str

    recommended_methods.remove_test_user_prefilled_records(user_id_for_test, db_columns=db_columns)
    df = get_users_df(db_columns, user_id_for_test)

    for method in db_columns:
        print("df[method]:")
        print(df[method])
        assert df[method].iloc[0] is None


def insert_user_recommende_to_db(methods, test_json, test_user_id, db):
    recommended_methods = RecommenderMethods()

    for method in methods:
        recommended_methods.insert_recommended_json_user_based(recommended_json=test_json, user_id=test_user_id,
                                                               db=db, method=method)


def get_users_df(db_columns, test_user_id):

    database_methods = DatabaseMethods()
    sql = """SELECT {}, {}, {} FROM users WHERE id = {};"""
    # NOTICE: Connection is ok here. Need to stay here due to calling from function that's executing thread
    # operation
    sql = sql.format(db_columns[0], db_columns[1], db_columns[2], test_user_id)
    database_methods.connect()
    # LOAD INTO A DATAFRAME
    df = pd.read_sql_query(sql, database_methods.get_cnx())
    database_methods.disconnect()

    return df
