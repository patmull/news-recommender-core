# TODO: ...

import pandas as pd
import pytest
from pandas.io.sql import DatabaseError

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.prefillers.prefiller import fill_recommended_collab_based
from src.recommender_core.data_handling.data_queries import RecommenderMethods


# Good day scenario
class TestGetUsersGoodDay:

    def test_get_all_users(self):
        recommender_methods = RecommenderMethods()
        methods = ['svd', 'user_keywords']
        for method in methods:
            column_name = "recommended_by_" + method
            all_users = recommender_methods.get_all_users(only_with_id_and_column_named=column_name)
            assert isinstance(all_users, pd.DataFrame)

    def test_get_all_users_database_method(self):
        database = DatabaseMethods()

        methods = ['svd', 'user_keywords']
        for method in methods:
            column_name = "recommended_by_" + method
            database.connect()
            all_users = database.get_all_users(column_name=column_name)
            database.disconnect()
            assert (isinstance(all_users, pd.DataFrame))

        database.connect()
        all_users = database.get_all_users()
        database.disconnect()
        assert (isinstance(all_users, pd.DataFrame))


# Bad day scenario
class TestGetUsersBadDay:

    @pytest.mark.parametrize("tested_input", [
        '',
        'blah-blah'
    ])
    def test_get_users_database_method_inputs_variant_1(self, tested_input):
        database = DatabaseMethods()
        with pytest.raises(DatabaseError):
            database.connect()
            database.get_all_users(tested_input)
            database.disconnect()

    @pytest.mark.parametrize("tested_input", [
        4,
        (),
    ])
    def test_get_users_database_method_inputs_variant_2(self, tested_input):
        database = DatabaseMethods()
        with pytest.raises(TypeError):
            database.connect()
            database.get_all_users(tested_input)
            database.disconnect()

    def test_fill_recommended_collab_based(self):
        with pytest.raises(DatabaseError):
            fill_recommended_collab_based(method='blah-blah', skip_already_filled=True)
