import os
import pickle
from pathlib import Path
from unittest import mock

import pandas as pd

# python -m pytest .\tests\test_data_handling\test_data_queries.py

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods

TEST_CACHED_PICKLE_PATH = 'tests/db_cache/cached_posts_dataframe.pkl'


# pytest.mark.integration
def test_insert_posts_dataframe_to_cache():
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache(TEST_CACHED_PICKLE_PATH)
    assert os.path.exists(TEST_CACHED_PICKLE_PATH)

    # TODO: Remove this  when not needed anymore
    """
        [{"slug": "z-hromady-kameni-povstal-hrad-hartenstejn-i-s-karlovarskou-vezi", "coefficient": 4.3861717013},
         {"slug": "porozumime-nekdy-reci-zvirat-zatim-to-umeji-jenom-pohadky", "coefficient": 1.0055361237}]
    """


@mock.patch('src.recommender_core.data_handling.data_manipulation.pd.read_pickle',
            side_effect=Exception("pickle data was truncated"))
def test_get_posts_dataframe_from_cache_unpickling_error(capsys):
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache(TEST_CACHED_PICKLE_PATH)

    database_methods = DatabaseMethods()
    df = database_methods.get_posts_dataframe_from_cache()
    assert isinstance(df, pd.DataFrame)
    test_cached_file_path = Path("tests/db_cache/cached_posts_dataframe.pkl")
    assert os.path.isfile(test_cached_file_path) is False
