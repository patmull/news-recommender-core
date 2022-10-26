import os

import pytest

# python -m pytest .\tests\test_data_handling\test_data_queries.py
from src.recommender_core.data_handling.data_queries import RecommenderMethods

TEST_CACHED_PICKLE_PATH = 'tests/testing_files/cached_posts_dataframe_test.pkl'


@pytest.mark.integtest
def test_insert_posts_dataframe_to_cache():
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache(TEST_CACHED_PICKLE_PATH)
    assert os.path.exists(TEST_CACHED_PICKLE_PATH)

# TODO: Remove this  when not needed anymore
"""
    [{"slug": "z-hromady-kameni-povstal-hrad-hartenstejn-i-s-karlovarskou-vezi", "coefficient": 4.3861717013},
     {"slug": "porozumime-nekdy-reci-zvirat-zatim-to-umeji-jenom-pohadky", "coefficient": 1.0055361237}]
"""