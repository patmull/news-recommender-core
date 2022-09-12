import pandas as pd
import pytest

from src.recommender_core.data_handling.data_queries import RecommenderMethods

TEST_CACHED_PICKLE_PATH = 'db_cache/cached_posts_dataframe_test.pkl'
CRITICAL_COLUMNS_POSTS = ['slug', 'all_features_preprocessed', 'body_preprocessed', 'trigrams_full_text']
CRITICAL_COLUMNS_USERS = ['name', 'slug']
CRITICAL_COLUMNS_RATINGS = ['value', 'user_id', 'post_id']
CRITICAL_COLUMNS_CATEGORIES = ['title']
CRITICAL_COLUMNS_EVALUATION_RESULTS = ['searched_id', 'query_slug', 'results_part_1', 'results_part_2', 'user_id',
                                       'model_name', 'model_variant', 'created_at']

# TODO: Unit tests
