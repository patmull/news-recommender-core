from src.prefillers.prefilling_all import prefiller_additional
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from tests.testing_methods.random_posts_generator import get_random_post_id


def null_column(column_name, random_post_id):
    database_methods = DatabaseMethods()
    database_methods.null_prefilled_record([column_name], random_post_id)


def test_start_preprocessed_body_prefilling():
    recommender_methods = RecommenderMethods()

    random_post_id = get_random_post_id()
    not_preprocessed_posts_before = recommender_methods.get_posts_with_no_features_preprocessed('body_preprocessed')
    null_column('body_preprocessed', random_post_id)
    not_preprocessed_posts_after_nulling = recommender_methods.get_posts_with_no_features_preprocessed('body_preprocessed')
    # Because of the 2 random on call on different posts
    assert len(not_preprocessed_posts_after_nulling) > 0

    start_preprocessed_columns_prefilling("Test full text", random_post_id)

    not_preprocessed_posts_after = recommender_methods.get_posts_with_no_features_preprocessed('body_preprocessed')

    assert len(not_preprocessed_posts_before) == len(not_preprocessed_posts_after)


def test_all_features_prefilling():
    recommender_methods = RecommenderMethods()

    random_post_id = get_random_post_id()
    null_column('all_features_preprocessed', random_post_id)
    not_preprocessed_posts_after_nulling = recommender_methods.get_posts_with_no_features_preprocessed('all_features_preprocessed')
    # Because of the 2 random on call on different posts
    assert len(not_preprocessed_posts_after_nulling) > 0

    prefiller_additional.fill_all_features_preprocessed(True, False)

    not_preprocessed_posts_after = recommender_methods.get_posts_with_no_features_preprocessed('all_features_preprocessed')

    assert len(not_preprocessed_posts_after) == 0


def test_fill_keywords():

    recommender_methods = RecommenderMethods()

    random_post_id = get_random_post_id()
    null_column('keywords', random_post_id)
    not_preprocessed_posts_after_nulling = recommender_methods.get_posts_with_no_features_preprocessed(method='keywords')
    # Because of the 2 random on call on different posts
    assert len(not_preprocessed_posts_after_nulling) > 0

    prefiller_additional.fill_keywords(True, False)
    not_preprocessed_posts_after = recommender_methods.get_posts_with_no_features_preprocessed(method='keywords')

    assert len(not_preprocessed_posts_after) == 0

