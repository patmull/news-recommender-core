from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


# pytest.mark.integration
from src.recommender_core.data_handling.data_queries import RecommenderMethods


def test_all_features_preprocessed_column():
    recommender_methods = RecommenderMethods()
    posts = recommender_methods.get_posts_with_no_features_preprocessed(method='all_features_preprocessed')
    return len(posts)


# pytest.mark.integration
def test_body_preprocessed_column():
    database = DatabaseMethods()
    database.connect()
    posts = database.get_posts_with_no_body_preprocessed()
    database.disconnect()
    return len(posts)


# pytest.mark.integration
def test_keywords_column():
    recommender_methods = RecommenderMethods()
    posts = recommender_methods.get_posts_with_no_features_preprocessed(method='keywords')
    return len(posts)


# python -m pytest .\tests\test_needed_columns.py::test_prefilled_features_columns
# pytest.mark.integration
def test_prefilled_features_columns():
    all_features_preprocessed = test_all_features_preprocessed_column()
    body_preprocessed = test_body_preprocessed_column()
    keywords = test_keywords_column()

    assert all_features_preprocessed == 0
    assert body_preprocessed == 0
    assert keywords == 0
