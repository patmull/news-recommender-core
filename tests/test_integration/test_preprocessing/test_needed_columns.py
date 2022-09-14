import pytest

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


@pytest.mark.integtest
def test_all_features_preprocessed_column():
    database = DatabaseMethods()
    database.connect()
    posts = database.get_posts_with_no_all_features_preprocessed()
    database.disconnect()
    return len(posts)


@pytest.mark.integtest
def test_body_preprocessed_column():
    database = DatabaseMethods()
    database.connect()
    posts = database.get_posts_with_no_body_preprocessed()
    database.disconnect()
    return len(posts)


@pytest.mark.integtest
def test_keywords_column():
    database = DatabaseMethods()
    database.connect()
    posts = database.get_posts_with_no_keywords()
    database.disconnect()
    return len(posts)


# python -m pytest .\tests\test_needed_columns.py::test_prefilled_features_columns
@pytest.mark.integtest
def test_prefilled_features_columns():
    all_features_preprocessed = test_all_features_preprocessed_column()
    body_preprocessed = test_body_preprocessed_column()
    keywords = test_keywords_column()

    assert all_features_preprocessed == 0
    assert body_preprocessed == 0
    assert keywords == 0
