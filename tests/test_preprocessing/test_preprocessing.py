import os

from src.preprocessing.stopwords_loading import get_cz_stopwords_file_path, get_general_stopwords_file_path, \
    load_general_stopwords, load_cz_stopwords


# python -m pytest .\tests\test_preprocessing\test_preprocessing.py
def test_if_stopwords_file_exists():
    assert os.path.exists(get_cz_stopwords_file_path())
    assert os.path.isfile(get_cz_stopwords_file_path())
    assert os.path.exists(get_general_stopwords_file_path())
    assert os.path.isfile(get_general_stopwords_file_path())


def test_loading_of_stopwords():
    assert type(load_cz_stopwords()) is list
    assert isinstance(load_cz_stopwords(), list) is True
    assert isinstance(load_cz_stopwords()[0], list) is False
    assert isinstance(load_cz_stopwords()[0], str) is True

    assert type(load_general_stopwords()) is list
    assert isinstance(load_general_stopwords(), list) is True
    assert isinstance(load_general_stopwords()[0], list) is False
    assert isinstance(load_general_stopwords()[0], str) is True


