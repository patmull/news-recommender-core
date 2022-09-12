from src.prefillers.preprocessing.stopwords_loading import load_cz_stopwords, load_general_stopwords


def test_loading_of_stopwords():
    assert type(load_cz_stopwords()) is list
    assert isinstance(load_cz_stopwords(), list) is True
    assert isinstance(load_cz_stopwords()[0], list) is False
    assert isinstance(load_cz_stopwords()[0], str) is True

    assert type(load_general_stopwords()) is list
    assert isinstance(load_general_stopwords(), list) is True
    assert isinstance(load_general_stopwords()[0], list) is False
    assert isinstance(load_general_stopwords()[0], str) is True
