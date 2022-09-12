# py.test tests/test_recommender_methods/test_user_preferences_methods.py -k 'test_user_keyword_bad_input'
import pytest

from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf


@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None
])
def test_user_keyword_bad_input(tested_input):

    with pytest.raises(ValueError):
        tfidf = TfIdf()
        tfidf.keyword_based_comparison(tested_input)