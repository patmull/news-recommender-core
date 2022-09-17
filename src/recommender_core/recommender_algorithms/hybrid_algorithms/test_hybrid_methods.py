import json

import pytest

from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.hybrid_methods import \
    get_most_similar_from_tfidf_matrix


def test_hybrid_by_svd_history_tfidf():
    test_user_id = 431
    searched_slug_1 = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    searched_slug_2 = "salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem"
    searched_slug_3 = "sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik"

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]
    most_similar_hybrid_by_tfidf = get_most_similar_from_tfidf_matrix(user_id=test_user_id, posts_to_compare=test_slugs)
    type_of_json = type(most_similar_hybrid_by_tfidf)
    assert type_of_json is str  # assert str
    try:
        json.loads(most_similar_hybrid_by_tfidf)
        assert True
    except ValueError:
        pytest.fail("Encountered an unexpected exception on trying to load JSON.")

