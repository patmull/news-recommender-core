import json
import os
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import \
    get_most_similar_by_hybrid, select_list_of_posts_for_user, get_similarity_matrix_from_pairs_similarity
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    load_bert_model, Classifier
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods, get_redis_connection


# RUN WITH:
# python -m pytest .tests\test_recommender_methods\test_content_based_methods.py::TestClass::test_method


@pytest.mark.integtest
def test_bert_loading():
    bert_model = load_bert_model()
    print(str(type(bert_model)))
    assert str(type(bert_model)) == "<class 'spacy.lang.xx.MultiLanguage'>"


@pytest.mark.integtest
def test_doc2vec_vector_representation():
    database = DatabaseMethods()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)

    doc2vec = Doc2VecClass()
    doc2vec.load_model()
    vector_representation = doc2vec.get_vector_representation(random_post_slug)

    assert type(vector_representation) is np.ndarray
    assert len(vector_representation) > 0


# RUN WITH: pytest tests/test_integration/test_recommender_methods/test_hybrid_methods.py::test_thumbs
def test_thumbs():
    database = DatabaseMethods()
    database.connect()
    user_categories_thumbs_df = database.get_posts_users_categories_thumbs()
    database.disconnect()
    print("user_categories_thumbs_df.columns")
    print(user_categories_thumbs_df.columns)
    assert isinstance(user_categories_thumbs_df, pd.DataFrame)
    THUMBS_COLUMNS_NEEDED = ['thumbs_values', 'thumbs_created_at', 'all_features_preprocessed', 'full_text']
    assert all(elem in user_categories_thumbs_df.columns.values for elem in THUMBS_COLUMNS_NEEDED)
    assert len(user_categories_thumbs_df.index) > 0  # assert there are rows in dataframe


def test_hybrid_by_svd_history_tfidf():
    test_user_id = 431
    searched_slug_1 = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    searched_slug_2 = "salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem"
    searched_slug_3 = "sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik"

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]
    tested_methods = ['tfidf', 'doc2vec']

    # posts delivered
    most_similar_hybrid_by_tfidf = get_most_similar_by_hybrid(user_id=test_user_id,
                                                              posts_to_compare=test_slugs,
                                                              list_of_methods=tested_methods)
    type_of_json = type(most_similar_hybrid_by_tfidf)
    assert type_of_json is str  # assert str
    try:
        json.loads(most_similar_hybrid_by_tfidf)
        assert True
    except ValueError:
        pytest.fail("Encountered an unexpected exception on trying to load JSON.")

    # posts not delivered
    most_similar_hybrid_by_tfidf = get_most_similar_by_hybrid(user_id=test_user_id, list_of_methods=tested_methods)
    type_of_json = type(most_similar_hybrid_by_tfidf)
    assert type_of_json is str  # assert str
    try:
        json.loads(most_similar_hybrid_by_tfidf)
        assert True
    except ValueError:
        pytest.fail("Encountered an unexpected exception on trying to load JSON.")


def test_get_similarity_matrix_from_pairs_similarity():
    test_user_id = 431
    searched_slug_1 = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    searched_slug_2 = "salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem"
    searched_slug_3 = "sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik"

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]

    # Unit
    list_of_slugs, list_of_slugs_from_history = select_list_of_posts_for_user(user_id=test_user_id,
                                                                              posts_to_compare=test_slugs)
    get_similarity_matrix_from_pairs_similarity("doc2vec", list_of_slugs, test_slugs, list_of_slugs_from_history)


@pytest.mark.parametrize("tested_input", [
    '',
    15505661,
    (),
    None,
    'ratings'
])
def test_svm_classifier_bad_user_id(tested_input):
    with pytest.raises(ValueError):
        svm = Classifier()
        assert svm.predict_relevance_for_user(use_only_sample_of=20, user_id=tested_input, relevance_by='stars')
