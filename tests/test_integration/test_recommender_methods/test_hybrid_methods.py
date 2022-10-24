import json
import logging
import os.path
import random
from pathlib import Path
from unittest import mock, TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import \
    get_most_similar_by_hybrid, select_list_of_posts_for_user, get_similarity_matrix_from_pairs_similarity, \
    LIST_OF_SUPPORTED_METHODS, SIM_MATRIX_OF_ALL_POSTS_PATH, SIM_MATRIX_NAME_BASE, \
    precalculate_and_save_sim_matrix_for_all_posts, load_posts_from_sim_matrix
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    load_bert_model, Classifier
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods, get_redis_connection

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from hybrid_methods.")

# RUN WITH:
# python -m pytest .tests\test_recommender_methods\test_content_based_methods.py::TestClass::test_method
from tests.testing_methods.random_posts_generator import get_three_unique_posts


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

    searched_slug_1, searched_slug_2, searched_slug_3 = get_three_unique_posts()

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]
    tested_methods = ['tfidf', 'doc2vec'] # TODO: Why is Word2Vec not here?

    # posts delivered
    most_similar_hybrid_by_tfidf = get_most_similar_by_hybrid(user_id=test_user_id,
                                                              svd_posts_to_compare=test_slugs,
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
    searched_slug_1, searched_slug_2, searched_slug_3 = get_three_unique_posts()

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]

    # Unit
    list_of_slugs, list_of_slugs_from_history = select_list_of_posts_for_user(user_id=test_user_id,
                                                                              posts_to_compare=test_slugs)
    result = get_similarity_matrix_from_pairs_similarity("doc2vec", list_of_slugs)

    assert isinstance(result, pd.DataFrame)


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


def test_get_similarity_matrix_from_pairs_similarity_test_2():
    recommender_methods = RecommenderMethods()
    all_posts = recommender_methods.get_posts_dataframe(from_cache=False)

    all_posts_slugs = all_posts['slug'].values.tolist()
    shrinked_slugs = all_posts_slugs[:5]
    logging.debug("shrinked_slugs:")
    logging.debug(shrinked_slugs)

    """
    random_choice = random.randrange(2)
    """

    for method in LIST_OF_SUPPORTED_METHODS:
        similarity_matrix_of_all_posts = get_similarity_matrix_from_pairs_similarity(method=method,
                                                                                     list_of_slugs=shrinked_slugs)
        assert isinstance(similarity_matrix_of_all_posts, pd.DataFrame)


class TestSimMatrixPrecalc(TestCase):
    TESTED_PATH = 'tests/testing_matrices'

    recommender_methods = RecommenderMethods()

    @pytest.mark.order(1)
    @patch('src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods.SIM_MATRIX_OF_ALL_POSTS_PATH',
           Path(TESTED_PATH))
    @patch('src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods.SIM_MATRIX_NAME_BASE',
           'testing_sim_matrix_of_all_posts')
    @mock.patch('src.recommender_core.data_handling.data_queries.RecommenderMethods.update_cache_of_posts_df',
                autospec=True)
    def test_precalculate_and_save_sim_matrix_for_all_posts(self, mocked_update_cache_of_posts_df):

        mock.patch('src.recommender_core.data_handling.data_queries.RecommenderMethods.update_cache_of_posts_df')
        mocked_update_cache_of_posts_df(self)

        precalculate_and_save_sim_matrix_for_all_posts()
        for method in LIST_OF_SUPPORTED_METHODS:
            # TODO: MOCK ALSO FEATHER LOCATION, incorporate TESTED_PATH
            file_name = "%s_%s.feather" % (SIM_MATRIX_NAME_BASE, method)
            logging.debug("file_name:")
            logging.debug(file_name)
            tested_path = Path.joinpath(SIM_MATRIX_OF_ALL_POSTS_PATH, file_name).as_posix()
            logging.debug("tested_path")
            logging.debug(tested_path)
            assert os.path.exists(tested_path)

    @pytest.mark.order(2)
    @patch('src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods.SIM_MATRIX_OF_ALL_POSTS_PATH',
           Path(TESTED_PATH))
    @patch('src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods.SIM_MATRIX_NAME_BASE',
           'testing_sim_matrix_of_all_posts')
    def test_load_posts_from_sim_matrix(self):

        # NOTICE: Order mey not be expected, also needs to be runnable alone
        precalculate_and_save_sim_matrix_for_all_posts()

        recommender_methods = RecommenderMethods()
        all_posts = recommender_methods.get_posts_dataframe(from_cache=False)
        all_posts_slugs = all_posts['slug'].values.tolist()
        logging.debug("all_posts_slugs")
        logging.debug(all_posts_slugs)
        shrinked_slugs = all_posts_slugs[:5]
        logging.debug("shrinked_slugs")
        logging.debug(shrinked_slugs)
        for method in LIST_OF_SUPPORTED_METHODS:
            assert load_posts_from_sim_matrix(method, shrinked_slugs)
