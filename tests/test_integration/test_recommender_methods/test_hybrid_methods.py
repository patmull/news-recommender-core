import logging

import numpy as np
import pandas as pd
import pytest

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import \
    select_list_of_posts_for_user, get_similarity_matrix_from_pairs_similarity
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    load_bert_model, Classifier

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from hybrid_methods.")

# RUN WITH:
# python -m pytest .tests\test_recommender_methods\test_content_based_methods.py::TestClass::test_method
from tests.testing_methods.random_posts_generator import get_three_unique_posts


# pytest.mark.integration
def test_bert_loading():
    bert_model = load_bert_model()
    assert str(type(bert_model)) == "<class 'spacy.lang.xx.MultiLanguage'>"


# pytest.mark.integration
def test_doc2vec_vector_representation():
    database = DatabaseMethods()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]

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
    assert isinstance(user_categories_thumbs_df, pd.DataFrame)
    THUMBS_COLUMNS_NEEDED = ['thumbs_values', 'thumbs_created_at', 'all_features_preprocessed', 'full_text']
    assert all(elem in user_categories_thumbs_df.columns.values for elem in THUMBS_COLUMNS_NEEDED)
    assert len(user_categories_thumbs_df.index) > 0  # assert there are rows in dataframe


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
