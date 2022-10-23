import logging
import unittest
from pathlib import Path

import pytest
from gensim.models import KeyedVectors

from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.lda import Lda, prepare_post_categories_df
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods

# python -m pytest .tests\test_recommender_methods\test_content_based_methods.py::TestClass::test_method

# py.test tests/test_recommender_methods/test_content_based_methods.py -k 'test_tfidf_method_bad_input'
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from tests.test_integration.common_asserts import assert_recommendation


@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None
])
@pytest.mark.integtest
def test_tfidf_method_bad_input(tested_input):
    with pytest.raises(ValueError):
        tfidf = TfIdf()
        tfidf.recommend_posts_by_all_features_preprocessed(tested_input)


# python -m pytest .tests\test_content_based_methods.py::test_tfidf_method
# py.test tests/test_recommender_methods/test_content_based_methods.py -k 'test_tfidf_method'
@pytest.mark.integtest
def test_tfidf_method():
    tfidf = TfIdf()
    # random_order article
    database = DatabaseMethods()
    posts = database.get_posts_dataframe(from_cache=False)
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = tfidf.recommend_posts_by_all_features_preprocessed(random_post_slug)
    print("similar_posts")
    print(similar_posts)
    assert len(random_post.index) == 1
    assert_recommendation(similar_posts)

    # newest article
    posts = posts.sort_values(by="created_at")
    # noinspection DuplicatedCode
    latest_post_slug = posts['slug'].iloc[0]
    print("random_post slug:")
    print(latest_post_slug)
    similar_posts = tfidf.recommend_posts_by_all_features_preprocessed(latest_post_slug)
    print("similar_posts")
    print(similar_posts)
    assert len(random_post.index) == 1
    assert_recommendation(similar_posts)


# pytest tests/test_integration/test_recommender_methods/test_content_based_methods.py::test_tfidf_method_bad_input
@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None,
    'blah-blah'
])
@pytest.mark.integtest
def test_word2vec_method_bad_input(tested_input):
    with pytest.raises(ValueError):
        word2vec = Word2VecClass()
        word2vec.get_similar_word2vec(searched_slug=tested_input, model_name='idnes_3', posts_from_cache=False,
                                      force_update_data=True)


# pytest tests/test_integration/test_recommender_methods/test_content_based_methods.py::test_doc2vec_method_bad_input
@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None,
    'blah-blah'
])
@pytest.mark.integtest
def test_doc2vec_method_bad_input(tested_input):
    with pytest.raises(ValueError):
        doc2vec = Doc2VecClass()
        doc2vec.get_similar_doc2vec(searched_slug=tested_input, posts_from_cache=False)


@pytest.mark.integtest
def test_doc2vec_method_for_random_post():
    doc2vec = Doc2VecClass()
    # random_order article
    database = DatabaseMethods()
    posts = database.get_posts_dataframe(from_cache=False)
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = doc2vec.get_similar_doc2vec(searched_slug=random_post_slug, posts_from_cache=False)
    print("similar_posts")
    print(similar_posts)
    print("similar_posts type:")
    print(type(similar_posts))

    assert len(random_post.index) == 1
    assert_recommendation(similar_posts)


class TestLda:

    """
    pytest tests/test_integration/test_recommender_methods/test_content_based_methods.py::TestLda::test_get_searched_doc_id
    """
    def test_get_searched_doc_id(self):
        database = DatabaseMethods()
        posts = database.get_posts_dataframe(from_cache=False)
        random_post = posts.sample()
        random_post_slug = random_post['slug'].iloc[0]

        recommender_methods = RecommenderMethods()
        recommender_methods = prepare_post_categories_df(recommender_methods, True, random_post_slug)
        lda = Lda()
        searched_doc_id = lda.get_searched_doc_id(recommender_methods, random_post_slug)
        print('searched_doc_id:')
        print(searched_doc_id)
        assert type(searched_doc_id) is int


@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None,
    'blah-blah'
])
@pytest.mark.integtest
def test_tfidf_full_text_method_bad_input(tested_input):
    with pytest.raises(ValueError):
        tfidf = TfIdf()
        tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(tested_input, posts_from_cache=False)


@pytest.mark.integtest
def test_tfidf_full_text_method():
    tfidf = TfIdf()
    # random_order article
    database = DatabaseMethods()
    posts = database.get_posts_dataframe(from_cache=False)
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(random_post_slug,
                                                                                      posts_from_cache=False)
    print("similar_posts")
    print(similar_posts)
    print("similar_posts type:")
    print(type(similar_posts))

    assert len(random_post.index) == 1
    assert_recommendation(similar_posts)


@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None,
    'blah-blah'
])
@pytest.mark.integtest
def test_doc2vec_full_text_method_bad_inputs(tested_input):
    with pytest.raises(ValueError):
        doc2vec = Doc2VecClass()
        doc2vec.get_similar_doc2vec_with_full_text(tested_input, posts_from_cache=False)


# python -m pytest tests/test_integration/test_recommender_methods/test_content_based_methods.py::TestTfIdf
@pytest.mark.integtest
class TestTfIdf(unittest.TestCase):

    def test_load_matrix(self):
        tf_idf = TfIdf()
        matrix, saved = tf_idf.load_matrix(test_call=True)
        print(type(matrix))
        assert str(type(matrix)) == "<class 'scipy.sparse._csr.csr_matrix'>"
        assert saved is False


# python -m pytest tests/test_integration/test_recommender_methods/test_content_based_methods.py::TestWord2Vec
class TestWord2Vec:

    def test_load_word2vec_model(self):
        path_to_model = Path("models/w2v_model_limited")
        w2v_model = KeyedVectors.load(path_to_model.as_posix())
        assert w2v_model
