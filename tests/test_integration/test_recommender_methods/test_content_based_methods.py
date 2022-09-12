import pytest

from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.lda import Lda
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods

# python -m pytest .tests\test_recommender_methods\test_content_based_methods.py::TestClass::test_method

# py.test tests/test_recommender_methods/test_content_based_methods.py -k 'test_tfidf_method_bad_input'
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass


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
    assert type(similar_posts) is list
    assert len(similar_posts) > 0
    print(type(similar_posts[0]['slug']))
    assert type(similar_posts[0]['slug']) is str
    assert type(similar_posts[0]['coefficient']) is float
    assert len(similar_posts) > 0

    # newest article
    posts = posts.sort_values(by="created_at")
    latest_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(latest_post_slug)
    similar_posts = tfidf.recommend_posts_by_all_features_preprocessed(latest_post_slug)
    print("similar_posts")
    print(similar_posts)
    assert len(random_post.index) == 1
    assert type(similar_posts) is list
    assert len(similar_posts) > 0
    print(type(similar_posts[0]['slug']))
    assert type(similar_posts[0]['slug']) is str
    assert type(similar_posts[0]['coefficient']) is float
    assert len(similar_posts) > 0


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
        word2vec.get_similar_word2vec(searched_slug=tested_input, posts_from_cache=False)


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
    assert type(similar_posts) is list
    assert len(similar_posts) > 0
    print(type(similar_posts[0]['slug']))
    assert type(similar_posts[0]['slug']) is str
    assert type(similar_posts[0]['coefficient']) is float
    assert len(similar_posts) > 0


# pytest tests/test_integration/test_recommender_methods/test_content_based_methods.py::test_lda_method_bad_input
@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None,
    'blah-blah'
])
@pytest.mark.integtest
def test_lda_method_bad_input(tested_input):

    with pytest.raises(ValueError):
        lda = Lda()
        lda.get_similar_lda(tested_input, posts_from_cache=False)


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
    assert type(similar_posts) is list
    assert len(similar_posts) > 0
    print(type(similar_posts[0]['slug']))
    assert type(similar_posts[0]['slug']) is str
    assert type(similar_posts[0]['coefficient']) is float
    assert len(similar_posts) > 0


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
