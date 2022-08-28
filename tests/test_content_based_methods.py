from src.content_based_algorithms.doc2vec import Doc2VecClass
from src.content_based_algorithms.doc_sim import DocSim
from src.content_based_algorithms.lda import Lda
from src.content_based_algorithms.tfidf import TfIdf
from src.content_based_algorithms.word2vec import Word2VecClass
from src.data_connection import Database

# python -m pytest .tests\test_recommender_methods\test_content_based_methods.py::TestClass::test_method


# python -m pytest .tests\test_content_based_methods.py::test_tfidf_method
def test_tfidf_method():
    tfidf = TfIdf()
    # random_order article
    database = Database()
    posts = database.get_posts_dataframe()
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


# python -m pytest .\tests\test_content_based_methods.py::test_word2vec_method
def test_word2vec_method():
    word2vec = Word2VecClass()
    # random_order article
    database = Database()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    list_of_idnes_models = ["idnes_3"]

    for model in list_of_idnes_models:
        ds = DocSim()
        docsim_index, dictionary = ds.load_docsim_index_and_dictionary(source="idnes", model=model)
        print("random_post slug:")
        print(random_post_slug)
        similar_posts = word2vec.get_similar_word2vec(random_post_slug, model=model, docsim_index=docsim_index,
                                                      dictionary=dictionary)
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


def test_doc2vec_method():
    doc2vec = Doc2VecClass()
    # random_order article
    database = Database()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = doc2vec.get_similar_doc2vec(random_post_slug)
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


def test_lda_method():
    lda = Lda()
    # random_order article
    database = Database()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = lda.get_similar_lda(random_post_slug)
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


def test_tfidf_full_text_method():
    tfidf = TfIdf()
    # random_order article
    database = Database()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(random_post_slug)
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


def test_doc2vec_full_text_method():
    doc2vec = Doc2VecClass()
    # random_order article
    database = Database()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = doc2vec.get_similar_doc2vec_with_full_text(random_post_slug)
    print("similar_posts:")
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


def test_lda_full_text_method():
    lda = Lda()
    # random_order article
    database = Database()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = lda.get_similar_lda_full_text(random_post_slug)
    print("similar_posts:")
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
