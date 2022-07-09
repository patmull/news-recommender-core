from content_based_algorithms.word2vec import Word2VecClass
from data_connection import Database

#  python -m pytest .\tests\test_content_based_methods.py


def test_word2vec_method():
    word2vec = Word2VecClass()
    # random article
    database = Database()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['post_slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = word2vec.get_similar_word2vec(random_post_slug)
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