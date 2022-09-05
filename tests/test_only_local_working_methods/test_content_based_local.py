import os
import random
import urllib

import certifi

from src.recommender_core.data_handling.data_manipulation import Database
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc_sim import DocSim
from src.recommender_core.recommender_algorithms.content_based_algorithms.lda import Lda
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass


def test_user_keywords():

    path_to_words_file = "tests/datasets/english_words.txt"
    if not os.path.exists(path_to_words_file):
        word_url = "https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
        response = urllib.request.urlopen(word_url, cafile=certifi.where())
        long_txt = response.read().decode()
        words = long_txt.splitlines()
    else:
        with open(path_to_words_file, "r") as word_list:
            words = word_list.read().split('\n')
    # words = long_txt.splitlines()s)
    random_phrase = words[random.randint(0, len(words))] + ' ' + words[random.randint(0, len(words))]

    # TODO: Check whether some currently working library exists
    # translate = Translation()
    # translated_random_phrase = translate.convert('en', 'cs', random_phrase)
    # print("translated_random_phrase:")
    # print(translated_random_phrase)

    tfidf = TfIdf()
    json_keywords = {"keywords": random_phrase}
    # input_json_keywords = json.dumps(json_keywords)
    similar_posts = tfidf.keyword_based_comparison(json_keywords["keywords"])
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