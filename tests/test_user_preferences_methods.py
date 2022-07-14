import random
import urllib.request

import pandas as pd
import urllib3

from content_based_algorithms.tfidf import TfIdf
from pytranslate import google_translate

# Run with:
# python -m pytest .\tests\test_user_preferences_methods.py::test_user_keywords
def test_user_keywords():

    word_url = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
    response = urllib.request.urlopen(word_url)
    long_txt = response.read().decode()
    words = long_txt.splitlines()
    upper_words = [word for word in words if word[0].isupper()]
    name_words = [word for word in upper_words if not word.isupper()]
    random_phrase = ' '.join([name_words[random.randint(0, len(name_words))] for i in range(2)])

    translated_random_phrase = google_translate(random_phrase)
    print("translated_random_phrase:")
    print(translated_random_phrase)

    tfidf = TfIdf()
    similar_posts = tfidf.get_recommended_posts_for_keywords(translated_random_phrase)
    assert type(similar_posts) is list
    assert len(similar_posts) > 0
    print(type(similar_posts[0]['slug']))
    assert type(similar_posts[0]['slug']) is str
    assert type(similar_posts[0]['coefficient']) is float
    assert len(similar_posts) > 0



# TODO:
"""
def user_categories():
"""