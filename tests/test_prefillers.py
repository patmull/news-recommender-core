from data_connection import Database
from prefilling_all import prepare_and_run


def test_word2vec_recommendation_prefiller(database, method, full_text, reverse, random):
    prepare_and_run(database, method, full_text, reverse, random)


def test_prefillers():
    database = Database()
    method = "word2vec"
    reverse = True
    random = False
    test_word2vec_recommendation_prefiller(database=database, method=method, full_text=False, reverse=reverse, random=random)
