import warnings

from content_based_algorithms.data_queries import RecommenderMethods
from content_based_algorithms.lda import Lda
from content_based_algorithms.word2vec import Word2VecClass
from content_based_algorithms.tfidf import TfIdf
from data_connection import Database
from prefilling_all import prepare_and_run, run_prefilling
from preprocessing.bigrams_phrases import BigramPhrases


def try_word2vec_recommendation_prefiller(database, method, full_text, reverse, random):
    prepare_and_run(database, method, full_text, reverse, random)


def try_prefillers():
    database = Database()
    method = "word2vec"
    reverse = True
    random = False
    try_word2vec_recommendation_prefiller(database=database, method=method, full_text=False, reverse=reverse, random=random)


# try_prefillers()
# run_prefilling()

"""
lda = Lda()
lda.get_similar_lda_full_text('zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy')
"""

"""
# warning ignoring filter
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache()
    run_prefilling() 
"""

"""
word2vecClass = Word2VecClass()
word2vecClass.find_optimal_model_idnes()
"""
"""
tfidf = TfIdf()
tfidf.analyze('zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy')
"""
bigram_phrases = BigramPhrases()
bigram_phrases.train_phrases_from_mongo_idnes()