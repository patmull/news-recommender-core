from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.prefillers.prefilling_all import prepare_and_run
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass


def try_word2vec_recommendation_prefiller(database, method, full_text, reverse, random):
    prepare_and_run(database, method, full_text, reverse, random)


def try_prefillers():
    database = DatabaseMethods()
    method = "word2vec"
    reverse = True
    random = False
    try_word2vec_recommendation_prefiller(database=database, method=method, full_text=False,
                                          reverse=reverse, random=random)


# try_prefillers()
# run_prefilling()


doc2vec = Doc2VecClass()
doc2vec.train_final_doc2vec_model(source="cswiki")

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
word2vec = Word2VecClass()
word2vec.final_training_idnes_model()
"""

"""
word2vec = Word2VecClass()
print(word2vec.get_similar_word2vec("chripkova-sezona-muze-letos-nemile-prekvapit-jak-se-na-ni-pripravit"))
"""
"""
doc2vec = Doc2VecClass()
doc2vec.find_best_doc2vec_model(source="cswiki")
"""

"""
word2vec = Word2VecClass()
word2vec.create_or_update_corpus_and_dict_from_mongo_idnes()
"""

"""
word2vecClass = Word2VecClass()
word2vecClass.find_optimal_model_idnes(random_search=True)
"""
word2vecClass = Word2VecClass()
word2vecClass.find_optimal_model(source="cswiki", random_search=True)

"""
tfidf = TfIdf()
tfidf.analyze('zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy')
"""

# preprocess_question_words_file()

"""
w2vec = Word2VecClass()
w2vec.eval_wiki()

bigram_phrases = BigramPhrases()
bigram_phrases.train_phrases_from_mongo_idnes()
"""

