from content_based_algorithms.doc2vec import Doc2VecClass
from content_based_algorithms.doc_sim import DocSim
from content_based_algorithms.word2vec import Word2VecClass
from preprocessing.bigrams_phrases import BigramPhrases
from data_connection import Database
from prefilling_all import prepare_and_run, run_prefilling

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
word2vec = Word2VecClass()
word2vec.final_training_idnes_model()
"""


word2vec = Word2VecClass()
# print(word2vec.get_similar_word2vec("chripkova-sezona-muze-letos-nemile-prekvapit-jak-se-na-ni-pripravit"))
ds = DocSim()
docsim_index, dictionary = ds.load_docsim_index_and_dictionary()
print(word2vec.get_similar_word2vec("zdrazil-vam-dodavatel-elektrinu-nebo-plyn-brante-se-moznosti-je-nekolik",
                                    model="idnes_1", docsim_index=docsim_index, dictionary=dictionary))
print(word2vec.get_similar_word2vec("zdrazil-vam-dodavatel-elektrinu-nebo-plyn-brante-se-moznosti-je-nekolik",
                                    model="idnes_2", docsim_index=docsim_index, dictionary=dictionary))
print(word2vec.get_similar_word2vec("zdrazil-vam-dodavatel-elektrinu-nebo-plyn-brante-se-moznosti-je-nekolik",
                                    model="idnes_3", docsim_index=docsim_index, dictionary=dictionary))
print(word2vec.get_similar_word2vec("zdrazil-vam-dodavatel-elektrinu-nebo-plyn-brante-se-moznosti-je-nekolik",
                                    model="idnes_4", docsim_index=docsim_index, dictionary=dictionary))
"""
doc2vec = Doc2VecClass()
doc2vec.find_best_doc2vec_model()
"""

"""
word2vec = Word2VecClass()
word2vec.create_or_update_corpus_and_dict_from_mongo_idnes()
"""

"""
word2vecClass = Word2VecClass()
word2vecClass.find_optimal_model_idnes(random_search=True)
"""

"""
tfidf = TfIdf()
tfidf.analyze('zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy')
"""

# preprocess_question_words_file()

"""
w2v_model = Word2VecClass()
w2v_model.eval_wiki()

bigram_phrases = BigramPhrases()
bigram_phrases.train_phrases_from_mongo_idnes()
"""

# try_ap()
