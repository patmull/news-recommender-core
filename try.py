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
print(word2vec.get_similar_word2vec("chripkova-sezona-muze-letos-nemile-prekvapit-jak-se-na-ni-pripravit"))


"""
word2vec = Word2VecClass()
word2vec.create_corpus_and_dict_from_mongo_idnes()
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
"""
path_to_cropped_wordsim_file = 'research/word2vec/similarities/WordSim353-cs-cropped.tsv'
path_to_cropped_wordsim_file = 'models/w2v_model_limited'
w2v_model = Word2Vec.load("models/w2v_idnes.model")
word_pairs_eval = w2v_model.wv.evaluate_word_pairs(path_to_cropped_wordsim_file, case_insensitive=True)
print("Word pairs test:")
print('Word_pairs_test_Pearson_coeff: ' + str(word_pairs_eval[0][0]))
print('Word_pairs_test_Pearson_p-val: ' + str(word_pairs_eval[0][1]))
print('Word_pairs_test_Spearman_coeff: ' + str(word_pairs_eval[1][0]))
print('Word_pairs_test_Spearman_p-val: ' + str(word_pairs_eval[1][1]))
print('Word_pairs_test_Out-of-vocab_ratio: ' + str(word_pairs_eval[2]))
print(word_pairs_eval)
print("Similarity test:")
print(w2v_model.wv.most_similar('tygr'))
overall_score, _ = w2v_model.wv.evaluate_word_analogies('research/word2vec/analogies/questions-words-cs.txt', case_insensitive=True)
print("Analogies evaluation of iDnes.cz model:")
print(overall_score)
"""

# try_ap()
