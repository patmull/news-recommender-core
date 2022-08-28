from content_based_algorithms.tfidf import TfIdf
from data_handling.data_queries import RecommenderMethods
from data_manipulation import Database
from prefilling_all import prepare_and_run
from user_based_recommendation import UserBasedRecommendation


def try_word2vec_recommendation_prefiller(database, method, full_text, reverse, random):
    prepare_and_run(database, method, full_text, reverse, random)


def try_prefillers():
    database = Database()
    method = "word2vec"
    reverse = True
    random = False
    try_word2vec_recommendation_prefiller(database=database, method=method, full_text=False, reverse=reverse,
                                          random=random)


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
"""
doc2vec = Doc2VecClass()
doc2vec.train_final_doc2vec_model_idnes()
"""

"""
word2vec = Word2VecClass()
# print(word2vec.get_similar_word2vec("chripkova-sezona-muze-letos-nemile-prekvapit-jak-se-na-ni-pripravit"))
print(word2vec.get_similar_word2vec("zdrazil-vam-dodavatel-elektrinu-nebo-plyn-brante-se-moznosti-je-nekolik"))
"""
"""
doc2vec = Doc2VecClass()
doc2vec.find_best_doc2vec_model()
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


# print_model_variant_relevances()
# save_model_variant_relevances(crop_by_date=True, last_n_by_date=80)
# print(print_model_variant_relevances_for_each_article(save_to_csv=True, crop_by_date=True))

"""
recommender_methods = RecommenderMethods()
posts_df = recommender_methods.get_posts_dataframe()
random_df_row = posts_df.sample(1)
random_slug = random_df_row['slug']
print("random_slug:")
print(random_slug.iloc[0])
found_df = recommender_methods.find_post_by_slug(random_slug.iloc[0])
print(found_df['slug'].iloc[0])
"""

"""
user_based_recommendation = UserBasedRecommendation()
user_based_recommendation.load_recommended_posts_for_user(371)
"""
tfidf = TfIdf()
print(tfidf.keyword_based_comparison("banik ostrava"))