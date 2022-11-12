import logging
import traceback

from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative
from src.prefillers.user_based_prefillers.prefilling_user_classifier import fill_bert_vector_representation, \
    predict_ratings_for_all_users_store_to_redis
from src.prefillers.prefiller import prefilling_job_content_based
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.prefillers.prefilling_additional import PreFillerAdditional
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import \
    precalculate_and_save_sim_matrix_for_all_posts

prefiller_additional = PreFillerAdditional()

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging in prefilling_all.")


def prefill_all_features_preprocessed():
    prefiller_additional.fill_all_features_preprocessed(skip_already_filled=True,
                                                        random_order=False)


def prefill_keywords():
    prefiller_additional.fill_keywords(skip_already_filled=True, random_order=False)


def prefill_body_preprocessed():
    prefiller_additional.fill_body_preprocessed(skip_already_filled=True, random_order=False)


def prefill_ngrams():
    prefiller_additional.fill_ngrams_for_all_posts(skip_already_filled=True, random_order=False,
                                                   full_text=True)


def prefill_bert_vector_representation():
    fill_bert_vector_representation(skip_already_filled=True, reversed_order=False, random_order=False)


def prefill_tfidf_similarity_matrix():
    precalculate_and_save_sim_matrix_for_all_posts(methods=['tfidf'])


def prefill_user_based():
    run_prefilling_collaborative()


def prefill_to_redis_based_on_user_ratings():
    predict_ratings_for_all_users_store_to_redis()


def run_prefilling(skip_cache_refresh=False, methods_short_text=None, methods_full_text=None):
    if skip_cache_refresh is False:
        # Cache update
        logging.debug("Refreshing post cache. Inserting recommender posts to cache...")
        recommender_methods = RecommenderMethods()
        recommender_methods.database.insert_posts_dataframe_to_cache()

    logging.debug("Check needed columns posts...")

    database = DatabaseMethods()
    columns_needing_prefill = check_needed_columns()

    if len(columns_needing_prefill) > 0:
        if 'all_features_preprocessed' in columns_needing_prefill:
            prefill_all_features_preprocessed()
        if 'keywords' in columns_needing_prefill:
            prefill_keywords()
        if 'body_preprocessed' in columns_needing_prefill:
            prefill_body_preprocessed()
        if 'trigrams_full_text' in columns_needing_prefill:
            prefill_ngrams()

    # Preparign for CB prefilling
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)

    tfidf = TfIdf()
    tfidf.save_sparse_matrix(for_hybrid=False)

    reverse = True
    random = False

    full_text = False
    if methods_short_text is None:
        methods = ["tfidf", "doc2vec"] # Supported short text methods
    else:
        methods = methods_short_text

    for method in methods:
        prepare_and_run(database, method, full_text, reverse, random)

    full_text = True
    if methods_full_text is None:
        methods = ["tfidf", "word2vec_eval_idnes_3", "lda"]  # NOTICE: Evaluated Word2Vec is full text!
    else:
        methods = methods_full_text

    for method in methods:
        prepare_and_run(database, method, full_text, reverse, random)

    recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)

    # prefill_bert_vector_representation()
    recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)

    # Takes too long. Abandoned for now.
    # prefill_tfidf_similarity_matrix()

    prefill_user_based()
    recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)

    prefill_to_redis_based_on_user_ratings()


def prepare_and_run(database, method, full_text, reverse, random):
    database.connect()
    not_prefilled_posts = database.get_not_prefilled_posts(method=method, full_text=full_text)
    database.disconnect()
    logging.info("Found " + str(len(not_prefilled_posts)) + " not prefilled posts in " + method + " | full text: "
                 + str(full_text))
    if len(not_prefilled_posts) > 0:
        try:
            prefilling_job_content_based(method=method, full_text=full_text, reversed_order=reverse,
                                         random_order=random)
        except Exception as e:
            print("Exception occurred " + str(e))
            traceback.print_exception(None, e, e.__traceback__)
    else:
        logging.info("No not prefilled posts found")
        logging.info("Skipping " + method + " full text: " + str(full_text))


def check_needed_columns():
    # TODO: Check needed columns
    # 'all_features_preprocessed' (probably every method relies on this)
    # 'keywords' (LDA but probably also other methods relies on this)
    # 'body_preprocessed' (LDA relies on this)
    needed_checks = []
    recommender_methods = RecommenderMethods()
    number_of_nans_in_all_features_preprocessed \
        = len(recommender_methods.get_posts_with_no_features_preprocessed(method='all_features_preprocessed'))
    number_of_nans_in_keywords \
        = len(recommender_methods.get_posts_with_no_features_preprocessed(method='keywords'))
    number_of_nans_in_body_preprocessed \
        = len(recommender_methods.get_posts_with_no_features_preprocessed(method='body_preprocessed'))
    number_of_nans_in_trigrams = len(
        recommender_methods.get_posts_with_no_features_preprocessed(method='trigrams_full_text'))

    if number_of_nans_in_all_features_preprocessed > 0:
        needed_checks.append("all_features_preprocessed")
    if number_of_nans_in_keywords > 0:
        needed_checks.append("keywords")
    if number_of_nans_in_body_preprocessed > 0:
        needed_checks.append("body_preprocessed")
    if number_of_nans_in_trigrams > 0:
        needed_checks.append("trigrams_full_text")

    logging.info("Values missing in:")
    logging.info(str(needed_checks))
    return needed_checks
