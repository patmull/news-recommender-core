import traceback

from src.prefillers.prefilling_hybrid_methods import fill_bert_vector_representation
from src.recommender_core.recommender_algorithms.content_based_algorithms.prefiller import prefilling_job
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.prefillers.prefilling_additional import PreFillerAdditional

prefiller_additional = PreFillerAdditional()


def prefill_all_features_preprocessed():
    prefiller_additional.fill_all_features_preprocessed(skip_already_filled=True, reversed_order=True,
                                                        random_order=False)


def prefill_keywords():
    prefiller_additional.fill_keywords(skip_already_filled=True, random_order=False, reversed_order=False)


def prefill_body_preprocessed():
    prefiller_additional.fill_body_preprocessed(skip_already_filled=True, random_order=False)


def prefill_bert_vector_representation():
    fill_bert_vector_representation()


def run_prefilling():
    print("Refreshing post cache. Inserting recommender posts to cache...")

    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache()

    print("Check needed columns posts...")

    database = DatabaseMethods()
    columns_needing_prefill = check_needed_columns(database)

    if len(columns_needing_prefill) > 0:
        if 'all_features_preprocessed' in columns_needing_prefill:
            prefill_all_features_preprocessed()
        if 'keywords' in columns_needing_prefill:
            prefill_keywords()
        if 'body_preprocessed' in columns_needing_prefill:
            prefill_body_preprocessed()

    reverse = True
    random = False

    full_text = False
    methods = ["tfidf", "word2vec", "doc2vec", "lda"]

    for method in methods:
        prepare_and_run(database, method, full_text, reverse, random)

    full_text = True

    for method in methods:
        prepare_and_run(database, method, full_text, reverse, random)


def prepare_and_run(database, method, full_text, reverse, random):
    database.connect()
    not_prefilled_posts = database.get_not_prefilled_posts(method=method, full_text=full_text)
    database.disconnect()
    print("Found " + str(len(not_prefilled_posts)) + " not prefilled posts in " + method + " full text: "
          + str(full_text))
    if len(not_prefilled_posts) > 0:
        try:
            prefilling_job(method=method, full_text=full_text, reversed_order=reverse, random_order=random)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " full text")


def check_needed_columns(database):
    # TODO: Check needed columns
    # TODO: 'all_features_preprocessed' (probably every method relies on this)
    # TODO: 'keywords' (LDA but probably also other columns relies on this)
    # TODO: 'body_preprocessed' (LDA relies on this)
    needed_checks = []
    database.connect()
    number_of_nans_in_all_features_preprocessed = len(database.get_posts_with_no_all_features_preprocessed())
    number_of_nans_in_keywords = len(database.get_posts_with_no_keywords())
    number_of_nans_in_body_preprocessed = len(database.get_posts_with_no_body_preprocessed())
    database.disconnect()

    if number_of_nans_in_all_features_preprocessed:
        needed_checks.extend("all_features_preprocessed")
    if number_of_nans_in_keywords:
        needed_checks.extend("keywords")
    if number_of_nans_in_body_preprocessed:
        needed_checks.extend("body_preprocessed")

    print("Values missing in:")
    print(str(needed_checks))
    return needed_checks
