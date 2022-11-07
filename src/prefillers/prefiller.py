import json
import logging
import os
import random
import time as t
from pathlib import Path

import gensim
import psycopg2
from gensim.models import KeyedVectors
from pandas.io.sql import DatabaseError

from src.constants.file_paths import W2V_MODELS_FOLDER_PATHS_AND_MODEL_NAMES
from src.custom_exceptions.exceptions import TestRunException
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods
from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import get_most_similar_by_hybrid
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_keywords_recommendation import \
    UserBasedMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc_sim import DocSim
from src.recommender_core.recommender_algorithms.content_based_algorithms.lda import Lda
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.collaboration_based_recommendation import \
    SvdClass

val_error_msg_db = "Not allowed DB model_variant was passed for prefilling. Choose 'pgsql' or 'redis'."
val_error_msg_algorithm = "Selected model_variant does not correspondent with any implemented model_variant."

LOGGING_FILE_PATH = 'tests/logs/logging_testing.txt'
# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename=LOGGING_FILE_PATH,
                    filemode='w',
                    level=logging.DEBUG)
logging.debug("Testing logging in prefiller.")


def fill_recommended_collab_based(method, skip_already_filled, user_id=None, test_run=False):
    """

    @param method: i.e. "svd", "user_keywords" etc.
    @param skip_already_filled:
    @param user_id: Insert user id if it is supposed to prefill recommendation only for a single user,
    otherwise will prefill for all
    @param test_run: Using for tests ensuring that the method is called
    @return:
    """
    if test_run is True:
        print("Test run exception raised.")
        raise TestRunException("This is test run")

    user_methods = UserMethods()
    column_name = "recommended_by_" + method
    if user_id is None:
        try:
            users = user_methods.get_all_users(only_with_id_and_column_named=column_name)
        except DatabaseError as e:
            print("Database error occurred while executing pandas command. Check the column names.")
            raise e
    else:
        # For single user
        users = user_methods.get_user_dataframe(user_id)

    for user in users.to_dict("records"):
        print("user:")
        print(user)
        current_user_id = user['id']
        current_recommended = user[column_name]
        print("current_user_id:")
        print(current_user_id)
        print("current_recommended:")
        print(current_recommended)
        if method == "svd":
            svd = SvdClass()
            try:
                actual_json = svd.run_svd(user_id=current_user_id, num_of_recommendations=10)
            except ValueError as e:
                print("Value Error had occurred in computing " + method + ". Skipping record.")
                print(e)
                continue
        elif method == "user_keywords":
            try:
                tfidf = TfIdf()
                input_keywords = user_methods.get_user_keywords(current_user_id)
                input_keywords = ' '.join(input_keywords["keyword_name"])
                print("input_keywords:")
                print(input_keywords)
                actual_json = tfidf.keyword_based_comparison(input_keywords)
            except ValueError as e:
                print("Value Error had occurred in computing " + method + ". Skipping record.")
                print(e)
                continue
        elif method == "best_rated_by_others_in_user_categories":
            try:
                user_based_methods = UserBasedMethods()
                actual_json = user_based_methods.load_best_rated_by_others_in_user_categories(current_user_id)
            except ValueError as e:
                print("Value Error had occurred in computing " + method + ". Skipping record.")
                print(e)
                continue
        elif method == "hybrid":
            try:
                actual_json = get_most_similar_by_hybrid(user_id=current_user_id, load_from_precalc_sim_matrix=True)
            except ValueError as e:
                print("Value Error had occurred in computing " + method + ". Skipping record.")
                print(e)
                continue
        else:
            raise ValueError("Method not implemented.")

        # NOTICE: Hybrid is already doing this
        if not method == "hybrid":
            # TODO: Shouldn't this be handled for other methods too inside the mathod and not here like in hybrid?
            print("dict actual_svd_json")
            print(actual_json)
            actual_json = json.dumps(actual_json)
            print("dumped actual_svd_json")
            print(actual_json)

        if skip_already_filled is True:
            if current_recommended is None:
                try:
                    user_methods.insert_recommended_json_user_based(recommended_json=actual_json,
                                                                    user_id=current_user_id, db="pgsql",
                                                                    method=method)
                except Exception as e:
                    print("Error in DB insert. Skipping.")
                    print(e)
                    pass
        else:
            try:
                user_methods.insert_recommended_json_user_based(recommended_json=actual_json,
                                                                user_id=current_user_id, db="pgsql",
                                                                method=method)
            except Exception as e:
                print("Error in DB insert. Skipping.")
                print(e)
                pass


# TODO: Test this method alone, i.e. removing prefilled record, check logging for positive addition
def fill_recommended_content_based(method, skip_already_filled, full_text=True, random_order=False,
                                   reversed_order=False):
    docsim_index, dictionary = None, None
    database_methods = DatabaseMethods()
    if skip_already_filled is False:
        database_methods.connect()
        posts = database_methods.get_all_posts()
        database_methods.disconnect()
    else:
        database_methods.connect()
        posts = database_methods.get_not_prefilled_posts(full_text, method=method)
        database_methods.disconnect()

    number_of_inserted_rows = 0

    if reversed_order is True:
        print("Reversing list of posts...")
        posts.reverse()

    if random_order is True:
        print("Starting random_order iteration...")
        t.sleep(5)
        random.shuffle(posts)

    if method.startswith("word2vec_"):
        dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_idnes.gensim')

        if method in W2V_MODELS_FOLDER_PATHS_AND_MODEL_NAMES:
            selected_model_name = W2V_MODELS_FOLDER_PATHS_AND_MODEL_NAMES[method][1]
            path_to_folder = W2V_MODELS_FOLDER_PATHS_AND_MODEL_NAMES[method][0]

        else:
            raise ValueError("Wrong word2vec model name chosen.")

        if method.startswith("word2vec_eval_idnes_"):
            file_name = "w2v_idnes.model"
            path_to_model = path_to_folder + file_name
            w2v_model = KeyedVectors.load(path_to_model)
            source = "idnes"
        elif method.startswith("word2vec_eval_cswiki_"):
            file_name = "w2v_cswiki.model"
            path_to_model = path_to_folder + file_name
            w2v_model = KeyedVectors.load(path_to_model)
            source = "cswiki"
        else:
            ValueError("Wrong doc2vec_model name chosen.")

        ds = DocSim(w2v_model)
        docsim_index = ds.load_docsim_index(source=source, model_name=selected_model_name)
    elif method == 'word2vec':
        selected_model_name = "idnes"
        source = "idnes"
        path_to_model = Path("models/w2v_model_limited")  # type: ignore
        w2v_model = KeyedVectors.load(path_to_model.as_posix())
        ds = DocSim(w2v_model)
        docsim_index = ds.load_docsim_index(source=source, model_name=selected_model_name)
        logging.info("Loading dictionary for Word2Vec")
        dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_idnes.gensim')
    elif method.startswith("doc2vec_"):
        if method == "doc2vec_eval_cswiki_1":
            # Notice: Doc2Vec model gets loaded inside the Doc2Vec's class method
            logging.debug("Similarities on FastText doc2vec_model.")
            logging.debug("Loading Dov2Vec cs.Wikipedia.org doc2vec_model...")
    elif method.startswith("test_"):
        logging.debug("Testing method")
    else:
        raise ValueError("Non from selected method is supported. Check the 'method' parameter"
                         "value.")

    for post in posts:
        if len(posts) < 1:
            break

        post_id = post[0]
        slug = post[3]

        if full_text is False:
            if method == "tfidf":
                current_recommended = post[24]
            elif method == "word2vec":
                current_recommended = post[22]
            elif method == "doc2vec":
                current_recommended = post[26]
            elif method == "lda":
                current_recommended = post[28]
            else:
                current_recommended = None
        else:
            if method == "tfidf":
                current_recommended = post[25]
            elif method == "word2vec":
                current_recommended = post[23]
            elif method == "doc2vec":
                current_recommended = post[27]
            elif method == "lda":
                current_recommended = post[29]
            elif method == "word2vec_eval_idnes_1":
                current_recommended = post[33]
            elif method == "word2vec_eval_idnes_2":
                current_recommended = post[34]
            elif method == "word2vec_eval_idnes_3":
                current_recommended = post[35]
            elif method == "word2vec_eval_idnes_4":
                current_recommended = post[36]
            elif method == "word2vec_limited_fasttext":
                current_recommended = post[37]
            elif method == "word2vec_limited_fasttext_full_text":
                current_recommended = post[38]
            elif method == "word2vec_eval_cswiki_1":
                current_recommended = post[39]
            elif method == "doc2vec_eval_cswiki_1":
                current_recommended = post[40]
            else:
                current_recommended = None

        logging.info("Searching similar articles for article: ")
        logging.info(slug)

        if skip_already_filled is True:
            if current_recommended is None:
                logging.debug("Post:")
                logging.debug(slug)
                logging.debug("Has currently no recommended posts.")
                logging.debug("Trying to find recommended...")
                if full_text is False:
                    if "PYTEST_CURRENT_TEST" in os.environ:
                        logging.debug('In testing environment, inserting testing actual_recommended_json.')
                        if method == "test_prefilled_all":
                            actual_recommended_json = "[{test: test-json}]"
                    else:
                        if method == "tfidf":
                            tfidf = TfIdf()
                            actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed(slug)
                        elif method == "word2vec":
                            # TODO: Fix error: ValueError: DocSim index is not set.
                            if docsim_index is None:
                                raise ValueError("DocSim index is not set.")
                            if dictionary is None:
                                raise ValueError("Dictionary is not set")
                            word2vec = Word2VecClass()
                            actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                    model=w2v_model,
                                                                                    model_name='idnes',
                                                                                    docsim_index=docsim_index,
                                                                                    dictionary=dictionary)
                        elif method == "doc2vec":
                            doc2vec = Doc2VecClass()
                            actual_recommended_json = doc2vec.get_similar_doc2vec(searched_slug=slug)
                        else:
                            raise ValueError("Method %s not implemented." % method)
                else:
                    if method == "tfidf":
                        tfidf = TfIdf()
                        actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(
                            slug)
                    elif method == "word2vec":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec_full_text(searched_slug=slug)
                    elif method == "doc2vec":
                        doc2vec = Doc2VecClass()
                        actual_recommended_json = doc2vec.get_similar_doc2vec_with_full_text(slug)
                    elif method == "lda":
                        lda = Lda()
                        actual_recommended_json = lda.get_similar_lda_full_text(slug)
                    elif method == "word2vec_eval_idnes_1":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model=w2v_model,
                                                                                model_name='idnes_1',
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_eval_idnes_2":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model=w2v_model,
                                                                                model_name='idnes_2',
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_eval_idnes_3":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model=w2v_model,
                                                                                model_name='idnes_3',
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_eval_idnes_4":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model=w2v_model,
                                                                                model_name='idnes_4',
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_fasttext":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model=w2v_model,
                                                                                model_name=method,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_fasttext_full_text":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model=w2v_model,
                                                                                model_name=method,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_eval_cswiki_1":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model=w2v_model,
                                                                                model_name='cswiki',
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "doc2vec_eval_cswiki_1":
                        doc2vec = Doc2VecClass()
                        actual_recommended_json = doc2vec.get_similar_doc2vec(searched_slug=slug)
                    else:
                        raise ValueError("Method %s not implemented." % method)
                if len(actual_recommended_json) == 0:
                    print("No recommended post found. Skipping.")
                    continue
                else:
                    actual_recommended_json = json.dumps(actual_recommended_json)

                if full_text is False:
                    try:
                        database_methods.connect()
                        database_methods.insert_recommended_json_content_based(
                            articles_recommended_json=actual_recommended_json,
                            article_id=post_id, full_text=False, db="pgsql",
                            method=method)
                        database_methods.disconnect()
                    except Exception as e:
                        print("Error in DB insert. Skipping.")
                        print(e)
                        pass
                else:
                    try:
                        database_methods.connect()
                        database_methods.insert_recommended_json_content_based(
                            articles_recommended_json=actual_recommended_json,
                            article_id=post_id, full_text=True, db="pgsql",
                            method=method)
                        database_methods.disconnect()
                        number_of_inserted_rows += 1
                        print("Inserted rows in current prefilling round: " + str(number_of_inserted_rows))
                    except Exception as e:
                        print("Error in DB insert. Skipping.")
                        print(e)
                        pass
            else:
                print("Skipping.")


def prefilling_job_content_based(method: str, full_text: bool, random_order=False, reversed_order=True,
                                 test_call=False):

    while True:
        try:
            fill_recommended_content_based(method=method, full_text=full_text, skip_already_filled=True,
                                           random_order=random_order, reversed_order=reversed_order)

        except psycopg2.OperationalError:
            logging.debug("DB operational error. Waiting few seconds before trying again...")
            if test_call:
                break
            t.sleep(30)  # wait 30 seconds then try again
            continue

        break


class UserBased:

    def prefilling_job_user_based(self, method, db, user_id=None, test_run=False, skip_already_filled=False):
        while True:
            if db == "pgsql":
                try:
                    fill_recommended_collab_based(method=method, skip_already_filled=skip_already_filled,
                                                  user_id=user_id, test_run=test_run)
                except psycopg2.OperationalError:
                    print("DB operational error. Waiting few seconds before trying again...")
                    t.sleep(30)  # wait 30 seconds then try again
                    continue
                except TestRunException as e:
                    raise e
                break
            else:
                raise NotImplementedError("Other DB source than PostgreSQL not implemented yet.")
