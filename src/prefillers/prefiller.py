import json
import logging
import os
import random
import time as t
from pathlib import Path

import gensim
import psycopg2
from gensim.models import KeyedVectors

from src.constants.file_paths import W2V_MODELS_FOLDER_PATHS_AND_MODEL_NAMES
from src.custom_exceptions.exceptions import TestRunException
from src.data_handling.data_manipulation import DatabaseMethods
from src.data_handling.data_queries import TfIdfDataHandlers, RecommenderMethods
from src.data_handling.model_methods.user_methods import UserMethods
from src.methods.content_based.doc2vec import Doc2VecClass
from src.methods.content_based.doc_sim import DocSim, load_docsim_index
from src.methods.content_based.ldaclass import LdaClass
from src.methods.content_based.tfidf import TfIdf
from src.methods.content_based.word2vec.word2vec import Word2VecClass
from src.methods.hybrid.hybrid_methods import get_most_similar_by_hybrid
from src.methods.user_based.collaboration_based_recommendation import SvdClass

val_error_msg_db = "Not allowed DB model_variant was passed for prefilling. Choose 'pgsql' or 'redis'."
val_error_msg_algorithm = "Selected model_variant does not correspondent with any implemented model_variant."

LOGGING_FILE_PATH = 'tests/logs/prefiller_logging.txt'
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

DB_INSERTION_ERROR = "Error in DB insert. Skipping."

# defining globals
_actual_json = None
actual_json_fuzzy = None
actual_recommended_json = None
fit_by_full_text = None
fit_by_title = None
fit_by_all_features_matrix = None
tf_idf_data_handlers = None
input_text = None


def return_recommended(method, current_user_id):
    global _actual_json
    _actual_json_fuzzy = None
    user_methods = UserMethods()
    if method == "svd":
        svd = SvdClass()
        _actual_json = svd.run_svd(user_id=current_user_id, num_of_recommendations=20)
    elif method == "user_keywords":
        tfidf = TfIdf()
        input_keywords = ' '.join(user_methods.get_user_keywords(current_user_id)["keyword_name"])
        _actual_json = tfidf.keyword_based_comparison(input_keywords)
    elif method == "hybrid":
        _actual_json, _actual_json_fuzzy = get_most_similar_by_hybrid(user_id=current_user_id,
                                                                      load_from_precalc_sim_matrix=False,
                                                                      use_fuzzy=True)

    if method != "hybrid":
        _actual_json = json.dumps(_actual_json)
    else:
        raise ValueError("Method not implemented.")

    return _actual_json, _actual_json_fuzzy


def insert_recommended_json_colab(method, current_user_id, actual_json=None, _actual_json_fuzzy=None):
    user_methods = UserMethods()
    if actual_json is None and _actual_json_fuzzy is None:
        raise ValueError("_actual_json and actual_json_fuzzy cannot be None at the same time.")

    try:
        user_methods.insert_recommended_json_user_based(recommended_json=actual_json,
                                                        user_id=current_user_id, db="pgsql", method=method)
        user_methods.insert_recommended_json_user_based(recommended_json=actual_json,
                                                        user_id=current_user_id, db="redis", method=method)

        if method == "hybrid":
            method = 'hybrid_fuzzy'
            user_methods.insert_recommended_json_user_based(recommended_json=_actual_json_fuzzy,
                                                            user_id=current_user_id, db="pgsql",
                                                            method=method)
            user_methods.insert_recommended_json_user_based(recommended_json=_actual_json_fuzzy,
                                                            user_id=current_user_id, db="redis",
                                                            method=method)
    except Exception as e:
        logging.error(DB_INSERTION_ERROR)
        logging.warning(e)


def fill_recommended_collab_based(method, skip_already_filled, user_id=None, test_run=False):
    """
    Handler method for collab-based prefilling.

    @param method: i.e. "svd", "user_keywords" etc.
    @param skip_already_filled:
    @param user_id: Insert evalutation id if it is supposed to prefill recommendation only for a single evalutation,
    otherwise will prefill for all
    @param test_run: Using for tests ensuring that the method is called
    @return:
    """
    global actual_json_fuzzy
    if test_run:
        raise TestRunException("This is a test run")

    _user_methods = UserMethods()
    column_name = "recommended_by_" + method
    users = _user_methods.get_all_users(
        only_with_id_and_column_named=column_name) if user_id is None else _user_methods.get_user_dataframe(user_id)

    for user in users.to_dict("records"):
        current_user_id = user['id']
        current_recommended = user[column_name]

        actual_json, actual_json_fuzzy = return_recommended(method=method, current_user_id=current_user_id)

        if skip_already_filled:
            if current_recommended is None:
                insert_recommended_json_colab(method=method,
                                              current_user_id=current_user_id,
                                              actual_json=actual_json,
                                              _actual_json_fuzzy=actual_json_fuzzy)
        else:
            try:
                _user_methods.insert_recommended_json_user_based(recommended_json=actual_json, user_id=current_user_id,
                                                                 db="pgsql", method=method)
                _user_methods.insert_recommended_json_user_based(recommended_json=actual_json, user_id=current_user_id,
                                                                 db="redis", method=method)
            except Exception as e:
                logging.error(DB_INSERTION_ERROR)
                logging.warning(e)


def get_tfidf_full_text_fit():
    recommender_methods = RecommenderMethods()
    df = recommender_methods.get_posts_categories_dataframe(from_cache=False)

    _tf_idf_data_handlers = TfIdfDataHandlers(df)
    _fit_by_full_text = _tf_idf_data_handlers.get_fit_by_feature_('body_preprocessed')

    return _fit_by_full_text


def get_word2vec_docsim_index_model(method):
    if not method.startswith("word2vec_eval"):
        raise ValueError("Wrong method called! Method param. does not start with 'word2vec_eval'")

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
        raise ValueError("Wrong doc2vec_model name chosen.")

    docsim_index = load_docsim_index(source=source, model_name=selected_model_name, force_update=True)

    return docsim_index, w2v_model


def get_word2vec_docsim_dictionary():
    logging.debug("Loading Word2ec limited model...")

    selected_model_name = "idnes"
    source = "idnes"
    docsim_index = load_docsim_index(source=source, model_name=selected_model_name, force_update=True)
    dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_idnes.gensim')
    return dictionary, docsim_index


def get_current_recommended_short_text(method, post):
    match method:
        case "terms_frequencies":
            current_recommended = post['recommended_tfidf']
        case "word2vec":
            current_recommended = post['recommended_tfidf_full_text']
        case "doc2vec":
            current_recommended = post['recommended_doc2vec']
        case "topics":
            current_recommended = post['recommended_doc2vec_full_text']
        case _:
            current_recommended = None

    return current_recommended


def get_current_recommended_full_text(method, post):
    match method:
        case "terms_frequencies":
            current_recommended = post['recommended_tfidf']
        case "word2vec":
            current_recommended = post['recommended_word2vec']
        case "doc2vec":
            current_recommended = post['recommended_doc2vec']
        case "topics":
            current_recommended = post['recommended_lda']
        case "word2vec_eval_idnes_1":
            current_recommended = post['recommended_word2vec_eval_1']
        case "word2vec_eval_idnes_2":
            current_recommended = post['recommended_word2vec_eval_2']
        case "word2vec_eval_idnes_3":
            current_recommended = post['recommended_word2vec_eval_3']
        case "word2vec_eval_idnes_4":
            current_recommended = post['recommended_word2vec_eval_4']
        case "word2vec_limited_fasttext":
            current_recommended = post['recommended_word2vec_limited_fasttext']
        case "word2vec_limited_fasttext_full_text":
            current_recommended = post['recommended_word2vec_limited_fasttest_full_text']
        case "word2vec_eval_cswiki_1":
            current_recommended = post['recommended_word2vec_eval_cswiki_1']
        case "doc2vec_eval_cswiki_1":
            current_recommended = post['recommended_doc2vec_eval_cswiki_1']
        case _:
            current_recommended = None

    return current_recommended


def load_recommended_tfidf(slug):
    tfidf = TfIdfDataHandlers()
    _actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed(slug)
    return _actual_recommended_json


def load_recommended_short_text(method, slug, docsim_index=None, dictionary=None, w2v_model=None):
    if "PYTEST_CURRENT_TEST" in os.environ:
        logging.debug('In testing environment, inserting testing _actual_recommended_json.')
        if method == "test_prefilled_all":
            _actual_recommended_json = "[{test: test-json}]"
            return _actual_recommended_json
    else:
        if method == "terms_frequencies":
            _actual_recommended_json = load_recommended_tfidf(slug)
            return _actual_recommended_json
        elif method == "word2vec":
            if docsim_index is None or dictionary is None or w2v_model is None:
                raise ValueError("docsim_index, _dictionary or w2v_model not set.")
            word2vec = Word2VecClass()
            _actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                     model=w2v_model,
                                                                     model_name='idnes',
                                                                     docsim_index=docsim_index,
                                                                     _dictionary=dictionary)
            return _actual_recommended_json
        elif method == "doc2vec":
            doc2vec = Doc2VecClass()
            _actual_recommended_json = doc2vec.get_similar_doc2vec(searched_slug=slug)
            return _actual_recommended_json
        else:
            raise ValueError("Method %s not implemented." % method)


def handle_word2vec_variants(method, slug, w2v_model, docsim_index, dictionary):
    global actual_recommended_json
    word2vec = Word2VecClass()
    if method == "word2vec_eval_idnes_1":
        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                model=w2v_model,
                                                                model_name='idnes_1',
                                                                docsim_index=docsim_index,
                                                                _dictionary=dictionary)

    elif method == "word2vec_eval_idnes_2":
        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                model=w2v_model,
                                                                model_name='idnes_2',
                                                                docsim_index=docsim_index,
                                                                _dictionary=dictionary)
    elif method == "word2vec_eval_idnes_3":
        try:
            actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                    model=w2v_model,
                                                                    model_name='idnes_3',
                                                                    docsim_index=docsim_index,
                                                                    _dictionary=dictionary)
        except ValueError as ve:
            logging.warning(ve)
            logging.warning("Skiping this record.")
            logging.warning("!!! This will cause a missing Word2Vce prefilled record which can cause"
                            "a problem later !!!")

    elif method == "word2vec_eval_idnes_4":
        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                model=w2v_model,
                                                                model_name='idnes_4',
                                                                docsim_index=docsim_index,
                                                                _dictionary=dictionary)
    elif method == "word2vec_fasttext" or method == "word2vec_fasttext_full_text":
        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                model=w2v_model,
                                                                model_name=method,
                                                                docsim_index=docsim_index,
                                                                _dictionary=dictionary)
    elif method == "word2vec_eval_cswiki_1":
        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                model=w2v_model,
                                                                model_name='cswiki',
                                                                docsim_index=docsim_index,
                                                                _dictionary=dictionary)
    else:
        raise ValueError("No method option matches.")

    return actual_recommended_json


def load_current_recommended_full_text(method, slug, w2v_model=None, docsim_index=None, dictionary=None):
    if method == "terms_frequencies":
        tfidf = TfIdf()
        _actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(
            searched_slug=slug,
            tf_idf_data_handlers=tf_idf_data_handlers,
            fit_by_all_features_matrix=fit_by_all_features_matrix,
            fit_by_title=fit_by_title,
            fit_by_full_text=fit_by_full_text
        )
    elif method == "word2vec":
        word2vec = Word2VecClass()
        _actual_recommended_json = word2vec.get_similar_word2vec_full_text(searched_slug=slug)
    elif method == "doc2vec":
        doc2vec = Doc2VecClass()
        _actual_recommended_json = doc2vec.get_similar_doc2vec_with_full_text(slug)
    elif method == "topics":
        lda = LdaClass()
        _actual_recommended_json = lda.get_similar_lda_full_text(slug)
    elif method == "word2vec_eval_idnes_1":
        if w2v_model is None and docsim_index is None and dictionary is None:
            _actual_recommended_json = handle_word2vec_variants(method, slug, w2v_model,
                                                                docsim_index, dictionary)
        else:
            raise ValueError("w2v_model, docsim_index and _dictionary must be None.")
    elif method == "doc2vec_eval_cswiki_1":
        doc2vec = Doc2VecClass()
        _actual_recommended_json = doc2vec.get_similar_doc2vec(searched_slug=slug)
    else:
        raise ValueError("Method %s not implemented." % method)

    return _actual_recommended_json


def insert_recommendation(post_id, full_text, method, _actual_recommended_json):
    if len(_actual_recommended_json) == 0:
        logging.info("No recommended post found. Skipping.")
    else:
        _actual_recommended_json = json.dumps(_actual_recommended_json)
        try:
            database_methods = DatabaseMethods()

            database_methods.connect()
            database_methods.insert_recommended_json_content_based(
                articles_recommended_json=_actual_recommended_json,
                article_id=post_id, full_text=full_text, db="pgsql",
                method=method)
            database_methods.disconnect()
        except Exception as e:
            logging.error(DB_INSERTION_ERROR)
            logging.warning(e)


def iterate_posts(posts, method, full_text, skip_already_filled=True,
                  docsim_index=None, dictionary=None, w2v_model=None):
    for post in posts:
        if len(posts) < 1:
            break

        post_id = post['post_id']
        slug = post['slug']

        if full_text is False:
            current_recommended = get_current_recommended_short_text(method, post)
        else:
            current_recommended = get_current_recommended_full_text(method, post)

        logging.info("Searching similar articles for article: ")
        logging.info(slug)

        if skip_already_filled is True and current_recommended is None:
            logging.debug("Post:")
            logging.debug(slug)
            logging.debug("Has currently no recommended posts.")
            logging.debug("Trying to find recommended...")
            if full_text is False:
                _actual_recommended_json = load_recommended_short_text(method,
                                                                       slug,
                                                                       docsim_index,
                                                                       dictionary,
                                                                       w2v_model
                                                                       )
            else:
                _actual_recommended_json = load_current_recommended_full_text(method,
                                                                              slug,
                                                                              docsim_index,
                                                                              dictionary,
                                                                              w2v_model
                                                                              )

            insert_recommendation(post_id, full_text, method, _actual_recommended_json)


def fill_recommended_content_based(method, skip_already_filled, full_text=True, random_order=False,
                                   reversed_order=False):
    """
    Method handling the I/O and recommending method call for the content-based prefilling.
    It loads the ML models or pre-calculated files, calls the recommending methods, then saves the recommendations
    into the databse. It also handles the order of the checking, skipping of prefilled files etc...

    @param method: i.e. "svd", "user_keywords" etc.
    @param skip_already_filled:
    @param full_text:
    @param random_order:
    @param reversed_order:
    @return:
    """

    global fit_by_full_text, fit_by_title, fit_by_all_features_matrix, tf_idf_data_handlers, \
        actual_recommended_json
    docsim_index, dictionary, w2v_model = None, None, None
    database_methods = DatabaseMethods()
    if skip_already_filled is False:
        database_methods.connect()
        posts = database_methods.get_all_posts()
        database_methods.disconnect()
    else:
        database_methods.connect()
        posts = database_methods.get_not_prefilled_posts(method=method)
        database_methods.disconnect()

    if reversed_order is True:
        posts.reverse()

    if random_order is True:
        t.sleep(5)
        random.shuffle(posts)

    if method.startswith("terms_frequencies"):
        fit_by_full_text = get_tfidf_full_text_fit()
        logging.info("Starting prefilling of the TF-IDF method.")
    elif method.startswith("word2vec_"):
        docsim_index, w2v_model = get_word2vec_docsim_index_model(method)
    elif method == 'word2vec':
        dictionary, docsim_index = get_word2vec_docsim_dictionary()
    elif method.startswith("doc2vec"):  # Here was 'doc2vec_' to distinguish full text variant
        if method == "doc2vec_eval_cswiki_1":
            # Notice: Doc2Vec model gets loaded inside the Doc2Vec's class method
            logging.debug("Similarities on FastText doc2vec_model.")
            logging.debug("Loading Dov2Vec cs.Wikipedia.org doc2vec_model...")
    elif method == "topics":
        logging.debug("Prefilling with LDA method.")
    elif method.startswith("test_"):
        logging.debug("Testing method")
    else:
        raise ValueError("Non from selected method is supported. Check the 'method' parameter"
                         "value.")

    iterate_posts(posts, method, full_text, skip_already_filled,
                  docsim_index, dictionary, w2v_model)


def prefilling_job_content_based(method: str, full_text: bool, random_order=False, reversed_order=True,
                                 test_call=False):
    """
    Exception handler for the content-based methods.

    @param method:
    @param full_text:
    @param random_order:
    @param reversed_order:
    @param test_call:
    @return:
    """

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
        """
        Exception handler for the evalutation based methods.

        @param method:
        @param db:
        @param user_id:
        @param test_run:
        @param skip_already_filled:
        @return:
        """
        while True:
            if db == "pgsql":
                try:
                    fill_recommended_collab_based(method=method, skip_already_filled=skip_already_filled,
                                                  user_id=user_id, test_run=test_run)
                except psycopg2.OperationalError:
                    logging.error("DB operational error. Waiting few seconds before trying again...")
                    t.sleep(30)  # wait 30 seconds then try again
                    continue
                except TestRunException:
                    raise TestRunException("This was a test run.")
                break
            else:
                raise NotImplementedError("Other DB _source than PostgreSQL not implemented yet.")
