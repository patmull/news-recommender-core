import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from src.recommender_core.data_handling.data_manipulation import get_redis_connection, RedisConstants
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.models_manipulation.models_loaders import \
    load_doc2vec_model
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.recommender_algorithms.user_based_algorithms.collaboration_based_recommendation \
    import SvdClass
from src.recommender_core.data_handling.data_queries import RecommenderMethods, unique_list

LIST_OF_SUPPORTED_METHODS = ['tfidf', 'doc2vec', 'word2vec']
SIM_MATRIX_OF_ALL_POSTS_PATH = Path('precalc_matrices')
SIM_MATRIX_OF_ALL_POSTS_PATH.mkdir(parents=True, exist_ok=True)
SIM_MATRIX_NAME_BASE = 'sim_matrix_of_all_posts'

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from hybrid_methods.")


class HybridConstants:

    def __init__(self):
        r = get_redis_connection()
        redis_constants = RedisConstants()
        self.coeff_and_hours_1 = (int(r.get(redis_constants.boost_fresh_keys[1]['coeff'])),
                                  int(r.get(redis_constants.boost_fresh_keys[1]['hours'])))
        self.coeff_and_hours_2 = (int(r.get(redis_constants.boost_fresh_keys[2]['coeff'])),
                                  int(r.get(redis_constants.boost_fresh_keys[2]['hours'])))
        self.coeff_and_hours_3 = (int(r.get(redis_constants.boost_fresh_keys[3]['coeff'])),
                                  int(r.get(redis_constants.boost_fresh_keys[3]['hours'])))
        self.coeff_and_hours_4 = (int(r.get(redis_constants.boost_fresh_keys[4]['coeff'])),
                                  int(r.get(redis_constants.boost_fresh_keys[4]['hours'])))

        self.dict_of_coeffs_and_hours = {
            'boost_fresh_1': self.coeff_and_hours_1,
            'boost_fresh_2': self.coeff_and_hours_2,
            'boost_fresh_3': self.coeff_and_hours_3,
            'boost_fresh_4': self.coeff_and_hours_4,
        }


# noinspection DuplicatedCode
def model_selection(tfidf_rate, word2vec_rate, doc2vec_rate, lda_rate, number_of_recommended,
                    tf_idf_results, word2vec_results, doc2vec_results, lda_results):
    number_of_recommended_tfidf = round((tfidf_rate / 100) * number_of_recommended)
    number_of_recommended_word2vec = round((word2vec_rate / 100) * number_of_recommended)
    number_of_recommended_doc2vec = round((doc2vec_rate / 100) * number_of_recommended)
    number_of_recommended_lda = round((lda_rate / 100) * number_of_recommended)

    logging.debug("number_of_recommended_tfidf")
    logging.debug(number_of_recommended_tfidf)
    logging.debug("number_of_recommended_word2vec")
    logging.debug(number_of_recommended_word2vec)
    logging.debug("number_of_recommended_doc2vec")
    logging.debug(number_of_recommended_doc2vec)
    logging.debug("number_of_recommended_lda")
    logging.debug(number_of_recommended_lda)

    tf_idf_additional_selection = tf_idf_results.head(number_of_recommended_tfidf)
    word2vec_additional_selection = word2vec_results.head(number_of_recommended_word2vec)
    doc2vec_additional_selection = doc2vec_results.head(number_of_recommended_doc2vec)
    lda_additional_selection = lda_results.head(number_of_recommended_lda)
    # noinspection PyPep8
    return tf_idf_additional_selection, word2vec_additional_selection, doc2vec_additional_selection, \
           lda_additional_selection, number_of_recommended_tfidf, number_of_recommended_word2vec, \
           number_of_recommended_doc2vec, number_of_recommended_lda


def prepare_categories():
    recommender_methods = RecommenderMethods()
    gc.collect()
    post_category_df = recommender_methods.get_posts_categories_dataframe()

    gc.collect()

    post_category_df = post_category_df.rename(columns={'slug_x': 'slug'})
    post_category_df = post_category_df.rename(columns={'title_y': 'category'})
    post_category_df['model_name'] = 'tfidf'

    return post_category_df


def select_list_of_posts_for_user(user_id, posts_to_compare):
    """
    Appends posts from user history to posts from other algorithm, i.e. collab recommendation by SVD.

    Returns
    -------
    list
        a list of all slugs ([posts to compare] + [user reading history[), i.e. for similarity matrix building
    list
        a list of slugs from user reading history
    """
    if type(posts_to_compare) is not list:
        raise ValueError("'svd_posts_to_compare' parameter must be a list!")

    recommender_methods = RecommenderMethods()
    df_user_read_history_with_posts = recommender_methods.get_user_read_history_with_posts(user_id)

    list_of_slugs_from_history = df_user_read_history_with_posts['slug'].to_list()

    list_of_slugs = posts_to_compare + list_of_slugs_from_history
    list_of_slugs_unique = unique_list(list_of_slugs)
    return list_of_slugs_unique, list_of_slugs_from_history


def convert_similarity_matrix_to_results_dataframe(similarity_matrix):
    similarity_matrix['coefficient'] = similarity_matrix.sum(axis=1)
    results_df = similarity_matrix.sort_values(by='coefficient', ascending=False)
    results_df = results_df['coefficient']
    results_df = results_df.rename_axis('slug').reset_index()
    logging.debug("similarity_matrix after sorting:")
    logging.debug(results_df)
    return results_df


def drop_columns_from_similarity_matrix(similarity_matrix, posts_to_compare, list_of_slugs_from_history):
    similarity_matrix = similarity_matrix.drop(columns=posts_to_compare)
    similarity_matrix = similarity_matrix.drop(list_of_slugs_from_history)
    return similarity_matrix


def get_similarity_matrix_from_pairs_similarity(method, list_of_slugs, for_hybrid=True):
    w2v_model, d2v_model = None, None

    logging.debug("Calculating sim matrix for %d posts:" % (len(list_of_slugs)))

    if method == "tfidf":
        logging.debug('Calculating sim matrix for TF-IDF')
        tfidf = TfIdf()

        similarity_matrix = tfidf.get_similarity_matrix(list_of_slugs, for_hybrid=True)

        logging.debug("Similarity matrix:")
        logging.debug(similarity_matrix)
        logging.debug("Similarity matrix type:")
        logging.debug(type(similarity_matrix))

        # TODO: Everything above can be pre-computed and loaded. Priority: VERY HIGH
    else:
        if method == "word2vec":
            logging.debug('Calculating sim matrix for Word2Vec')

            path_to_model = Path("full_models/idnes/evaluated_models/word2vec_model_3/w2v_idnes.model")
            content_based_method = Word2VecClass()
            w2v_model = KeyedVectors.load(path_to_model.as_posix())
        elif method == "doc2vec":
            logging.debug('Calculating sim matrix for Doc2Vec')

            content_based_method = Doc2VecClass()
            d2v_model = load_doc2vec_model('models/d2v_full_text_limited.model')
        else:
            raise NotImplementedError("Method not supported.")

        if w2v_model is None and d2v_model is None:
            raise ValueError("Word2Vec and Doc2Vec variables are set to None. Cannot continue.")

        similarity_list = []
        i = 0
        for x in list_of_slugs:
            i += 1
            logging.debug("Post num. " + str(i))
            inner_list = []
            for y in list_of_slugs:
                logging.debug("Searching for features of post %d:" % i)
                if method == "word2vec":
                    inner_list.append(content_based_method.get_pair_similarity_word2vec(x, y, w2v_model))
                elif method == "doc2vec":
                    inner_list.append(content_based_method.get_pair_similarity_doc2vec(x, y, d2v_model))
            similarity_list.append(inner_list)
            logging.debug("similarity_list:")
            logging.debug(similarity_list)

        logging.debug("similarity_list:")
        logging.debug(similarity_list)

        similarity_matrix = pd.DataFrame(similarity_list, columns=list_of_slugs, index=list_of_slugs)
        # TODO: Everything above can be pre-computed and loaded. Priority: VERY HIGH

        logging.debug("Similarity matrix:")
        logging.debug(similarity_matrix)

        logging.debug("Similarity matrix type:")
        logging.debug(type(similarity_matrix))

    return similarity_matrix


def personalize_similarity_matrix(similarity_matrix, posts_to_compare, list_of_slugs_from_history):
    """
    Drops the articles from user read history from similarity matrix and drops articles from SVD (collab element) from axe.
    @return:
    """
    similarity_matrix = drop_columns_from_similarity_matrix(similarity_matrix, posts_to_compare,
                                                            list_of_slugs_from_history)

    return similarity_matrix


def prepare_posts():
    recommender_methods = RecommenderMethods()
    all_posts = recommender_methods.get_posts_dataframe()
    all_posts_slugs = all_posts['slug'].values.tolist()
    return all_posts_slugs


def prepare_sim_matrix_path(method):
    file_name = "%s_%s.feather" % (SIM_MATRIX_NAME_BASE, method)
    logging.debug("file_name:")
    logging.debug(file_name)
    file_path = Path.joinpath(SIM_MATRIX_OF_ALL_POSTS_PATH, file_name).as_posix()
    return file_path


def precalculate_and_save_sim_matrix_for_all_posts(methods=None):
    if methods is None:
        methods = LIST_OF_SUPPORTED_METHODS
    recommender_methods = RecommenderMethods()
    recommender_methods.update_cache_of_posts_df()
    all_posts_slugs = prepare_posts()

    for method in methods:
        logging.debug("Precalculating sim matrix for all posts for method: %s" % method)
        similarity_matrix_of_all_posts = get_similarity_matrix_from_pairs_similarity(method=method,
                                                                                     list_of_slugs=all_posts_slugs)
        file_path = prepare_sim_matrix_path(method)

        logging.debug("file_path")
        logging.debug(file_path)

        # NOTICE: Without reset_index(), there is: ValueError:
        similarity_matrix_of_all_posts = similarity_matrix_of_all_posts.reset_index()

        similarity_matrix_of_all_posts.to_feather(file_path)


# NOTICE: It would be possible to use @typechecked from typeguard here
def load_posts_from_sim_matrix(method, list_of_slugs):
    """

    @param method:
    @param list_of_slugs: slugs delivered from SVD algorithm = slugs that we are interested in
    @return:
    """
    logging.debug("method:")
    logging.debug(method)
    file_path = prepare_sim_matrix_path(method)
    sim_matrix = pd.read_feather(file_path)
    logging.debug("sim matrix after load from feather:")
    logging.debug(sim_matrix.head(5))
    try:
        sim_matrix.index = sim_matrix['slug']
        sim_matrix = sim_matrix.drop('slug', axis=1)
    except KeyError as ke:
        logging.warning(ke)
        sim_matrix.index = sim_matrix['index']
        sim_matrix = sim_matrix.drop('index', axis=1)

    # Selecting columns
    sim_matrix = sim_matrix[list_of_slugs]
    logging.debug("sim_matrix.columns")
    logging.debug(sim_matrix.columns)

    sim_matrix = sim_matrix.loc[list_of_slugs, :]
    logging.debug("sim_matrix.index")
    logging.debug(sim_matrix.index)

    logging.debug("sim matrix after index dealing:")
    logging.debug(sim_matrix.head(5))

    return sim_matrix


def boost_by_article_freshness(results_df):
    def boost_coefficient(coeff_value, boost):
        logging.debug("coeff_value")
        logging.debug(coeff_value)
        logging.debug("boost")
        logging.debug(boost)
        d = coeff_value * boost
        if d == 0.0:
            d = d + boost
        return d

    # TODO: Get boost values from DB. Priority MEDIUM
    # See: hybrid_settings table where it was laid out
    now = pd.to_datetime('now')
    r = get_redis_connection()
    redis_constants = RedisConstants()
    results_df['coefficient'] = results_df.apply(
        lambda x: boost_coefficient(x['coefficient'], int(r.get(redis_constants.boost_fresh_keys[1]['coeff'])))
        if ((now - x['post_created_at']) < pd.Timedelta(int(r.get(redis_constants.boost_fresh_keys[1]['hours'])), 'h'))
        else (boost_coefficient(x['coefficient'], int(r.get(redis_constants.boost_fresh_keys[2]['coeff']))))
        if ((now - x['post_created_at']) < pd.Timedelta(int(r.get(redis_constants.boost_fresh_keys[2]['hours'])), 'h'))
        else (boost_coefficient(x['coefficient'], int(r.get(redis_constants.boost_fresh_keys[3]['coeff']))))
        if ((now - x['post_created_at']) < pd.Timedelta(int(r.get(redis_constants.boost_fresh_keys[3]['hours'])), 'h'))
        else (boost_coefficient(x['coefficient'], int(r.get(redis_constants.boost_fresh_keys[4]['coeff']))))
        , axis=1
    )
    return results_df


def get_most_similar_by_hybrid(user_id: int, load_from_precalc_sim_matrix=True, svd_posts_to_compare=None,
                               list_of_methods=None, save_result=False,
                               load_saved_result=False):
    """
    Get most similar from content based matrix and delivered posts.

    Parameters
    ----------
    posts_to_compare: i.e. svd_recommended_posts; if not supplied, it will calculate fresh SVD
    user_id: int by user id from DB
    @param load_from_precalc_sim_matrix: completely skips sim_matrix creation, instead load from pre-calculated
    sim. matrix and derives the needed dataframe of interested posts (recommended by SVD posts) from index and column
    @param svd_posts_to_compare:
    @param user_id:
    @param list_of_methods:
    @param save_result: saves the results (i.e. for debugging, this can be loaded with load_saved_result method below).
    Added to help with debugging of final boosting
    @param load_saved_result: if True, skips the recommending calculation and jumps to final calculations.
    Added to help with debugging of final boosting
    """
    path_to_save_results = Path('research/hybrid/results_df.pkl')  # Primarily for debugging purposes

    if load_saved_result is False or not os.path.exists(path_to_save_results):
        if type(user_id) is not int:
            raise TypeError("User id muse be an int")

        if list_of_methods is None:
            list_of_methods = ['tfidf', 'doc2vec', 'word2vec']
        elif not set(list_of_methods).issubset(LIST_OF_SUPPORTED_METHODS) > 0:
            raise NotImplementedError("Inserted methods must correspond to DB columns.")
        if svd_posts_to_compare is None:
            svd = SvdClass()
            recommended_by_svd = svd.run_svd(user_id=user_id, dict_results=False, num_of_recommendations=5)
            svd_posts_to_compare = recommended_by_svd['slug'].to_list()
        
        list_of_slugs, list_of_slugs_from_history = select_list_of_posts_for_user(user_id, svd_posts_to_compare)

        list_of_similarity_results = []
        r = get_redis_connection()

        for method in list_of_methods:
            if method == "tfidf":
                try:
                    constant = r.get('settings:content-based:tfidf:coeff')
                except ConnectionError as ce:
                    logging.warning(ce)
                    logging.warning("Getting field number of posts. "
                                    "This is not getting values from Moje články settings!")
                    constant = 7.75
            elif method == "doc2vec":
                try:
                    constant = r.get('settings:content-based:doc2vec:coeff')
                except ConnectionError as ce:
                    logging.warning(ce)
                    logging.warning("Getting field number of posts. "
                                    "This is not getting values from Moje články settings!")
                    constant = 6.75
            elif method == "word2vec":
                try:
                    constant = r.get('settings:content-based:word2vec:coeff')
                except ConnectionError as ce:
                    logging.warning(ce)
                    logging.warning("Getting field number of posts. "
                                    "This is not getting values from Moje články settings!")
                    constant = 8.75
            else:
                raise ValueError("No from selected options available.")

            constant = np.float64(constant)

            if method == "tfidf":
                file_path = prepare_sim_matrix_path(method)
                # TODO: Derive from loaded feather of sim matrix instead
                if load_from_precalc_sim_matrix \
                        and os.path \
                        .exists(file_path):
                    # Loading posts we are interested in from pre-calculated similarity matrix
                    try:
                        similarity_matrix = load_posts_from_sim_matrix(method, list_of_slugs)
                    except KeyError as ke:
                        logging.warning("Key error occurred while trying to get posts from similarity matrix. "
                                        "Sim. matrix probably not updated. Calculating fresh similarity "
                                        "but this can take a time.")
                        logging.warning("Consider updating the similarity matrix in the code before to save a time.")
                        logging.warning("FULL ERROR:")
                        logging.warning(ke)
                        logging.info("Calculating similarities fresh only between supplied articles.")
                        similarity_matrix = get_similarity_matrix_from_pairs_similarity(method, list_of_slugs)
                else:
                    # Calculating new similarity matrix only based on posts we are interested
                    similarity_matrix = get_similarity_matrix_from_pairs_similarity(method, list_of_slugs)
                similarity_matrix = personalize_similarity_matrix(similarity_matrix, svd_posts_to_compare,
                                                                  list_of_slugs_from_history)
                similarity_matrix = similarity_matrix * constant
                results = convert_similarity_matrix_to_results_dataframe(similarity_matrix)
            elif method == "doc2vec" or "word2vec":
                file_path = prepare_sim_matrix_path(method)
                if load_from_precalc_sim_matrix \
                        and os.path.exists(file_path):

                    # Loading posts we are interested in from pre-calculated similarity matrix
                    try:
                        similarity_matrix = load_posts_from_sim_matrix(method, list_of_slugs)
                    except KeyError as ke:
                        logging.warning("KeyError occurred. Probably due to not updated sim matrix."
                                        "Sim matrix needs to be updated by "
                                        "precalculate_and_save_sim_matrix_for_all_posts() method.")
                        logging.warning("Consider updating the similarity matrix in the code before to save a time.")
                        logging.warning("FULL ERROR:")
                        logging.warning(ke)
                        logging.info("Calculating similarities fresh only between supplied articles.")
                        similarity_matrix = get_similarity_matrix_from_pairs_similarity(method, list_of_slugs)
                else:
                    # Calculating new similarity matrix only based on posts we are interested
                    similarity_matrix = get_similarity_matrix_from_pairs_similarity(method, list_of_slugs)
                similarity_matrix = personalize_similarity_matrix(similarity_matrix, svd_posts_to_compare,
                                                                  list_of_slugs_from_history)
                similarity_matrix = similarity_matrix * constant
                results = convert_similarity_matrix_to_results_dataframe(similarity_matrix)
            else:
                raise NotImplementedError("Supplied method not implemented")
            list_of_similarity_results.append(results)
            logging.debug("list_of_similarity_matrices:")
            logging.debug(list_of_similarity_results)
        logging.debug("list_of_similarity_matrices after finished for loop:")
        logging.debug(list_of_similarity_results)
        list_of_prefixed_methods = ['coefficient_' + x for x in list_of_methods if not str(x) == "nan"]
        list_of_keys = ['slug'] + list_of_prefixed_methods

        logging.debug("list_of_similarity_results[0]:")
        logging.debug(list_of_similarity_results[0])

        results_df = pd.concat([list_of_similarity_results[0]['slug']], axis=1)
        for result in list_of_similarity_results:
            results_df = pd.concat([results_df, result['coefficient']], axis=1)

        results_df.columns = list_of_keys

        logging.debug("results_df")
        logging.debug(results_df)

        coefficient_columns = list_of_prefixed_methods

        results_df[coefficient_columns] = (results_df[coefficient_columns] - results_df[coefficient_columns].mean()) \
                                          / results_df[coefficient_columns].std()
        logging.debug("normalized_df:")

        logging.debug(results_df)
        results_df['coefficient'] = results_df.sum(axis=1)

        recommender_methods = RecommenderMethods()
        df_posts_categories = recommender_methods.get_posts_categories_dataframe()

        logging.debug("results_df:")
        logging.debug(results_df)
        logging.debug(results_df.columns)

        results_df = results_df.merge(df_posts_categories, left_on='slug', right_on='slug')
        logging.debug("results_df after merge")
        logging.debug(results_df)
        logging.debug(results_df.columns)

        if save_result:
            path_to_save_results.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_pickle(path_to_save_results.as_posix())
    else:
        results_df = pd.read_pickle(path_to_save_results.as_posix())

    user_methods = UserMethods()
    user_categories = user_methods.get_user_categories(user_id)
    logging.debug("Categories for user " + str(user_id))
    logging.debug(user_categories)
    user_categories_list = user_categories['category_slug'].values.tolist()
    logging.debug("user_categories_list:")
    logging.debug(user_categories_list)

    # If post contains user category, then boost the coefficient
    results_df.coefficient = np.where(
        results_df["category_slug"].isin(user_categories_list),
        results_df.coefficient * 2.0,
        results_df.coefficient)

    results_df = results_df.rename(columns={'created_at_x': 'post_created_at'})

    logging.debug(results_df.columns)

    results_df = boost_by_article_freshness(results_df)

    results_df = results_df.set_index('slug')
    results_df = results_df.sort_values(by='coefficient', ascending=False)
    results_df = results_df['coefficient']
    results_df = results_df.rename_axis('slug').reset_index()

    hybrid_recommended_json = results_df.to_json(orient='records')
    parsed = json.loads(hybrid_recommended_json)
    hybrid_recommended_json = json.dumps(parsed)
    logging.debug(hybrid_recommended_json)

    return hybrid_recommended_json
