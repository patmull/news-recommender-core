import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from src.recommender_core.data_handling.model_methods.user_methods import UserMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.models_manipulation.models_loaders import \
    load_doc2vec_model
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.recommender_algorithms.user_based_algorithms.collaboration_based_recommendation \
    import SvdClass
from src.recommender_core.data_handling.data_queries import RecommenderMethods

LIST_OF_SUPPORTED_METHODS = ['tfidf', 'doc2vec', 'word2vec']
SIM_MATRIX_OF_ALL_POSTS_PATH = Path('precalc_matrices/sim_matrix_all_posts')


# noinspection DuplicatedCode
def model_selection(tfidf_rate, word2vec_rate, doc2vec_rate, lda_rate, number_of_recommended,
                    tf_idf_results, word2vec_results, doc2vec_results, lda_results):
    number_of_recommended_tfidf = round((tfidf_rate / 100) * number_of_recommended)
    number_of_recommended_word2vec = round((word2vec_rate / 100) * number_of_recommended)
    number_of_recommended_doc2vec = round((doc2vec_rate / 100) * number_of_recommended)
    number_of_recommended_lda = round((lda_rate / 100) * number_of_recommended)

    print("number_of_recommended_tfidf")
    print(number_of_recommended_tfidf)
    print("number_of_recommended_word2vec")
    print(number_of_recommended_word2vec)
    print("number_of_recommended_doc2vec")
    print(number_of_recommended_doc2vec)
    print("number_of_recommended_lda")
    print(number_of_recommended_lda)

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
    return list_of_slugs, list_of_slugs_from_history


def convert_similarity_matrix_to_results_dataframe(similarity_matrix):
    similarity_matrix['coefficient'] = similarity_matrix.sum(axis=1)
    results_df = similarity_matrix.sort_values(by='coefficient', ascending=False)
    results_df = results_df['coefficient']
    results_df = results_df.rename_axis('slug').reset_index()
    print("similarity_matrix after sorting:")
    print(results_df)
    return results_df


def drop_columns_from_similarity_matrix(similarity_matrix, posts_to_compare, list_of_slugs_from_history):
    similarity_matrix = similarity_matrix.drop(columns=posts_to_compare)
    similarity_matrix = similarity_matrix.drop(list_of_slugs_from_history)
    return similarity_matrix


def get_similarity_matrix_from_pairs_similarity(method, list_of_slugs):
    w2v_model, d2v_model = None, None

    logging.debug("Calculating sim matrice for %d posts:" % (len(list_of_slugs)))

    if method == "tfidf":
        logging.debug('Calculating sim matrix for TF-IDF')
        tfidf = TfIdf()

        similarity_matrix = tfidf.get_similarity_matrix(list_of_slugs)

        print("Similarity matrix:")
        print(similarity_matrix)
        print("Similarity matrix type:")
        print(type(similarity_matrix))

        # TODO: Everything above can be pre-computed and loaded. Priority: VERY HIGH
    else:
        if method == "word2vec":
            logging.debug('Calculating sim matrix for Word2Vec')

            path_to_model = Path("full_models/idnes/evaluated_models/word2vec_model_3/w2v_idnes.model")
            method_class = Word2VecClass()
            w2v_model = KeyedVectors.load(path_to_model.as_posix())
        elif method == "doc2vec":
            logging.debug('Calculating sim matrix for Doc2Vec')

            method_class = Doc2VecClass()
            d2v_model = load_doc2vec_model('models/d2v_full_text_limited.model')
        else:
            raise NotImplementedError("Method not supported.")

        content_based_method = method_class

        if w2v_model is None and d2v_model is None:
            raise ValueError("Word2Vec and Doc2Vec variables are set to None. Cannot continue.")

        similarity_list = []
        for x in list_of_slugs:
            inner_list = []
            for y in list_of_slugs:
                if method == "word2vec":
                    inner_list.append(content_based_method.get_pair_similarity_word2vec(x, y, w2v_model))
                elif method == "doc2vec":
                    inner_list.append(content_based_method.get_pair_similarity_doc2vec(x, y, d2v_model))
            similarity_list.append(inner_list)
            print("similarity_list:")
            print(similarity_list)

        print("similarity_list:")
        print(similarity_list)

        similarity_matrix = pd.DataFrame(similarity_list, columns=list_of_slugs, index=list_of_slugs)
        # TODO: Everything above can be pre-computed and loaded. Priority: VERY HIGH

        print("Similarity matrix:")
        print(similarity_matrix)

        print("Similarity matrix type:")
        print(type(similarity_matrix))

    return similarity_matrix


def personalize_similarity_matrix(similarity_matrix, posts_to_compare, list_of_slugs_from_history):
    """
    Drops the articles from user read history from similarity matrix and drops articles from SVD (collab element) from axe.
    @return:
    """
    similarity_matrix = drop_columns_from_similarity_matrix(similarity_matrix, posts_to_compare,
                                                            list_of_slugs_from_history)

    return similarity_matrix


# TODO: Add to prefillers.
def precalculate_and_save_sim_matrix_for_all_posts():
    recommender_methods = RecommenderMethods()
    all_posts = recommender_methods.get_posts_dataframe()
    all_posts_slugs = all_posts['slug'].values.tolist()
    for method in LIST_OF_SUPPORTED_METHODS:
        similarity_matrix_of_all_posts = get_similarity_matrix_from_pairs_similarity(method=method,
                                                                                     list_of_slugs=all_posts_slugs)
        similarity_matrix_of_all_posts.to_feather("%s_%s.feather".format(SIM_MATRIX_OF_ALL_POSTS_PATH.as_posix(), method))


# NOTICE: It would be possible to use @typechecked from typeguard here
def load_posts_from_sim_matrix(method, list_of_slugs):
    """

    @param method:
    @param list_of_slugs: slugs delivered from SVD algorithm = slugs that we are interested in
    @return:
    """
    sim_matrix = pd.read_feather("%s_%s.feather".format(SIM_MATRIX_OF_ALL_POSTS_PATH.as_posix(), method))
    # select from column and rows only desired articles
    sim_matrix = sim_matrix.loc[list_of_slugs]
    sim_matrix = sim_matrix[list_of_slugs]
    return sim_matrix


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
    @param save_result: saves the results (i.e. for debugging, this can be loaded with load_saved_result method below). Added to help with debugging of final boosting
    @param load_saved_result: if True, skips the recommending calculation and jumps to final calculations. Added to help with debugging of final boosting
    """
    path_to_save_results = Path('research/hybrid/results_df.pkl')

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
        for method in list_of_methods:
            if method == "tfidf":
                constant = 1.75
                # TODO: Derive from loaded feather of sim matrix instead
                if load_from_precalc_sim_matrix and os.path.exists("%s_%s.feather".format(SIM_MATRIX_OF_ALL_POSTS_PATH
                                                                                                  .as_posix(), method)):
                    # Loading posts we are interested in from pre-calculated similarity matrix
                    similarity_matrix = load_posts_from_sim_matrix(method, list_of_slugs)
                else:
                    # Calculating new similarity matrix only based on posts we are interested
                    similarity_matrix = get_similarity_matrix_from_pairs_similarity(method, list_of_slugs)
                similarity_matrix = personalize_similarity_matrix(similarity_matrix, svd_posts_to_compare,
                                                                  list_of_slugs_from_history)
                similarity_matrix = similarity_matrix * constant
                results = convert_similarity_matrix_to_results_dataframe(similarity_matrix)
            elif method == "doc2vec" or "word2vec":
                # TODO: Derive from loaded feather of sim matrix instead
                if load_from_precalc_sim_matrix and os.path.exists("%s_%s.feather".format(SIM_MATRIX_OF_ALL_POSTS_PATH
                                                                                                  .as_posix(), method)):
                    # Loading posts we are interested in from pre-calculated similarity matrix
                    similarity_matrix = load_posts_from_sim_matrix(method, list_of_slugs)
                else:
                    # Calculating new similarity matrix only based on posts we are interested
                    similarity_matrix = get_similarity_matrix_from_pairs_similarity(method, list_of_slugs)
                similarity_matrix = personalize_similarity_matrix(similarity_matrix, svd_posts_to_compare, list_of_slugs_from_history)
                if method == "doc2vec":
                    constant = 1.7
                elif method == "word2vec":
                    constant = 1.85
                else:
                    raise NotImplementedError("Supplied method not implemented")
                similarity_matrix = similarity_matrix * constant
                results = convert_similarity_matrix_to_results_dataframe(similarity_matrix)
            else:
                raise NotImplementedError("Supplied method not implemented")
            list_of_similarity_results.append(results)
            print("list_of_similarity_matrices:")
            print(list_of_similarity_results)
        print("list_of_similarity_matrices after finished for loop:")
        print(list_of_similarity_results)
        list_of_prefixed_methods = ['coefficient_' + x for x in list_of_methods if not str(x) == "nan"]
        list_of_keys = ['slug'] + list_of_prefixed_methods

        print("list_of_similarity_results[0]:")
        print(list_of_similarity_results[0])

        results_df = pd.concat([list_of_similarity_results[0]['slug']], axis=1)
        for result in list_of_similarity_results:
            results_df = pd.concat([results_df, result['coefficient']], axis=1)

        results_df.columns = list_of_keys

        print("results_df")
        print(results_df)

        coefficient_columns = list_of_prefixed_methods

        results_df[coefficient_columns] = (results_df[coefficient_columns] - results_df[coefficient_columns].mean()) \
                                          / results_df[coefficient_columns].std()
        print("normalized_df:")
        print(results_df)
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
    print("Categories for user " + str(user_id))
    print(user_categories)
    user_categories_list = user_categories['category_slug'].values.tolist()
    print("user_categories_list:")
    print(user_categories_list)

    # If post contains user category, then boost the coefficient
    results_df.coefficient = np.where(
        results_df["category_slug"].isin(user_categories_list),
        results_df.coefficient * 2.0,
        results_df.coefficient)

    # TODO: Boost posts based on freshness (see Document 'Hybridní algoritmus.docx'). Priority: HIGH
    def boost_coefficient(coeff_value, boost):
        d = coeff_value * boost
        return d

    results_df = results_df.rename(columns={'created_at_x': 'post_created_at'})

    logging.debug(results_df.columns)

    now = pd.to_datetime('now')
    results_df['coefficient'] = results_df.apply(
        lambda x: boost_coefficient(x['coefficient'], 15)
        if ((now - x['post_created_at']) < pd.Timedelta(1, 'h'))
        else (boost_coefficient(x['coefficient'], 10)
              if ((now - x['post_created_at']) < pd.Timedelta(1, 'd'))
              else (boost_coefficient(x['coefficient'], 8)
                    if ((now - x['post_created_at']) < pd.Timedelta(5, 'd'))
                    else (boost_coefficient(x['coefficient'], 1))
                    )
              ), axis=1
    )

    results_df = results_df.set_index('slug')
    results_df = results_df.sort_values(by='coefficient', ascending=False)
    results_df = results_df['coefficient']
    results_df = results_df.rename_axis('slug').reset_index()

    hybrid_recommended_json = results_df.to_json(orient='records')
    parsed = json.loads(hybrid_recommended_json)
    hybrid_recommended_json = json.dumps(parsed)
    print(hybrid_recommended_json)

    return hybrid_recommended_json
