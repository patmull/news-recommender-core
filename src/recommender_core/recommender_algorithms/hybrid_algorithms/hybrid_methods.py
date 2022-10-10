import gc
import json
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
        raise ValueError("'posts_to_compare' parameter must be a list!")

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


# NOTICE: It would be possible to use @typechecked from typeguard here
def get_most_similar_by_hybrid(user_id: int, posts_to_compare=None, list_of_methods=None):
    """
    Get most similar from content based matrix and delivered posts.

    Parameters
    ----------
    posts_to_compare: i.e. svd_recommended_posts; if not supplied, it will calculate fresh SVD
    user_id: int by user id from DB
    @param posts_to_compare:
    @param user_id:
    @param list_of_methods:
    """

    if type(user_id) is not int:
        raise TypeError("User id muse be an int")

    list_of_supported_methods = ['tfidf', 'doc2vec', 'word2vec']
    if list_of_methods is None:
        list_of_methods = ['tfidf', 'doc2vec', 'word2vec']
    elif not set(list_of_methods).issubset(list_of_supported_methods) > 0:
        raise NotImplementedError("Inserted methods must correspond to DB columns.")
    if posts_to_compare is None:
        svd = SvdClass()
        recommended_by_svd = svd.run_svd(user_id=user_id, dict_results=False, num_of_recommendations=5)
        posts_to_compare = recommended_by_svd['slug'].to_list()

    list_of_slugs, list_of_slugs_from_history = select_list_of_posts_for_user(user_id, posts_to_compare)

    list_of_similarity_results = []
    for method in list_of_methods:
        if method == "tfidf":
            similarity_matrix = get_similarity_matrix_tfidf(list_of_slugs, posts_to_compare, list_of_slugs_from_history)
            similarity_matrix = similarity_matrix * 1.75
            results = convert_similarity_matrix_to_results_dataframe(similarity_matrix)
        elif method == "doc2vec" or "word2vec":
            similarity_matrix = get_similarity_matrix_from_pairs_similarity(method, list_of_slugs, posts_to_compare,
                                                                            list_of_slugs_from_history)
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

    print("results_df:")
    print(results_df)

    results_df = results_df.merge(df_posts_categories, left_on='slug', right_on='slug')
    print("results_df after merge")
    print(results_df)

    user_methods = UserMethods()
    user_categories = user_methods.get_user_categories(user_id)
    print("Categories for user " + str(user_id))
    print(user_categories)
    user_categories_list = user_categories['category_slug'].values.tolist()
    print("user_categories_list:")
    print(user_categories_list)

    results_df.coefficient = np.where(
        results_df["category_slug"].isin(user_categories_list),
        results_df.coefficient * 2.0,
        results_df.coefficient)

    results_df = results_df.set_index('slug')
    results_df = results_df.sort_values(by='coefficient', ascending=False)
    results_df = results_df['coefficient']
    results_df = results_df.rename_axis('slug').reset_index()

    hybrid_recommended_json = results_df.to_json(orient='records')
    parsed = json.loads(hybrid_recommended_json)
    hybrid_recommended_json = json.dumps(parsed)
    print(hybrid_recommended_json)

    return hybrid_recommended_json


def drop_columns_from_similarity_matrix(similarity_matrix, posts_to_compare, list_of_slugs_from_history):
    similarity_matrix = similarity_matrix.drop(columns=posts_to_compare)
    similarity_matrix = similarity_matrix.drop(list_of_slugs_from_history)
    return similarity_matrix


def get_similarity_matrix_tfidf(list_of_slugs, posts_to_compare, list_of_slugs_from_history):
    tfidf = TfIdf()

    similarity_matrix = tfidf.get_similarity_matrix(list_of_slugs)

    print("Similarity matrix:")
    print(similarity_matrix)
    print("Similarity matrix type:")
    print(type(similarity_matrix))

    similarity_matrix = drop_columns_from_similarity_matrix(similarity_matrix, posts_to_compare,
                                                            list_of_slugs_from_history)

    print("similarity_matrix:")
    print(similarity_matrix)
    print(similarity_matrix.columns)

    return similarity_matrix


def get_similarity_matrix_from_pairs_similarity(method, list_of_slugs, posts_to_compare,
                                                list_of_slugs_from_history):
    w2v_model, d2v_model = None, None

    if method == "word2vec":
        path_to_model = Path("full_models/idnes/evaluated_models/word2vec_model_3/w2v_idnes.model")
        method_class = Word2VecClass()
        w2v_model = KeyedVectors.load(path_to_model.as_posix())
    elif method == "doc2vec":
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

    print("Similarity matrix:")
    print(similarity_matrix)

    print("Similarity matrix type:")
    print(type(similarity_matrix))

    similarity_matrix = similarity_matrix.drop(columns=posts_to_compare)
    similarity_matrix = similarity_matrix.drop(list_of_slugs_from_history)

    print("similarity_matrix:")
    print(similarity_matrix)
    print(similarity_matrix.columns)

    return similarity_matrix
