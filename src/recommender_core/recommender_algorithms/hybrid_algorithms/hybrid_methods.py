import gc
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix
from gensim.similarities.annoy import AnnoyIndexer
from sklearn.preprocessing import OneHotEncoder

from src.recommender_core.recommender_algorithms.user_based_algorithms.user_keywords_recommendation import \
    UserBasedMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.models_manipulation.models_loaders import \
    load_doc2vec_model
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.learn_to_rank.learn_to_rank_methods import preprocess_one_hot, \
    make_post_feature, train_lightgbm_user_based


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


def get_posts_lightgbm(results, use_categorical_columns=True):
    one_hot_encoder, categorical_columns_after_encoding = None, None

    consider_only_top_limit = 20
    if use_categorical_columns is True:
        one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int32)

    features = ["user_id", "coefficient", "relevance", "relevance_val", "views", "model_name"]
    categorical_columns = [
        'category_title', 'model_name'
    ]

    results['coefficient'] = results['coefficient'].astype(np.float16)

    print("tf_idf_results.dtypes:")
    print(results.dtypes)

    post_category_df = prepare_categories()

    print("results.columns:")
    print(results.columns)
    print("post_category_df.columns:")
    print(post_category_df.columns)

    results = results.merge(post_category_df, left_on='slug', right_on='slug')
    results = results.rename({"doc2vec_representation": "doc2vec"}, axis=1)
    df2 = pd.DataFrame(results)
    doc2vec_column_name_base = "doc2vec_col_"

    df2.dropna(subset=['doc2vec'], inplace=True)

    df2['doc2vec'] = df2['doc2vec'].apply(lambda x: json.loads(x))
    df2 = pd.DataFrame(df2['doc2vec'].to_list(), index=df2.index).add_prefix(doc2vec_column_name_base)
    for column in df2.columns:
        df2[column] = df2[column].astype(np.float16)
    results = pd.concat([results, df2], axis=1)
    print("df2.dtypes")
    print(df2.dtypes)
    del df2
    gc.collect()

    #####
    # TODO: Find and fill missing Doc2Vec values (like in the training phase)

    tf_idf_results_old = results
    if use_categorical_columns is True:
        numerical_columns = [
            "coefficient", "views", 'doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3',
            'doc2vec_col_4', 'doc2vec_col_5',
            'doc2vec_col_6', 'doc2vec_col_7'
        ]
        one_hot_encoder.fit(post_category_df[categorical_columns])
        del post_category_df
        gc.collect()
        results = preprocess_one_hot(results, one_hot_encoder, numerical_columns,
                                     categorical_columns)
        results['slug'] = tf_idf_results_old['slug']
    else:
        del post_category_df
        gc.collect()

    # noinspection PyPep8Naming
    features_X = ['coefficient', 'views']

    all_columns = ['user_id', 'query_id', 'slug', 'query_slug', 'coefficient', 'relevance', 'post_id', 'title_x',
                   'excerpt', 'body', 'views', 'keywords', 'category', 'description', 'all_features_preprocessed',
                   'body_preprocessed']
    if use_categorical_columns is True:
        categorical_columns_after_encoding = [x for x in all_columns if x.startswith("category_")]
        features.extend(categorical_columns_after_encoding)
    if use_categorical_columns is True:
        features_X.extend(categorical_columns_after_encoding)
        features_X.extend(
            ['doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3', 'doc2vec_col_4', 'doc2vec_col_5',
             'doc2vec_col_6', 'doc2vec_col_7'])

    pred_df = make_post_feature(results)
    lightgbm_model_file = Path("models/lightgbm.pkl")
    if lightgbm_model_file.exists():
        model = pickle.load(open('models/lightgbm.pkl', 'rb'))
    else:
        print("LightGBMMethods model not found. Training from available relevance testing results datasets...")
        train_lightgbm_user_based()
        model = pickle.load(open('models/lightgbm.pkl', 'rb'))
    # noinspection PyPep8Naming
    predictions = model.predict(results[features_X])  # .values.reshape(-1,1) when single feature is used
    del results
    gc.collect()
    topk_idx = np.argsort(predictions)[::-1][:consider_only_top_limit]
    recommend_df = pred_df.loc[topk_idx].reset_index(drop=True)
    recommend_df['predictions'] = predictions

    recommend_df.sort_values(by=['predictions'], inplace=True, ascending=False)
    recommend_df = recommend_df[['slug', 'predictions']]
    recommend_df.to_json()
    result = recommend_df.to_json(orient="records")
    parsed = json.loads(result)
    return json.dumps(parsed, indent=4)


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


def get_most_similar_by_hybrid(user_id, posts_to_compare):
    """
    Get most similar from content based matrix and delivered posts.

    Parameters
    ----------
    posts_to_compare: i.e. svd_recommended_posts
    """
    list_of_slugs, list_of_slugs_from_history = select_list_of_posts_for_user(user_id, posts_to_compare)

    # TF-IDF
    similarity_matrix = get_similarity_matrix_tfidf(list_of_slugs, posts_to_compare, list_of_slugs_from_history)
    results_df_tfidf = convert_similarity_matrix_to_results_dataframe(similarity_matrix)
    # Word2Vec
    method = "word2vec"
    similarity_matrix = get_similarity_matrix_from_pairs_similarity(method, list_of_slugs, posts_to_compare,
                                                                    list_of_slugs_from_history)
    results_df_word2vec = convert_similarity_matrix_to_results_dataframe(similarity_matrix)
    # Doc2Vec
    method = "doc2vec"
    similarity_matrix = get_similarity_matrix_from_pairs_similarity(method, list_of_slugs, posts_to_compare,
                                                                    list_of_slugs_from_history)
    results_df_doc2vec = convert_similarity_matrix_to_results_dataframe(similarity_matrix)

    print("Results tfidf")
    print(results_df_tfidf)

    print("results_df_word2vec")
    print(results_df_word2vec)

    print("results_df_doc2vec")
    print(results_df_doc2vec)

    results_df = pd.concat([results_df_tfidf['slug'], results_df_tfidf['coefficient'],
                            results_df_word2vec['coefficient'], results_df_doc2vec['coefficient']],
                           axis=1, keys=['slug', 'coefficient_tfidf', 'coefficient_word2vec', 'coefficient_doc2vec'])
    print("results_df")
    print(results_df)

    cofficient_columns = ['coefficient_tfidf', 'coefficient_word2vec', 'coefficient_doc2vec']

    results_df[cofficient_columns] = (results_df[cofficient_columns] - results_df[cofficient_columns].mean()) \
                                     / results_df[cofficient_columns].std()
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

    recommend_methods = RecommenderMethods()
    user_categories = recommend_methods.get_user_categories(user_id)
    print("Categories for user " + str(user_id))
    print(user_categories)
    user_categories_list = user_categories['category_slug'].values.tolist()
    print("user_categories_list:")
    print(user_categories_list)

    results_df.coefficient = np.where(
        results_df["category_slug"].isin(user_categories_list),
        results_df.coefficient * 1.5,
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
        content_based_method = Word2VecClass()
        w2v_model = KeyedVectors.load(path_to_model.as_posix())
    elif method == "doc2vec":
        content_based_method = Doc2VecClass()
        d2v_model = load_doc2vec_model('models/d2v_full_text_limited.model')
    else:
        raise NotImplementedError("Method not supported.")

    if w2v_model is None or d2v_model is None:
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
