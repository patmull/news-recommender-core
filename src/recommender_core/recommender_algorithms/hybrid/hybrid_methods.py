import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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

    features_X = ['coefficient', 'views']

    all_columns = ['user_id', 'query_id', 'slug', 'query_slug', 'coefficient', 'relevance', 'id_x', 'title_x',
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
