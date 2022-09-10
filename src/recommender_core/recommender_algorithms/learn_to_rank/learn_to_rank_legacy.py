import ast
import gc
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import redis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRanker

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.user_based_algorithms.collaboration_based_recommendation import \
    SvdClass
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_based_recommendation import UserBasedRecommendation

REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

SEED = 2021


# TODO: Remove completely in the next code review
# noinspection DuplicatedCode
@DeprecationWarning
class LightGBM:
    tfidf = TfIdf()
    user_based_recommendation = UserBasedRecommendation()
    svd = SvdClass()

    feature_list = []

    def get_results_dataframe(self):
        recommender_methods = RecommenderMethods()
        return recommender_methods.get_relevance_results_dataframe()  # load posts to dataframe

    # TODO: Remove completely in the next code review
    # noinspection DuplicatedCode
    @DeprecationWarning
    def get_user_keywords_based(self, tfidf, user_based_recommendation, user_id):
        user_keywords = user_based_recommendation.get_user_keywords(user_id)
        keyword_list = user_keywords['keyword_name'].tolist()
        tfidf_keywords = ''
        if len(keyword_list) > 0:
            keywords = ' '.join(keyword_list)
            print(keywords)
            tfidf_keywords = tfidf.keyword_based_comparison(keywords, number_of_recommended_posts=20)

        return tfidf_keywords

    # TODO: Remove completely in the next code review
    # noinspection DuplicatedCode
    @DeprecationWarning
    def get_results_single_coeff_user_as_query(self):
        evaluation_results_df = self.get_results_dataframe()
        print("evaluation_results_df:")
        print(evaluation_results_df)
        dict_of_jsons = {}
        for index, row in evaluation_results_df.iterrows():
            dict_of_jsons[row['user_id']] = row['results_part_2']

        print("dict_of_jsons:")
        print(dict_of_jsons)
        dataframes = []
        for id, json_dict in dict_of_jsons.items():
            df_from_json = pd.DataFrame.from_dict(json_dict)
            print("df_from_json:")
            print(df_from_json.to_string())
            df_from_json['user_id'] = id
            dataframes.append(df_from_json)
        df_merged = pd.concat(dataframes, ignore_index=True)

        print("df_merged columns")
        print(df_merged.columns)

        df_merged = df_merged[['user_id', 'slug', 'coefficient', 'relevance']]
        # converting indexes to columns
        # df_merged.reset_index(level=['coefficient', 'relevance'], inplace=True)
        print("df_merged:")
        print(df_merged.to_string())
        print("cols:")
        print(df_merged.columns)
        print("index:")
        print(df_merged.index)
        return df_merged

    # TODO: Remove completely in the next code review
    # noinspection DuplicatedCode
    @DeprecationWarning
    def get_results_single_coeff_searched_doc_as_query(self):
        evaluation_results_df = self.get_results_dataframe()
        print("evaluation_results_df:")
        print(evaluation_results_df)
        dict_of_jsons = {}
        for index, row in evaluation_results_df.iterrows():
            dict_of_jsons[row['id']] = [row['results_part_2'], row['user_id'], row['query_slug'], row['model_name'],
                                        row['model_variant']]

        print("dict_of_jsons:")
        print(dict_of_jsons)
        dataframes = []
        for id, json_dict in dict_of_jsons.items():
            df_from_json = pd.DataFrame.from_dict(json_dict[0])

            df_from_json['query_id'] = id
            df_from_json['user_id'] = json_dict[1]
            df_from_json['query_slug'] = json_dict[2]
            df_from_json['model_name'] = json_dict[3]
            df_from_json['model_variant'] = json_dict[4]

            # converting binary relevance to 0-7 relevance and sorting by relevance
            df_from_json.sort_values(by=['relevance', 'coefficient'], inplace=True, ascending=False)
            df_from_json.reset_index(inplace=True)
            df_from_json['relevance_val'] = 0
            df_from_json.loc[0, ['relevance_val']] = 7
            df_from_json.loc[1, ['relevance_val']] = 6
            df_from_json.loc[2, ['relevance_val']] = 5
            df_from_json.loc[3:4, ['relevance_val']] = 4
            df_from_json.loc[5:6, ['relevance_val']] = 3
            df_from_json.loc[7:9, ['relevance_val']] = 2
            df_from_json.loc[10:13, ['relevance_val']] = 1
            df_from_json.loc[14:19, ['relevance_val']] = 0

            print("df_from_json:")
            print(df_from_json.to_string())

            dataframes.append(df_from_json)

        df_merged = pd.concat(dataframes, ignore_index=True)

        print("df_merged columns")
        print(df_merged.columns)

        df_merged = df_merged[

            ['user_id', 'query_id', 'slug', 'query_slug', 'coefficient', 'relevance', 'relevance_val', 'model_name', 'model_variant']]

        # converting indexes to columns
        # df_merged.reset_index(level=['coefficient', 'relevance'], inplace=True)
        print("df_merged:")
        print(df_merged.to_string())
        print("cols:")
        print(df_merged.columns)
        print("index:")
        print(df_merged.index)
        return df_merged

    def make_post_feature(self, df):
        # convert object to a numeric type, replacing Unknown with nan.
        df['coefficient'] = df['coefficient'].apply(lambda x: np.nan if x == 'Unknown' else float(x))

        # add genre ctegory columns
        # df = genre_to_category(df)

        return df

    def make_user_feature(self, df):
        df['rating_count'] = df.groupby('user_id')['slug'].transform('count')
        df['rating_mean'] = df.groupby('user_id')['relevance'].transform('mean')
        return df

    def preprocess(self, df):
        df = self.make_post_feature(df)
        merged_df = self.make_user_feature(df)
        return merged_df

    def preprocess_one_hot(self, df, one_hot_encoder, num_cols, cat_cols):
        df = df.copy()

        cat_one_hot_cols = one_hot_encoder.get_feature_names(cat_cols)

        df_one_hot = pd.DataFrame(
            one_hot_encoder.transform(df[cat_cols]),
            columns=cat_one_hot_cols
        )
        df_preprocessed = pd.concat([
            df[num_cols],
            df_one_hot
        ], axis=1)
        return df_preprocessed

    # TODO: Remove completely in the next code review
    # noinspection DuplicatedCode
    @DeprecationWarning
    def train_lightgbm_user_based(self):

        # TODO: Remove user id if it's needed
        df_results = self.get_results_single_coeff_searched_doc_as_query()

        recommender_methods = RecommenderMethods()

        post_category_df = recommender_methods.get_posts_categories_dataframe()
        post_category_df = post_category_df.rename(columns={'slug_x': 'slug'})
        post_category_df = post_category_df.rename(columns={'title_y': 'category'})

        print(df_results.columns)
        print(post_category_df.columns)

        categorical_columns = [
            "category", "model_name", "model_variant"
        ]

        numerical_columns = [
            "user_id", "coefficient", "relevance_val", "views"
        ]

        df_results_merged = df_results.merge(post_category_df, on='slug')
        print("df_results_merged.columns")
        print(df_results_merged.columns)

        time.sleep(60)

        print("Loading Doc2Vec model...")
        df_results_merged = df_results_merged.rename({"doc2vec_representation": "doc2vec"}, axis=1)
        print("df_results_merged:")
        print(df_results_merged.to_string())
        df2 = pd.DataFrame(df_results_merged)
        print("df2:")
        print(df2.to_string())
        """
        print("Searching for Doc2Vec missing values...")
        df2['doc2vec'] = df2.apply(lambda row: json.dumps(doc2vec.get_vector_representation(row['slug']).tolist()) if pd.isnull(row['doc2vec']) else row['doc2vec'], axis=1)
        """
        print("doc2vec:")
        print(df2['doc2vec'])
        print("Removing rows with Doc2Vec still set to None")
        df2.dropna(subset=['doc2vec'], inplace=True)
        print("df2 after dropna:")
        print(df2)
        df2['doc2vec'] = df2['doc2vec'].apply(lambda x: json.loads(x))
        doc2vec_column_name_base = "doc2vec_col_"
        df2 = pd.DataFrame(df2['doc2vec'].to_list(), index=df2.index).add_prefix(doc2vec_column_name_base)
        df_results_merged = pd.concat([df_results_merged, df2], axis=1)
        print("df_results_merged.dtypes:")
        print(df_results_merged.dtypes)
        # df_results_merged = df_results_merged.columns.drop("doc2vec")

        # noinspection DuplicatedCode
        df_results_merged_old = df_results_merged

        print("Splitting dataset.")
        features = ["user_id", "coefficient", "relevance_val", "views"]
        train_df, validation_df = train_test_split(df_results_merged, test_size=0.2)

        print("Normalizing coeffficient and views")
        train_df[['coefficient', 'views']] = train_df[['coefficient', 'views']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int32)
        one_hot_encoder.fit(df_results_merged[categorical_columns])

        df_results_merged = self.preprocess_one_hot(df_results_merged, one_hot_encoder, numerical_columns, categorical_columns)

        df_results_merged['query_slug'] = df_results_merged_old['query_slug']
        df_results_merged['slug'] = df_results_merged_old['slug']

        # df_unseen = df_results_merged.iloc[:20,:]
        # df_results_merged = df_results_merged.iloc[20:,:]

        all_columns_of_train_df = train_df.columns.values.tolist()
        print("Columns values:")
        print(train_df.columns.values.tolist())

        # title, excerpt --> doc2vec
        features.extend(['doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3', 'doc2vec_col_4', 'doc2vec_col_5', 'doc2vec_col_6', 'doc2vec_col_7'])
        # category --> OneHotEncoding (each category its own column, binary values)
        categorical_columns_after_encoding = [x for x in all_columns_of_train_df if x.startswith("category_")]
        features.extend(categorical_columns_after_encoding)
        print('number of one hot encoded categorical columns: ',
              len(one_hot_encoder.get_feature_names(categorical_columns)))

        print("train_df")
        print(train_df)
        train_df = train_df[features]
        validation_df = validation_df[features]

        print("train_df")
        print(train_df)

        print("train_df after hot encoding")
        print(train_df.to_string())

        print("validation_df after hot encoding")
        print(validation_df.to_string())

        query_train = train_df.groupby("user_id")["user_id"].count().to_numpy()
        query_validation = validation_df.groupby("user_id")["user_id"].count().to_numpy()
        # query_test = [test_df.shape[0] / 2000] * 2000

        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            min_child_samples=1
        )

        features_X = ['coefficient', 'views']
        features_X.extend(categorical_columns_after_encoding)
        features_X.extend(['doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3', 'doc2vec_col_4', 'doc2vec_col_5', 'doc2vec_col_6', 'doc2vec_col_7'])

        print("features_X")
        print(features_X)

        model.fit(train_df[features_X], train_df[['relevance_val']],
                             group=query_train,
                             verbose=10,
                             eval_set=[(validation_df[features_X], validation_df[['relevance_val']])],
                             eval_group=[query_validation],
                             eval_at=10, # Make evaluation for target=1 ranking, I choosed arbitrarily
                  )

        pickle.dump(model, open('models/lightgbm.pkl', 'wb'))

    # TODO: Remove completely in the next code review
    # noinspection DuplicatedCode
    @DeprecationWarning
    def get_posts_lightgbm(self, slug, variant="full-text", use_categorical_columns=True):
        global one_hot_encoder, categorical_columns_after_encoding
        consider_only_top_limit = 20
        if use_categorical_columns is True:
            one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int32)

        features = ["user_id", "coefficient", "relevance", "relevance_val", "views", "model_name", "model_variant"]
        categorical_columns = [
            'category_title', 'model_name', "model_variant"
        ]

        print("Loading TfIdf results")

        tf_idf = TfIdf()
        tf_idf_results = tf_idf.get_prefilled_full_text(slug, variant)
        tf_idf_results = ast.literal_eval(tf_idf_results)
        json_data = json.loads(json.dumps(tf_idf_results))
        tf_idf_results = pd.json_normalize(json_data)
        tf_idf_results['coefficient'] = tf_idf_results['coefficient'].astype(np.float16)

        print("tf_idf_results.dtypes:")
        print(tf_idf_results.dtypes)

        # noinspection DuplicatedCode
        recommender_methods = RecommenderMethods()
        gc.collect()
        post_category_df = recommender_methods.get_posts_categories_dataframe()

        gc.collect()

        post_category_df = post_category_df.rename(columns={'slug_x': 'slug'})
        post_category_df = post_category_df.rename(columns={'title_y': 'category'})

        post_category_df['model_name'] = 'tfidf'
        post_category_df['model_variant'] = 'tfidf-full-text'

        print("post_category_df.dtypes:")
        print(post_category_df.dtypes)

        tf_idf_results = tf_idf_results.merge(post_category_df, on='slug')

        # noinspection DuplicatedCode
        tf_idf_results = tf_idf_results.rename({"doc2vec_representation": "doc2vec"}, axis=1)
        df2 = pd.DataFrame(tf_idf_results)
        doc2vec_column_name_base = "doc2vec_col_"

        df2.dropna(subset=['doc2vec'], inplace=True)

        df2['doc2vec'] = df2['doc2vec'].apply(lambda x: json.loads(x))
        df2 = pd.DataFrame(df2['doc2vec'].to_list(), index=df2.index).add_prefix(doc2vec_column_name_base)
        for column in df2.columns:
            df2[column] = df2[column].astype(np.float16)
        tf_idf_results = pd.concat([tf_idf_results, df2], axis=1)
        print("df2.dtypes")
        print(df2.dtypes)
        del df2
        gc.collect()

        #####
        # TODO: Find and fill missing Doc2Vec values (like in the training phase)

        tf_idf_results_old = tf_idf_results
        if use_categorical_columns is True:
            numerical_columns = [
                "coefficient", "views", 'doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3', 'doc2vec_col_4', 'doc2vec_col_5',
                 'doc2vec_col_6', 'doc2vec_col_7'
            ]
            one_hot_encoder.fit(post_category_df[categorical_columns])
            del post_category_df
            gc.collect()
            tf_idf_results = self.preprocess_one_hot(tf_idf_results, one_hot_encoder, numerical_columns,
                                                     categorical_columns)
            tf_idf_results['slug'] = tf_idf_results_old['slug']
        else:
            del post_category_df
            gc.collect()

        features_X = ['coefficient', 'views']

        all_columns = ['user_id', 'query_id', 'slug', 'query_slug', 'coefficient', 'relevance', 'id_x', 'title_x', 'excerpt', 'body', 'views', 'keywords', 'category', 'description', 'all_features_preprocessed', 'body_preprocessed']
        if use_categorical_columns is True:
            categorical_columns_after_encoding = [x for x in all_columns if x.startswith("category_")]
            features.extend(categorical_columns_after_encoding)
        if use_categorical_columns is True:
            features_X.extend(categorical_columns_after_encoding)
            features_X.extend(
                ['doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3', 'doc2vec_col_4', 'doc2vec_col_5',
                 'doc2vec_col_6', 'doc2vec_col_7'])

        pred_df = self.make_post_feature(tf_idf_results)
        lightgbm_model_file = Path("models/lightgbm.pkl")
        if lightgbm_model_file.exists():
            model = pickle.load(open('models/lightgbm.pkl', 'rb'))
        else:
            print("LightGBMMethods model not found. Training from available relevance testing results datasets...")
            self.train_lightgbm_user_based()
            model = pickle.load(open('models/lightgbm.pkl', 'rb'))
        predictions = model.predict(tf_idf_results[features_X])  # .values.reshape(-1,1) when single feature is used
        del tf_idf_results
        gc.collect()
        topk_idx = np.argsort(predictions)[::-1][:consider_only_top_limit]
        recommend_df = pred_df.loc[topk_idx].reset_index(drop=True)
        recommend_df['predictions'] = predictions
        # df_unseen['predictions'] = predictions
        # print("df_unseen:")
        # print(df_unseen.to_string())
        # recommend_df = recommend_df.loc[recommend_df['user_id'].isin([user_id])]
        # recommend_df = df_unseen.loc[df_unseen['query_slug'].isin([slug])]
        recommend_df.sort_values(by=['predictions'], inplace=True, ascending=False)
        recommend_df = recommend_df[['slug', 'predictions']]
        recommend_df.to_json()
        result = recommend_df.to_json(orient="records")
        parsed = json.loads(result)
        return json.dumps(parsed, indent=4)

    # TODO: Remove completely in the next code review
    # noinspection DuplicatedCode
    @DeprecationWarning
    def train_lightgbm_document_based(self, slug, k=20):

        df_results = self.get_results_single_coeff_searched_doc_as_query()
        dataframe_length = len(df_results.index)
        post_features = ["slug"]
        features = ["coefficient", "relevance"]
        col_use = [c for c in df_results.columns if c not in features]
        split_train = int(dataframe_length * 0.8)
        split_validation = int(dataframe_length - split_train)
        train_df = df_results[:split_train]  # first 80%
        train_df = train_df[["query_id", "coefficient", "relevance"]]
        validation_df = df_results[split_validation:]  # remaining 20%
        validation_df = validation_df[["query_id", "coefficient", "relevance"]]

        print("train_df")
        print(train_df)

        query_train = train_df.groupby("query_id")["query_id"].count().to_numpy()
        query_val = validation_df.groupby("query_id")["query_id"].count().to_numpy()
        # query_test = [test_df.shape[0] / 2000] * 2000

        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg"
        )

        model.fit(train_df[['coefficient']], train_df[['relevance']],
                             group=query_train,
                             verbose=10,
                             eval_set=[(validation_df[['coefficient']], validation_df[['relevance']])],
                             eval_group=[query_val],
                             eval_at=10, # Make evaluation for target=1 ranking, I choosed arbitrarily
                         )

        evaluation_results_df = self.get_results_dataframe()
        evaluation_results_df = evaluation_results_df.rename(columns={'id': 'query_id'})

        consider_only_top_limit = 1000
        df_results_merged = pd.merge(df_results, evaluation_results_df, on='query_id', how='right')
        pred_df = self.make_post_feature(df_results_merged)
        preds = model.predict(validation_df['coefficient'].values.reshape(-1,1))
        topk_idx = np.argsort(preds)[::-1][:consider_only_top_limit]
        recommend_df = pred_df.loc[topk_idx].reset_index(drop=True)
        print("evaluation_results_df:")
        print(evaluation_results_df.to_string())
        print("recommend_df:")
        print(recommend_df.to_string())
        recommend_df = recommend_df.loc[recommend_df['query_slug'].isin([slug])]
        print('---------- Recommend ----------')
        print(recommend_df.to_string())

        """
        # TODO: Repeated trial for avg execution time
        start_time = time.time()
        tfidf_posts_full = self.get_tfidf(self.tfidf, slug)
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        tfidf_keywords = self.get_user_keywords_based(self.tfidf, self.user_based_recommendation, self.user_id)
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        doc2vec_posts = self.get_doc2vec(self.doc2vec, slug)
        print("--- %s seconds ---" % (time.time() - start_time))
        """

    def get_posts_df(self):
        database = DatabaseMethods()
        posts_df = database.get_posts_dataframe()
        return posts_df

    def get_categories_df(self):
        database = DatabaseMethods()
        database.connect()
        posts_df = database.get_categories_dataframe()
        database.disconnect()
        return posts_df


class LearnToRank:

    # noinspection DuplicatedCode
    def get_user_keywords_based(self, tfidf, user_based_recommendation, user_id):
        # noinspection DuplicatedCode
        learn_to_rank = LearnToRank()
        return learn_to_rank.get_user_keywords_based()

    def get_tfidf(self, tfidf, slug):
        tfidf.prepare_dataframes()
        tfidf_prefilled_posts = tfidf.get_prefilled_full_text()
        print("tfidf_prefilled_posts:")
        print(tfidf_prefilled_posts)
        found_row = tfidf_prefilled_posts.loc[tfidf_prefilled_posts['slug_x'] == slug]
        tfidf_results_json = json.loads(found_row['recommended_tfidf_full_text'].iloc[0])
        tfidf_results_df = pd.json_normalize(tfidf_results_json)
        print("tfidf_results_df:")
        print(tfidf_results_df)
        return tfidf_results_df

    def intersect(self, a, b):
        return pd.merge(a, b, how='inner', on=['slug'])

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def redis_test(self):
        r = redis.Redis(host='redis-10115.c3.eu-west-1-2.ec2.cloud.redislabs.com', port=10115, db=0, username="admin",
                        password=REDIS_PASSWORD)
        r.set('foo', 'bar')
        print(r.get('foo'))

