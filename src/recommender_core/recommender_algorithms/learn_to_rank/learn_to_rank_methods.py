import itertools
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import redis
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRanker
from src.recommender_core.recommender_algorithms.hybrid import evaluation_results
from src.recommender_core.recommender_algorithms.user_based_algorithms.collaboration_based_recommendation import SvdClass
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.data_handling.data_manipulation import Database
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.lda import Lda
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_based_recommendation import UserBasedRecommendation
from sklearn.linear_model import LogisticRegression

import optuna
import seaborn

"""
from dask.distributed import Client

Engine.put("dask")  # Modin will use Ray
os.environ["MODIN_ENGINE"] = "dask"

pd.DEFAULT_NPARTITIONS = 10
"""

REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
NUM_OF_POSTS = 8194

SEED = 2021


class LightGBM:
    tfidf = TfIdf()
    doc2vec = Doc2VecClass()
    lda = Lda()
    user_based_recommendation = UserBasedRecommendation()
    svd = SvdClass()

    feature_list = []

    def get_user_keywords_based(self, tfidf, user_id):
        recommender_methods = RecommenderMethods()
        user_keywords = recommender_methods.get_user_keywords(user_id)
        keyword_list = user_keywords['keyword_name'].tolist()
        tfidf_keywords = ''
        if len(keyword_list) > 0:
            keywords = ' '.join(keyword_list)
            print(keywords)
            tfidf_keywords = tfidf.keyword_based_comparison(keywords, number_of_recommended_posts=20)

        return tfidf_keywords

    def get_results_single_coeff_user_as_query(self):
        evaluation_results_df = evaluation_results.get_results_dataframe()
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

    def get_results_single_coeff_searched_doc_as_query(self):
        evaluation_results_df = evaluation_results.get_results_dataframe()
        print("evaluation_results_df:")
        print(evaluation_results_df)
        dict_of_jsons = {}
        for index, row in evaluation_results_df.iterrows():
            dict_of_jsons[row['id']] = [row['results_part_2'], row['user_id'], row['query_slug'], row['model_name']]

        print("dict_of_jsons:")
        print(dict_of_jsons)
        dataframes = []
        for id, json_dict in dict_of_jsons.items():
            df_from_json = pd.DataFrame.from_dict(json_dict[0])

            df_from_json['query_id'] = id
            df_from_json['user_id'] = json_dict[1]
            df_from_json['query_slug'] = json_dict[2]
            df_from_json['model_name'] = json_dict[3]

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

        df_merged = df_merged[['user_id', 'query_id', 'slug', 'query_slug', 'coefficient', 'relevance', 'relevance_val', 'model_name']]
        # converting indexes to columns
        # df_merged.reset_index(level=['coefficient', 'relevance'], inplace=True)
        print("df_merged:")
        print(df_merged.to_string())
        print("cols:")
        print(df_merged.columns)
        print("index:")
        print(df_merged.index)
        return df_merged

    def get_tfidf(self, tfidf, post_slug):
        tfidf.get_prefilled_full_text()
        tfidf_prefilled_posts = tfidf.get_prefilled_full_text()
        print("tfidf_prefilled_posts:")
        print(tfidf_prefilled_posts)
        found_row = tfidf_prefilled_posts.loc[tfidf_prefilled_posts['slug'] == post_slug]
        tfidf_results_json = json.loads(found_row['recommended_tfidf_full_text'].iloc[0])
        tfidf_results_df = pd.json_normalize(tfidf_results_json)
        print("tfidf_results_df:")
        print(tfidf_results_df)
        return tfidf_results_df

    def get_doc2vec(self, doc2vec, post_slug):
        doc2vec_posts = doc2vec.get_prefilled_full_text(post_slug)
        doc2vec_posts_full = doc2vec.get_similar_doc2vec(post_slug, number_of_recommended_posts=NUM_OF_POSTS)
        return doc2vec_posts_full

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

    # try parameter tuning
    def objective(self, trial):
        # search param
        param = {
            'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
            # 'subsample': trial.suggest_uniform('subsample', 1e-8, 1),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        # train model
        model = LGBMRanker(n_estimators=1000, **param, random_state=SEED, )
        model.fit(
            self.train['coefficient'],
            self.train[self.target_col],
            group=self.train_query,
            eval_set=[(self.test['coefficient'], self.test[self.target_col])],
            eval_group=[list(self.test_query)],
            eval_at=[1, 3, 5, 10, 20],  # calc validation ndcg@1,3,5,10,20
            early_stopping_rounds=50,
            verbose=10
        )

        # maximize mean ndcg
        scores = []
        for name, score in model.best_score_['valid_0'].items():
            scores.append(score)
        return np.mean(scores)

    def preprocess(self, df):
        df = self.make_post_feature(df)
        merged_df = self.make_user_feature(df)
        return merged_df

    def recommend_for_user(self, user, k, sample_anime_num):
        database = Database()
        posts_df = database.get_posts_dataframe_from_sql()
        pred_df = posts_df.sample(sample_anime_num).reset_index(drop=True)  # sample recommend candidates
        results_df = self.get_results_single_coeff_user_as_query()

        # preprocess for model prediction
        user_df = results_df.query('user_id==@user')
        user_df = self.make_user_feature(user_df)
        for col in user_df.columns:
            if col in self.features:
                pred_df[col] = user_df[col].values[0]
        pred_df = self.make_post_feature(pred_df)

        # recommend
        model = self.recommend_posts()
        preds = model.predict(pred_df[self.features])
        topk_idx = np.argsort(preds)[::-1][:k]
        recommend_df = pred_df.loc[topk_idx].reset_index(drop=True)

        # check recommend
        print('---------- Recommend ----------')
        for i, row in recommend_df.iterrows():
            print(f'{i + 1}: {row["slug"]}:{row["title"]}')

        print('---------- Actual ----------')
        user_df = user_df.merge(posts_df, left_on='slug', how='inner')
        for i, row in user_df.sort_values('relevance', ascending=False).iterrows():
            print(f'relevance:{row["relevance"]}: {row["slug"]}:{row["title"]}')

        return recommend_df


    def recommend_posts(self):
        self.features = ['coefficient', 'rating_count', 'rating_mean']

        df_results = self.get_results_single_coeff_user_as_query()

        train, test = train_test_split(df_results, test_size=0.2, random_state=SEED)
        print('train shape: ', train.shape)
        print('tests shape: ', test.shape)
        user_col = 'user_id'
        item_col = 'slug'
        self.target_col = 'relevance'
        self.train = train.sort_values('user_id').reset_index(drop=True)
        self.test = test.sort_values('user_id').reset_index(drop=True)
        # model query data
        self.train_query = train[user_col].value_counts().sort_index()
        self.test_query = test[user_col].value_counts().sort_index()

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED)  # fix random_order seed
                                    )
        study.optimize(self.objective, n_trials=10)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        best_params = study.best_trial.params
        model = LGBMRanker(n_estimators=1000, **best_params, random_state=SEED, )
        model.fit(
            train[self.features],
            train[self.target_col],
            group=self.train_query,
            eval_set=[(test[self.features], test[self.target_col])],
            eval_group=[list(self.test_query)],
            eval_at=[1, 3, 5, 10, 20],
            early_stopping_rounds=50,
            verbose=10
        )

        TOP_N = 20
        model.predict(test.iloc[:TOP_N][self.features])

        # feature imporance
        plt.figure(figsize=(10, 7))
        df_plt = pd.DataFrame({'feature_name': self.features, 'feature_importance': model.feature_importances_})
        df_plt.sort_values('feature_importance', ascending=False, inplace=True)
        seaborn.barplot(x="feature_importance", y="feature_name", data=df_plt)
        plt.title('feature importance')

        return model

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

    def train_lightgbm_user_based(self):

        # TODO: Remove user id if it's needed
        df_results = self.get_results_single_coeff_searched_doc_as_query()
        recommenderMethods = RecommenderMethods()

        post_category_df = recommenderMethods.join_posts_ratings_categories()
        post_category_df = post_category_df.rename(columns={'slug': 'slug'})
        post_category_df = post_category_df.rename(columns={'category_title': 'category'})

        print(df_results.columns)
        print(post_category_df.columns)

        categorical_columns = [
            "category", "model_name"
        ]

        numerical_columns = [
            "user_id", "coefficient", "relevance_val", "views"
        ]

        df_results_merged = df_results.merge(post_category_df, on='slug')
        print("df_results_merged.columns")
        print(df_results_merged.columns)
        time.sleep(60)

        print("Loading Doc2Vec model...")
        doc2vec = Doc2VecClass()
        doc2vec.load_model()
        df_results_merged = df_results_merged.rename({"doc2vec_representation": "doc2vec"}, axis=1)
        print("df_results_merged:")
        print(df_results_merged.to_string())
        df2 = pd.DataFrame(df_results_merged)
        print("df2:")
        print(df2.to_string())
        print("Searching for Doc2Vec missing values...")
        df2['doc2vec'] = df2.apply(lambda row: json.dumps(doc2vec.get_vector_representation(row['slug']).tolist()) if pd.isnull(row['doc2vec']) else row['doc2vec'], axis=1)
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
        # df_results_merged = df_results_merged.columns.drop("doc2vec")

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

        pickle.dump(model, open('../../../../models/lightgbm.pkl', 'wb'))


    def get_posts_lightgbm(self, slug, use_categorical_columns=True):
        global one_hot_encoder, categorical_columns_after_encoding
        consider_only_top_limit = 20
        if use_categorical_columns is True:
            one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int32)

        features = ["user_id", "coefficient", "relevance", "relevance_val", "views", "model_name"]
        categorical_columns = [
            'category', 'model_name'
        ]

        # Loading TfIdf results
        tfidf = TfIdf()
        tf_idf_results = tfidf.recommend_posts_by_all_features_preprocessed(slug)
        print("tf_idf_results")
        print(tf_idf_results)
        print("type(tf_idf_results)")
        print(type(tf_idf_results))
        json_data = json.loads(json.dumps(tf_idf_results))
        print("json_data")
        print(json_data)
        print("type(json_data)")
        print(type(json_data))
        tf_idf_results = pd.json_normalize(json_data)
        print(tf_idf_results)

        recommenderMethods = RecommenderMethods()
        post_category_df = recommenderMethods.join_posts_ratings_categories()

        post_category_df = post_category_df.rename(columns={'slug': 'slug'})
        post_category_df = post_category_df.rename(columns={'category_title': 'category'})
        post_category_df['model_name'] = 'tfidf'

        tf_idf_results = tf_idf_results.merge(post_category_df, on='slug')

        tf_idf_results = tf_idf_results.rename({"doc2vec_representation": "doc2vec"}, axis=1)
        df2 = pd.DataFrame(tf_idf_results)
        doc2vec_column_name_base = "doc2vec_col_"

        print("df_results_merged.to_list()")
        print(tf_idf_results.to_string())
        print("df2:")
        print(df2.to_string())
        print("Sizes:")
        print(df2.doc2vec.tolist())
        print("df2")
        print(df2)
        df2.dropna(subset=['doc2vec'], inplace=True)
        print("df2 after dropna")
        print(df2)
        df2['doc2vec'] = df2['doc2vec'].apply(lambda x: json.loads(x))
        df2 = pd.DataFrame(df2['doc2vec'].to_list(), index=df2.index).add_prefix(doc2vec_column_name_base)

        print("df2 after convert to list")
        print(df2.to_string())
        tf_idf_results = pd.concat([tf_idf_results, df2], axis=1)

        #####
        # TODO: Find and fill missing Doc2Vec values (like in the training phase)
        print("tf_idf_results")
        print(tf_idf_results.to_string())

        tf_idf_results_old = tf_idf_results
        if use_categorical_columns is True:
            numerical_columns = [
                "coefficient", "views", 'doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3', 'doc2vec_col_4', 'doc2vec_col_5',
                 'doc2vec_col_6', 'doc2vec_col_7'
            ]
            one_hot_encoder.fit(post_category_df[categorical_columns])
            tf_idf_results = self.preprocess_one_hot(tf_idf_results, one_hot_encoder, numerical_columns,
                                                     categorical_columns)
            tf_idf_results['slug'] = tf_idf_results_old['slug']

        features_X = ['coefficient', 'views']

        all_columns = ['user_id', 'query_id', 'slug', 'query_slug', 'coefficient', 'relevance', 'id_x', 'post_title', 'excerpt', 'body', 'views', 'keywords', 'category', 'description', 'all_features_preprocessed', 'body_preprocessed']
        if use_categorical_columns is True:
            categorical_columns_after_encoding = [x for x in all_columns if x.startswith("category_")]
            features.extend(categorical_columns_after_encoding)
            print('number of one hot encoded categorical columns: ',
                  len(one_hot_encoder.get_feature_names(categorical_columns)))
        if use_categorical_columns is True:
            features_X.extend(categorical_columns_after_encoding)
            features_X.extend(
                ['doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3', 'doc2vec_col_4', 'doc2vec_col_5',
                 'doc2vec_col_6', 'doc2vec_col_7'])

        pred_df = self.make_post_feature(tf_idf_results)
        lightgbm_model_file = Path("../../../../models/lightgbm.pkl")
        if lightgbm_model_file.exists():
            model = pickle.load(open('../../../../models/lightgbm.pkl', 'rb'))
        else:
            print("LightGBM model not found. Training from available relevance testing results datasets...")
            self.train_lightgbm_user_based()
            model = pickle.load(open('../../../../models/lightgbm.pkl', 'rb'))
        predictions = model.predict(tf_idf_results[features_X])  # .values.reshape(-1,1) when single feature is used
        print("predictions:")
        print(predictions)
        topk_idx = np.argsort(predictions)[::-1][:consider_only_top_limit]
        recommend_df = pred_df.loc[topk_idx].reset_index(drop=True)
        recommend_df['predictions'] = predictions
        # df_unseen['predictions'] = predictions
        # print("df_unseen:")
        # print(df_unseen.to_string())
        # recommend_df = recommend_df.loc[recommend_df['user_id'].isin([user_id])]
        # recommend_df = df_unseen.loc[df_unseen['query_slug'].isin([slug])]
        recommend_df.sort_values(by=['predictions'], inplace=True, ascending=False)
        print('---------- Recommend ----------')
        print(recommend_df.to_string())

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

        evaluation_results_df = evaluation_results.get_results_dataframe()
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
        tfidf_posts_full = self.get_tfidf(self.tfidf, post_slug)
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        tfidf_keywords = self.get_user_keywords_based(self.tfidf, self.user_based_recommendation, self.user_id)
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        doc2vec_posts = self.get_doc2vec(self.doc2vec, post_slug)
        print("--- %s seconds ---" % (time.time() - start_time))
        """

    def get_posts_df(self):
        database = Database()
        posts_df = database.get_posts_dataframe()
        return posts_df

    def get_categories_df(self):
        database = Database()
        posts_df = database.get_categories_dataframe()
        return posts_df


class LearnToRank:

    def get_user_keywords_based(self, tfidf, user_based_recommendation, user_id):
        user_keywords = user_based_recommendation.get_user_keywords(user_id)
        keyword_list = user_keywords['keyword_name'].tolist()
        tfidf_keywords = ''
        if len(keyword_list) > 0:
            keywords = ' '.join(keyword_list)
            print(keywords)
            tfidf_keywords = tfidf.keyword_based_comparison(keywords, all_posts=True)

        return tfidf_keywords

    def get_tfidf(self, tfidf, post_slug):
        tfidf.prepare_dataframes()
        tfidf_prefilled_posts = tfidf.get_prefilled_full_text()
        print("tfidf_prefilled_posts:")
        print(tfidf_prefilled_posts)
        found_row = tfidf_prefilled_posts.loc[tfidf_prefilled_posts['slug'] == post_slug]
        tfidf_results_json = json.loads(found_row['recommended_tfidf_full_text'].iloc[0])
        tfidf_results_df = pd.json_normalize(tfidf_results_json)
        print("tfidf_results_df:")
        print(tfidf_results_df)
        return tfidf_results_df

    def get_doc2vec(self, doc2vec, post_slug):
        doc2vec_posts = doc2vec.get_prefilled_full_text(post_slug)
        doc2vec_posts_full = doc2vec.get_similar_doc2vec(post_slug, number_of_recommended_posts=NUM_OF_POSTS)
        return doc2vec_posts_full

    def linear_regression(self, user_id, post_slug):

        global tfidf_keywords_full, tfidf_keywords_full_df, tfidf_keywords_df
        tfidf = TfIdf()
        doc2vec = Doc2VecClass()
        lda = Lda()
        user_based_recommendation = UserBasedRecommendation()
        svd = SvdClass()

        feature_list = []

        NUM_OF_POSTS = len(tfidf.database.get_posts_dataframe().index)

        tfidf_posts = tfidf.recommend_posts_by_all_features_preprocessed(post_slug)
        print("tfidf_posts")
        print(tfidf_posts)
        tfidf_all_posts = tfidf.recommend_posts_by_all_features_preprocessed(post_slug,
                                                                              num_of_recommendations=NUM_OF_POSTS)
        print("tfidf_all_posts")
        print(tfidf_all_posts)

        user_keywords = user_based_recommendation.get_user_keywords(user_id)
        keyword_list = user_keywords['keyword_name'].tolist()
        tfidf_keywords = ''
        if len(keyword_list) > 0:
            keywords = ' '.join(keyword_list)
            print(keywords)
            tfidf_keywords = tfidf.keyword_based_comparison(keywords, number_of_recommended_posts=10)
            tfidf_keywords_full = tfidf.keyword_based_comparison(keywords, number_of_recommended_posts=NUM_OF_POSTS)

        doc2vec_posts = doc2vec.get_similar_doc2vec(post_slug)
        doc2vec_all_posts = doc2vec.get_similar_doc2vec(post_slug, number_of_recommended_posts=NUM_OF_POSTS)

        lda_posts = lda.get_similar_lda(post_slug)
        lda_all_posts = lda.get_similar_lda(post_slug, N=NUM_OF_POSTS)

        user_preferences_posts = user_based_recommendation.load_recommended_posts_for_user(user_id,
                                                                                           num_of_recommendations=20)
        print("user_preferences_posts")
        print(user_preferences_posts)

        user_preferences_posts, user_preferences_posts_full = svd.get_average_post_rating()

        user_collaboration_posts = svd.run_svd(user_id, num_of_recommendations=20)
        df = pd.DataFrame.from_dict(user_collaboration_posts, orient='index').transpose()
        user_collaboration_posts_df = pd.DataFrame(df['data'].tolist(),
                                                   columns=['post_id', 'slug', 'rating_predicted'])
        user_collaboration_posts_df = user_collaboration_posts_df[['slug', 'rating_predicted']]
        print("user_collaboration_posts_df")
        print(user_collaboration_posts_df)
        user_collaboration_posts_dict = user_collaboration_posts_df.to_dict('records')

        user_collaboration_posts_full = svd.run_svd(user_id, num_of_recommendations=NUM_OF_POSTS)
        df = pd.DataFrame.from_dict(user_collaboration_posts_full, orient='index').transpose()
        user_collaboration_posts_full_df = pd.DataFrame(df['data'].tolist(),
                                                        columns=['post_id', 'slug', 'rating_predicted'])
        user_collaboration_posts_full_df = user_collaboration_posts_full_df[['slug', 'rating_predicted']]
        print("user_collaboration_posts_full_df")
        print(user_collaboration_posts_full_df)
        user_collaboration_posts_full_dict = user_collaboration_posts_full_df.to_dict('records')

        if len(keyword_list) > 0:
            feature_list.append([tfidf_posts, tfidf_keywords, doc2vec_posts, lda_posts, user_preferences_posts,
                                 user_collaboration_posts])
        else:
            feature_list.append([tfidf_posts, doc2vec_posts, lda_posts, user_preferences_posts,
                                 user_collaboration_posts])
        print("tfidf_posts")
        print(tfidf_posts)
        print("tfidf_keywords")
        print(tfidf_keywords)
        print("doc2vec_posts")
        print(doc2vec_posts)
        print("lda_posts")
        print(lda_posts)
        print("user_preferences_posts")
        print(user_preferences_posts)
        print("user_collaboration_posts")
        print(user_collaboration_posts_dict)

        print("feature_list")
        print(feature_list)

        tfidf_posts_df = pd.DataFrame(tfidf_posts)
        tfidf_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'score_tfidf_posts'}, inplace=True)
        print("tfidf_posts_df:")
        print(tfidf_posts_df)
        tfidf_posts_df = tfidf_posts_df.set_index('slug')
        print("tfidf_posts_df:")
        print(tfidf_posts_df)

        tfidf_all_posts_df = pd.DataFrame(tfidf_all_posts)
        tfidf_all_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'score_tfidf_posts'}, inplace=True)
        print("tfidf_all_posts_df:")
        print(tfidf_all_posts_df)
        tfidf_all_posts_df = tfidf_all_posts_df.set_index('slug')
        print("tfidf_all_posts_df:")
        print(tfidf_all_posts_df)

        if len(keyword_list) > 0:
            tfidf_keywords_df = pd.DataFrame(tfidf_keywords)
            tfidf_keywords_df.rename(columns={'slug': 'slug', 'coefficient': 'score_tfidf_keywords'}, inplace=True)
            print("tfidf_keywords_df:")
            print(tfidf_keywords_df)

            tfidf_keywords_full_df = pd.DataFrame(tfidf_keywords_full)
            tfidf_keywords_full_df.rename(columns={'slug': 'slug', 'coefficient': 'score_tfidf_keywords'}, inplace=True)
            print("tfidf_keywords_full_df:")
            print(tfidf_keywords_full_df)

        doc2vec_posts_df = pd.DataFrame(doc2vec_posts)
        doc2vec_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'score_doc2vec_posts'}, inplace=True)
        print("doc2vec_posts_df:")
        print(doc2vec_posts_df)

        doc2vec_all_posts_df = pd.DataFrame(doc2vec_all_posts)
        doc2vec_all_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'score_doc2vec_posts'},
                                     inplace=True)
        print("doc2vec_posts_df_full:")
        print(doc2vec_all_posts_df)

        lda_posts_df = pd.DataFrame(lda_posts)
        lda_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'score_lda_posts'}, inplace=True)

        lda_all_posts_df = pd.DataFrame(lda_all_posts)
        lda_all_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'score_lda_posts'}, inplace=True)
        print("lda_all_posts_df:")
        print(lda_all_posts_df)

        user_preferences_posts_df = pd.DataFrame(user_preferences_posts)
        user_preferences_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'rating_actual'},
                                         inplace=True)
        user_preferences_posts_full_df = pd.DataFrame(user_preferences_posts_full)
        user_preferences_posts_full_df.rename(columns={'slug': 'slug', 'coefficient': 'rating_actual'},
                                              inplace=True)

        user_preferences_posts_full_df = user_preferences_posts_full_df[['slug', 'rating_actual']]

        user_collaboration_posts_df = pd.DataFrame(user_collaboration_posts_dict)
        user_collaboration_posts_df.rename(
            columns={'slug': 'slug', 'coefficient': 'score_user_collaboration_posts_dict'}, inplace=True)
        print("user_collaboration_posts_dict_df:")
        print(user_collaboration_posts_df)

        user_collaboration_posts_full_df = pd.DataFrame(user_collaboration_posts_full_dict)
        user_collaboration_posts_df.rename(
            columns={'slug': 'slug', 'coefficient': 'score_user_collaboration_posts_dict'}, inplace=True)
        print("user_collaboration_posts_full_df:")
        print(user_collaboration_posts_full_df)

        # Convert to Dictionary to show also names of the dataframes?
        if len(user_keywords) > 0:
            recommender_dataframes = [tfidf_posts_df, tfidf_keywords_df, doc2vec_posts_df, lda_posts_df,
                                      user_preferences_posts_df, user_collaboration_posts_df]
        else:
            recommender_dataframes = [tfidf_posts_df, doc2vec_posts_df, lda_posts_df,
                                      user_preferences_posts_df, user_collaboration_posts_df]

        # Find intersections of elements, then fill with the rest of recommendations that are not yet in the list by intersections
        i, j = 0, 0
        intersection_list = []
        for dataframe_i, dataframe_j in itertools.combinations(recommender_dataframes, 2):
            print("dataframe_i")
            print(dataframe_i)
            print("dataframe_j")
            print(dataframe_j)
            dictionary_intersection = self.intersect(dataframe_i, dataframe_j)
            print("dictionary_intersection")
            print(dictionary_intersection.to_string())
            if not dictionary_intersection.empty:
                intersection_list.append(dictionary_intersection)

        intersections_df_merged = [df.set_index('slug') for df in intersection_list]
        intersections_df_merged = pd.concat(intersections_df_merged).drop_duplicates()

        if 'score_tfidf_posts' not in intersections_df_merged.columns:
            intersections_df_merged["score_tfidf_posts"] = np.nan
        if 'score_tfidf_keywords' not in intersections_df_merged.columns:
            intersections_df_merged["score_tfidf_keywords"] = np.nan
        if 'score_lda_posts' not in intersections_df_merged.columns:
            intersections_df_merged["score_lda_posts"] = np.nan
        if 'score_doc2vec_posts' not in intersections_df_merged.columns:
            intersections_df_merged["score_doc2vec_posts"] = np.nan
        if 'rating_actual' not in intersections_df_merged.columns:
            intersections_df_merged["rating_actual"] = np.nan
        if 'rating_predicted' not in intersections_df_merged.columns:
            intersections_df_merged["rating_predicted"] = np.nan

        intersections_df_merged = intersections_df_merged[
            ['rating_actual', 'rating_predicted', 'score_tfidf_posts', 'score_tfidf_keywords', 'score_doc2vec_posts',
             'score_lda_posts']]

        print("Found intersections:")
        print(intersections_df_merged.to_string())
        intersections_df_merged['score_tfidf_posts'] = intersections_df_merged['score_tfidf_posts'].fillna(
            tfidf_posts_df['score_tfidf_posts'])
        """
          + intersections_df_merged['rating_predicted'].fillna(tfidf_posts_df['rating_predicted']) \
          + intersections_df_merged['rating_actual'].fillna(tfidf_posts_df['rating_actual']) \
          + intersections_df_merged['score_lda_posts'].fillna(tfidf_posts_df['score_lda_posts']) \
          + intersections_df_merged['score_doc2vec_posts'].fillna(tfidf_posts_df['score_doc2vec_posts'])
        """
        if len(keyword_list) > 0:
            intersections_df_merged['score_tfidf_keywords'] = intersections_df_merged['score_tfidf_keywords'].fillna(
                tfidf_keywords_df['score_tfidf_keywords'])
            tfidf_keywords_full_df = tfidf_keywords_full_df.set_index('slug')
            print("tfidf_keywords_full_df")
            print(tfidf_keywords_full_df)
        lda_all_posts_df = lda_all_posts_df.set_index('slug')
        doc2vec_all_posts_df = doc2vec_all_posts_df.set_index('slug')
        user_collaboration_posts_full_df = user_collaboration_posts_full_df.set_index('slug')
        print("user_preferences_posts_full_df")
        print(user_preferences_posts_full_df)
        user_preferences_posts_full_df = user_preferences_posts_full_df.set_index('slug')
        print("tfidf_all_posts_df")
        print(tfidf_all_posts_df.head(20))
        print("lda_all_posts_df")
        print(lda_all_posts_df.head(20))
        print("doc2vec_all_posts_df")
        print(doc2vec_all_posts_df.head(20))
        print("user_collaboration_posts_full_df")
        print(user_collaboration_posts_full_df)
        print("user_preferences_posts_full_df")
        print(user_preferences_posts_full_df)

        if len(keyword_list) > 0:
            df_merged = pd.concat(
                [tfidf_all_posts_df, tfidf_keywords_df, lda_all_posts_df, doc2vec_all_posts_df,
                 user_collaboration_posts_full_df, user_preferences_posts_full_df], axis=1)
        else:
            df_merged = pd.concat(
                [tfidf_all_posts_df, lda_all_posts_df, doc2vec_all_posts_df, user_collaboration_posts_full_df,
                 user_preferences_posts_full_df], axis=1)

        print("Found intersections:")
        intersections_df_merged = intersections_df_merged.reset_index()
        tfidf_all_posts_df = tfidf_all_posts_df.reset_index()
        if len(keyword_list) > 0:
            tfidf_keywords_full_df = tfidf_keywords_df.reset_index()
            print("TfIdf full df:")
            print(tfidf_keywords_full_df)
        lda_all_posts_df = lda_all_posts_df.reset_index()
        doc2vec_all_posts_df = doc2vec_all_posts_df.reset_index()
        user_collaboration_posts_full_df = user_collaboration_posts_full_df.reset_index()
        user_preferences_posts_full_df = user_preferences_posts_full_df.reset_index()

        print("user_collaboration_posts_full_df")
        print(user_collaboration_posts_full_df.head(200).to_string())

        print(intersections_df_merged.to_string())
        print("Dataframe columns")
        print(intersections_df_merged.columns.tolist())
        intersections_df_merged['score_tfidf_posts'] = intersections_df_merged['score_tfidf_posts'].combine_first(
            intersections_df_merged['slug'].map(tfidf_all_posts_df.set_index('slug')['score_tfidf_posts']))
        if len(keyword_list) > 0:
            intersections_df_merged['score_tfidf_keywords'] = intersections_df_merged[
                'score_tfidf_keywords'].combine_first(
                intersections_df_merged['slug'].map(tfidf_keywords_df.set_index('slug')['score_tfidf_keywords']))
        intersections_df_merged['rating_predicted'] = intersections_df_merged['rating_predicted'].combine_first(
            intersections_df_merged['slug'].map(user_collaboration_posts_full_df.set_index('slug')['rating_predicted']))
        intersections_df_merged['rating_actual'] = intersections_df_merged['rating_actual'].combine_first(
            intersections_df_merged['slug'].map(user_preferences_posts_full_df.set_index('slug')['rating_actual']))

        print("lda_all_posts_df:")
        print(lda_all_posts_df)

        print("doc2vec_all_posts_df:")
        print(doc2vec_all_posts_df)

        intersections_df_merged['score_doc2vec_posts'] = intersections_df_merged['score_doc2vec_posts'].combine_first(
            intersections_df_merged['slug'].map(doc2vec_all_posts_df.set_index('slug')['score_doc2vec_posts']))

        intersections_df_merged['score_lda_posts'] = intersections_df_merged['score_lda_posts'].combine_first(
            intersections_df_merged['slug'].map(lda_all_posts_df.set_index('slug')['score_lda_posts']))


        print("intersections_df_merged")
        print(intersections_df_merged.to_string())

        print("Full merged DataFrame:")
        print(df_merged.head(100).to_string())
        # df_merged.to_csv("exports/df_recommender_features_merged.csv")

        df_merged = df_merged.dropna()
        df_merged = df_merged.loc[~(df_merged['rating_actual'] == 0)]

        print("Merged dataframe without missing values:")
        print(df_merged.to_string())
        # predictions(tfidf,doc2vec,lda,wor2vec,user_r

        ratings = df_merged[['rating_actual']]
        signals = df_merged.loc[:, 'score_tfidf_posts':'rating_predicted']

        # df_merged.to_csv("exports/all_posts_merged.csv")

        # rating_predicted = c0 + c1 * tfidf + c2 * doc2vec + c3 * lda + c5 * rating_average + c6 * thumbs
        if len(keyword_list) > 0:
            y = df_merged[['rating_predicted']]
            df_merged = df_merged.rename(columns={'rating_actual': 'score_rating_average'})
            X = df_merged.loc[:, ['score_tfidf_posts', 'score_tfidf_keywords', 'score_doc2vec_posts', 'score_lda_posts',
                                  'score_rating_average']]
            print("X:")
            print(X)
            X = (X - X.mean()) / X.std()
            print("X normalised:")
            print(X)
            features_dict = {0: 'TfIdf Posts', 1: 'TfIdf Keywords', 2: 'Doc2vec', 3: 'LDA', 4: 'Rating avg'}
        else:
            y = df_merged[['rating_predicted']]
            df_merged = df_merged.rename(columns={'rating_actual': 'score_rating_average'})
            X = df_merged.loc[:,
                ['score_tfidf_posts', 'score_doc2vec_posts', 'score_lda_posts', 'score_rating_average']]
            print("X:")
            print(X)
            X = (X - X.mean()) / X.std()
            print("X normalised:")
            print(X)
            features_dict = {0: 'TfIdf Posts', 1: 'Doc2vec', 2: 'LDA', 3: 'Rating avg'}

        # define the model
        model = XGBRegressor()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))

        user_has_keywords = None
        if len(user_keywords) > 0:
            user_has_keywords = True
        else:
            user_has_keywords = False

        return self.prepare_results(df_merged, importance, user_has_keywords)

    def prepare_results(self, df_merged, importance, user_has_keywords):

        final_combined_results_list = []
        print("Relevance scores:")

        if user_has_keywords is True:
            if user_has_keywords is True:
                for slug_index, row in df_merged.iterrows():
                    relevance_score = self.relevance_score_lin_combination(tfidf_coeff=importance[0],
                                                         tfidf_keywords_coeff=importance[1],
                                                         doc2vec_coeff=importance[2],
                                                         lda_coeff=importance[3],
                                                         rating_average_coeff=importance[4],
                                                         tfidf_score=row['score_tfidf_posts'],
                                                         tfidf_keywords_score=row['score_tfidf_keywords'],
                                                         doc2vec_score=row['score_doc2vec_posts'],
                                                         lda_score=row['score_lda_posts'],
                                                         rating_average_score=row['score_rating_average'])
                    final_combined_results_list.append({'slug': slug_index, 'coefficient': relevance_score})
        else:
            if user_has_keywords is True:
                for slug_index, row in df_merged.iterrows():
                    relevance_score = self.relevance_score_lin_combination(tfidf_coeff=importance[0],
                                                         doc2vec_coeff=importance[1],
                                                         lda_coeff=importance[2],
                                                         rating_average_coeff=importance[3],
                                                         tfidf_score=row['score_tfidf_posts'],
                                                         doc2vec_score=row['score_doc2vec_posts'],
                                                         lda_score=row['score_lda_posts'],
                                                         rating_average_score=row['score_rating_average'])
                    final_combined_results_list.append({'slug': slug_index, 'coefficient': relevance_score})
        # sorting results by coefficient
        final_combined_results_list = sorted(final_combined_results_list, key=lambda d: d['coefficient'], reverse=True)

        print(final_combined_results_list[0:20])
        return final_combined_results_list[0:20]

    def intersect(self, a, b):
        return pd.merge(a, b, how='inner', on=['slug'])

    def relevance_score_lin_combination(self, tfidf_coeff, doc2vec_coeff, lda_coeff, rating_average_coeff, tfidf_score, doc2vec_score, lda_score, rating_average_score, tfidf_keywords_coeff=None, tfidf_keywords_score=None, bias=0.05):
        if tfidf_keywords_coeff is None and tfidf_keywords_score is None:
            return (tfidf_coeff * tfidf_score) + (doc2vec_coeff * doc2vec_score) + (lda_coeff * lda_score) + (rating_average_coeff * rating_average_score) + bias
        else:
            return (tfidf_coeff * tfidf_score) + (tfidf_keywords_coeff * tfidf_keywords_score) + (doc2vec_coeff * doc2vec_score) + (lda_coeff * lda_score) + (rating_average_coeff * rating_average_score) + bias

    def relevance_score_logistic_regression(self, X, y):
        # define dataset
        # define the model
        model = LogisticRegression()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.coef_[0]

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def redis_test(self):
        r = redis.Redis(host='redis-10115.c3.eu-west-1-2.ec2.cloud.redislabs.com', port=10115, db=0, username="admin",
                        password=REDIS_PASSWORD)
        r.set('foo', 'bar')
        print(r.get('foo'))

def main():
    # client = Client()

    start_time = time.time()
    """
    user_id = 431
    post_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    learn_to_rank = LearnToRank()
    print(learn_to_rank.linear_regression(user_id, post_slug))
    """

    lighGBM = LightGBM()
    # lighGBM.train_lightgbm_document_based('tradicni-remeslo-a-rucni-prace-se-ceni-i-dnes-jejich-znacka-slavi-uspech')
    lighGBM.get_posts_lightgbm('zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy', True)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__": main()
