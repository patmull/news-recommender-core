import json
import os
import numpy as np
import redis
import pandas as pd

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.user_based_algorithms.collaboration_based_recommendation import \
    SvdClass
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_keywords_recommendation \
    import UserBasedMethods

REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

SEED = 2021


# TODO: Remove completely in the next code review
# noinspection DuplicatedCode
@DeprecationWarning
class LightGBM:
    tfidf = TfIdf()
    user_based_recommendation = UserBasedMethods()
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
        # noinspection DuplicatedCode
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

