import itertools
import json
import os

import numpy as np
import redis
import pandas as pd
# import modin.pandas as pd
# from modin.config import Engine

# from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor, XGBClassifier

from collaboration_based_recommendation import Svd
from content_based_algorithms.tfidf import TfIdf
from content_based_algorithms.doc2vec import Doc2VecClass
from content_based_algorithms.lda import Lda
from user_based_recommendation import UserBasedRecommendation
from sklearn.linear_model import LinearRegression, LogisticRegression

"""
from dask.distributed import Client

Engine.put("dask")  # Modin will use Ray
os.environ["MODIN_ENGINE"] = "dask"

pd.DEFAULT_NPARTITIONS = 10
"""

REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

class LearnToRank:

    def linear_regression(self, user_id, post_slug):

        tfidf = TfIdf()
        doc2vec = Doc2VecClass()
        lda = Lda()
        user_based_recommendation = UserBasedRecommendation()
        svd = Svd()

        feature_list = []

        NUM_OF_POSTS = 8194

        tfidf.prepare_dataframes()
        tfidf_prefilled_posts = tfidf.get_prefilled_full_text()
        print("tfidf_prefilled_posts:")
        print(tfidf_prefilled_posts)
        found_row = tfidf_prefilled_posts.loc[tfidf_prefilled_posts['slug_x'] == post_slug]
        tfidf_results_json = json.loads(found_row['recommended_tfidf_full_text'].iloc[0])
        tfidf_results_df = pd.json_normalize(tfidf_results_json)
        print("tfidf_results_df:")
        print(tfidf_results_df)

        user_keywords = user_based_recommendation.get_user_keywords(user_id)
        keyword_list = user_keywords['keyword_name'].tolist()
        tfidf_keywords = ''
        if len(keyword_list) > 0:
            keywords = ' '.join(keyword_list)
            print(keywords)
            tfidf_keywords = tfidf.keyword_based_comparison(keywords, number_of_recommended_posts=10)
            tfidf_keywords_full = tfidf.keyword_based_comparison(keywords, number_of_recommended_posts=NUM_OF_POSTS)

        doc2vec_posts = doc2vec.get_similar_doc2vec(post_slug)
        doc2vec_posts_full = doc2vec.get_similar_doc2vec(post_slug, number_of_recommended_posts=NUM_OF_POSTS)

        lda_posts = lda.get_similar_lda(post_slug)
        lda_posts_full = lda.get_similar_lda(post_slug, N=NUM_OF_POSTS)

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
            feature_list.append([tfidf_posts_full, tfidf_keywords, doc2vec_posts, lda_posts, user_preferences_posts,
                                 user_collaboration_posts])
        else:
            feature_list.append([tfidf_posts_full, doc2vec_posts, lda_posts, user_preferences_posts,
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

        tfidf_posts_full_df = pd.DataFrame(tfidf_posts_full)
        tfidf_posts_full_df.rename(columns={'slug': 'slug', 'coefficient': 'score_tfidf_posts'}, inplace=True)
        print("tfidf_posts_full_df:")
        print(tfidf_posts_full_df)
        tfidf_posts_full_df = tfidf_posts_full_df.set_index('slug')
        print("tfidf_posts_full_df:")
        print(tfidf_posts_full_df)

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

        doc2vec_posts_full_df = pd.DataFrame(doc2vec_posts_full)
        doc2vec_posts_full_df.rename(columns={'slug': 'slug', 'coefficient': 'score_doc2vec_posts'},
                                     inplace=True)
        print("doc2vec_posts_df_full:")
        print(doc2vec_posts_full_df)

        lda_posts_df = pd.DataFrame(lda_posts)
        lda_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'score_lda_posts'}, inplace=True)

        lda_posts_full_df = pd.DataFrame(lda_posts_full)
        lda_posts_full_df.rename(columns={'slug': 'slug', 'coefficient': 'score_lda_posts'}, inplace=True)

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

        intersections_df_merged = intersections_df_merged[['rating_actual','rating_predicted','score_tfidf_posts','score_tfidf_keywords','score_doc2vec_posts','score_lda_posts']]

        print("Found intersections:")
        print(intersections_df_merged.to_string())
        intersections_df_merged['score_tfidf_posts'] = intersections_df_merged['score_tfidf_posts'].fillna(tfidf_posts_df['score_tfidf_posts'])
        """
          + intersections_df_merged['rating_predicted'].fillna(tfidf_posts_df['rating_predicted']) \
          + intersections_df_merged['rating_actual'].fillna(tfidf_posts_df['rating_actual']) \
          + intersections_df_merged['score_lda_posts'].fillna(tfidf_posts_df['score_lda_posts']) \
          + intersections_df_merged['score_doc2vec_posts'].fillna(tfidf_posts_df['score_doc2vec_posts'])
        """
        if len(keyword_list) > 0:
            intersections_df_merged['score_tfidf_keywords'] = intersections_df_merged['score_tfidf_keywords'].fillna(tfidf_keywords_df['score_tfidf_keywords'])
            tfidf_keywords_full_df = tfidf_keywords_full_df.set_index('slug')
            print("tfidf_keywords_full_df")
            print(tfidf_keywords_full_df)
        lda_posts_full_df = lda_posts_full_df.set_index('slug')
        doc2vec_posts_full_df = doc2vec_posts_full_df.set_index('slug')
        user_collaboration_posts_full_df = user_collaboration_posts_full_df.set_index('slug')
        print("user_preferences_posts_full_df")
        print(user_preferences_posts_full_df)
        user_preferences_posts_full_df = user_preferences_posts_full_df.set_index('slug')
        print("tfidf_posts_full_df")
        print(tfidf_posts_full_df)
        print("lda_posts_full_df")
        print(lda_posts_full_df)
        print("doc2vec_posts_full_df")
        print(doc2vec_posts_full_df)
        print("user_collaboration_posts_full_df")
        print(user_collaboration_posts_full_df)
        print("user_preferences_posts_full_df")
        print(user_preferences_posts_full_df)

        if len(keyword_list) > 0:
            df_merged = pd.concat([tfidf_posts_full_df, tfidf_keywords_full_df, lda_posts_full_df, doc2vec_posts_full_df, user_collaboration_posts_full_df, user_preferences_posts_full_df], axis=1)
        else:
            df_merged = pd.concat([tfidf_posts_full_df, lda_posts_full_df, doc2vec_posts_full_df, user_collaboration_posts_full_df, user_preferences_posts_full_df], axis=1)

        print("Found intersections:")
        intersections_df_merged = intersections_df_merged.reset_index()
        tfidf_posts_full_df = tfidf_posts_full_df.reset_index()
        if len(keyword_list) > 0:
            tfidf_keywords_full_df = tfidf_keywords_full_df.reset_index()
            print("TfIdf full df:")
            print(tfidf_keywords_full_df)
        lda_posts_full_df = lda_posts_full_df.reset_index()
        doc2vec_posts_full_df = doc2vec_posts_full_df.reset_index()
        user_collaboration_posts_full_df = user_collaboration_posts_full_df.reset_index()
        user_preferences_posts_full_df = user_preferences_posts_full_df.reset_index()


        print("user_collaboration_posts_full_df")
        print(user_collaboration_posts_full_df.to_string())
        print("user_preferences_posts_full_df")
        print(user_preferences_posts_full_df)

        print(intersections_df_merged.to_string())
        intersections_df_merged['score_tfidf_posts'] = intersections_df_merged['score_tfidf_posts'].combine_first(intersections_df_merged['slug'].map(tfidf_posts_full_df.set_index('slug')['score_tfidf_posts']))
        if len(keyword_list) > 0:
            intersections_df_merged['score_tfidf_keywords'] = intersections_df_merged['score_tfidf_keywords'].combine_first(intersections_df_merged['slug'].map(tfidf_keywords_full_df.set_index('slug')['score_tfidf_keywords']))
        intersections_df_merged['rating_predicted'] = intersections_df_merged['rating_predicted'].combine_first(intersections_df_merged['slug'].map(user_collaboration_posts_full_df.set_index('slug')['rating_predicted']))
        intersections_df_merged['rating_actual'] = intersections_df_merged['rating_actual'].combine_first(intersections_df_merged['slug'].map(user_preferences_posts_full_df.set_index('slug')['rating_actual']))
        intersections_df_merged['score_lda_posts'] = intersections_df_merged['score_lda_posts'].combine_first(intersections_df_merged['slug'].map(lda_posts_full_df.set_index('slug')['score_lda_posts']))
        intersections_df_merged['score_doc2vec_posts'] = intersections_df_merged['score_doc2vec_posts'].combine_first(intersections_df_merged['slug'].map(doc2vec_posts_full_df.set_index('slug')['score_doc2vec_posts']))

        print("intersections_df_merged")
        print(intersections_df_merged.to_string())

        print("Full merged DataFrame:")
        print(df_merged.to_string())
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
            X = df_merged.loc[:, ['score_tfidf_posts', 'score_tfidf_keywords', 'score_doc2vec_posts', 'score_lda_posts', 'score_rating_average']]
            print("X:")
            print(X)
            X = (X - X.mean()) / X.std()
            print("X normalised:")
            print(X)
            features_dict = {0: 'TfIdf Posts', 1: 'TfIdf Keywords', 2: 'Doc2vec', 3: 'LDA', 4: 'Rating avg'}
        else:
            y = df_merged[['rating_predicted']]
            df_merged = df_merged.rename(columns={'rating_actual': 'score_rating_average'})
            X = df_merged.loc[:, ['score_tfidf_posts', 'score_doc2vec_posts', 'score_lda_posts', 'score_rating_average']]
            print("X:")
            print(X)
            X = (X - X.mean()) / X.std()
            print("X normalised:")
            print(X)
            features_dict = {0: 'TfIdf Posts', 1: 'Doc2vec', 2: 'LDA', 3: 'Rating avg'}

        """
        # Linear regression (Sklearn)
        model = LinearRegression()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = np.ndarray.flatten(model.coef_)
        # summarize feature importance
        print("importance:")
        print(importance)
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        self.plot_feature_importance(features_dict,importance,"Linear Regression")

        # define the model
        # define dataset
        # define the model
        model = DecisionTreeRegressor()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        self.plot_feature_importance(features_dict,importance,"Decision Tree Regressor")

        # define the model
        model = RandomForestRegressor()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        self.plot_feature_importance(features_dict,importance,"Random Forest Regressor")
        """

        # define the model
        model = XGBRegressor()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        # self.plot_feature_importance(features_dict,importance,"XGBRegressor")

        """
        # define the model
        model = XGBClassifier()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        self.plot_feature_importance(features_dict,importance,"XGBClassifier")

        # define the model
        model = KNeighborsRegressor()
        # fit the model
        model.fit(X, y)
        # perform permutation importance
        results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
        # get importance
        importance = results.importances_mean
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        self.plot_feature_importance(features_dict,importance,"K Neighbors Regressor")
        """

        # Lin regression
        """
        butIRegress = LinearRegression()
        butIRegress.fit(X, y)
        importance = np.ndarray.flatten(butIRegress.coef_)
        print("butIRegress.coef_")
        print(butIRegress.coef_)
        """

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

    """
    def plot_feature_importance(self, features_dict, importance, title):
        pyplot.bar([features_dict[x] for x in range(len(importance))], importance)
        pyplot.title(title)
        pyplot.show()
    """

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


def main():
    # client = Client()

    user_id = 431
    post_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    learn_to_rank = LearnToRank()
    print(learn_to_rank.linear_regression(user_id, post_slug))

    """
    r = redis.Redis(host='redis-10115.c3.eu-west-1-2.ec2.cloud.redislabs.com', port=10115, db=0, username="admin", password=REDIS_PASSWORD)
    r.set('foo', 'bar')
    print(r.get('foo'))
    """

if __name__ == "__main__": main()
