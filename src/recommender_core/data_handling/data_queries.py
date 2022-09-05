import json
import traceback
from pathlib import Path
from threading import Thread

import dropbox
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender_core.recommender_algorithms.content_based_algorithms.similarities import CosineTransformer
from src.recommender_core.data_handling.data_manipulation import Database
import os

CACHED_FILE_PATH = "db_cache/cached_posts_dataframe.pkl"


def convert_df_to_json(dataframe):
    result = dataframe[["title", "excerpt", "body"]].to_json(orient="records", lines=True)
    parsed = json.loads(result)
    return parsed


def convert_to_json_keyword_based(post_recommendations):
    list_of_article_slugs = []
    post_recommendation_dictionary = post_recommendations.to_dict('records')
    list_of_article_slugs.append(post_recommendation_dictionary.copy())
    # print("------------------------------------")
    # print("JSON:")
    # print("------------------------------------")
    # print(list_of_article_slugs[0])
    return list_of_article_slugs[0]


def convert_dataframe_posts_to_json(post_recommendations, slug, cosine_sim_df):
    list_of_article_slugs = []
    list_of_coefficients = []

    # finding coefficient belonging to recommended posts compared to original post (for which we want to find
    # recommendations)
    if 'post_slug' in post_recommendations:
        post_recommendations = post_recommendations.rename(columns={'post_slug': 'slug'})

    for index, row in post_recommendations.iterrows():
        list_of_coefficients.append(cosine_sim_df.at[row['slug'], slug])

    post_recommendations['coefficient'] = list_of_coefficients
    posts_recommendations_dictionary = post_recommendations.to_dict('records')
    list_of_article_slugs.append(posts_recommendations_dictionary.copy())
    return list_of_article_slugs[0]


def dropbox_file_download(access_token, dropbox_file_path, local_folder_name):
    try:
        dbx = dropbox.Dropbox(access_token)
        # Files can be displayed here. See this method in older commits.
        dbx.files_download_to_file(dropbox_file_path, local_folder_name)

    except Exception as e:
        print(e)
        return False


class RecommenderMethods:

    def __init__(self):
        self.database = Database()
        self.cached_file_path = CACHED_FILE_PATH
        self.posts_df = None
        self.categories_df = None
        self.df = None

    def get_posts_dataframe(self, force_update=False):
        print("4.1.1")
        if force_update is True:
            self.database.connect()
            self.posts_df = self.database.insert_posts_dataframe_to_cache(self.cached_file_path)
            self.database.disconnect()
        else:
            print("Trying reading from cache as default...")
            if os.path.isfile(self.cached_file_path):
                try:
                    print("Reading from cache...")
                    self.posts_df = self.database.get_posts_dataframe_from_cache()
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    self.database.connect()
                    self.posts_df = self.get_df_from_sql_meanwhile_insert_cache()
                    self.database.disconnect()
            else:
                self.database.connect()
                self.posts_df = self.get_df_from_sql_meanwhile_insert_cache()
                self.database.disconnect()

        print("4.1.2")
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        print("4.1.3")
        return self.posts_df

    def get_df_from_sql_meanwhile_insert_cache(self):
        def update_cache(self):
            print("Inserting file to cache in the background...")
            self.database.insert_posts_dataframe_to_cache()

        print("Posts not found on cache. Will use PgSQL command.")
        posts_df = self.database.get_posts_dataframe_from_sql()
        thread = Thread(target=update_cache, kwargs={'self': self})
        thread.start()
        return posts_df

    def get_users_dataframe(self):
        self.database.connect()
        self.posts_df = self.database.get_users_dataframe()
        self.database.disconnect()
        return self.posts_df

    def get_ratings_dataframe(self):
        self.database.connect()
        self.posts_df = self.database.get_ratings_dataframe()
        self.database.disconnect()
        return self.posts_df

    def get_user_keywords(self, user_id):
        self.database.connect()
        df_user_keywords = self.database.get_user_keywords(user_id=user_id)
        self.database.disconnect()
        return df_user_keywords

    def get_user_rating_categories(self):
        self.database.connect()
        user_rating_categories_df = self.database.get_user_rating_categories()
        self.database.disconnect()
        return user_rating_categories_df

    def get_user_categories(self, user_id):
        self.database.connect()
        df_user_categories = self.database.get_user_categories(user_id)
        self.database.disconnect()
        return df_user_categories

    def get_categories_dataframe(self):
        # rename_title (defaul=False): for ensuring that category title does not collide with post title
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe()
        self.database.disconnect()
        if 'slug_y' in self.categories_df.columns:
            self.categories_df = self.categories_df.rename(columns={'slug_y': 'category_slug'})
        elif 'slug' in self.categories_df.columns:
            self.categories_df = self.categories_df.rename(columns={'slug': 'category_slug'})
        return self.categories_df

    @DeprecationWarning
    def get_user_posts_ratings(self):
        database = Database()

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id AS user_id, u.name, r.value 
        AS rating_value
        FROM posts p
        JOIN ratings r ON r.post_id = p.id
        JOIN users u ON r.user_id = u.id;"""
        # LOAD INTO A DATAFRAME
        df_ratings = pd.read_sql_query(sql_rating, database.get_cnx())
        return df_ratings

    def get_ranking_evaluation_results_dataframe(self):
        self.database.connect()
        results_df = self.database.get_relevance_testing_dataframe()
        self.database.disconnect()
        results_df.reset_index(inplace=True)
        print("self.results_df:")
        print(results_df)
        results_df_ = results_df[
            ['id', 'query_slug', 'results_part_1', 'results_part_2', 'user_id', 'model_name', 'model_variant',
             'created_at']]
        return results_df_

    def get_item_evaluation_results_dataframe(self):
        self.database.connect()
        results_df = self.database.get_thumbs_dataframe()
        self.database.disconnect()
        results_df.reset_index(inplace=True)
        print("self.results_df:")
        print(results_df)
        results_df_ = results_df[
            ['id', 'value', 'user_id', 'post_id',
             'created_at']]
        return results_df_


    @DeprecationWarning
    def join_posts_ratings_categories(self, full_text=True, include_prefilled=False):

        self.posts_df = self.get_posts_dataframe()
        print("4.1")
        self.get_categories_dataframe()
        print("4.2")
        print(self.posts_df.columns)
        self.posts_df = self.posts_df.rename(columns={'title': 'post_title', 'slug': 'post_slug'})
        self.categories_df = self.categories_df.rename(columns={'title': 'category_title'})

        if include_prefilled is False:
            self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
            # clean up from unnecessary columns
        else:
            self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
            # clean up from unnecessary columns
            try:
                self.df = self.df[
                    ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title',
                     'description',
                     'all_features_preprocessed', 'body_preprocessed',
                     'recommended_tfidf_full_text', 'trigrams_full_text']]
            except KeyError:
                self.df = self.database.insert_posts_dataframe_to_cache()
                self.posts_df.drop_duplicates(subset=['title'], inplace=True)
                print(self.df.columns.values)
                self.df = self.df[
                    ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title',
                     'description', 'all_features_preprocessed', 'body_preprocessed',
                     'recommended_tfidf_full_text', 'trigrams_full_text']]
        print("4.3")
        print(self.df.columns)

        if full_text is True:
            try:
                self.df = self.df[
                    ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title',
                     'description',
                     'all_features_preprocessed', 'body_preprocessed', 'doc2vec_representation', 'full_text',
                     'trigrams_full_text']]
            except KeyError as key_error:
                self.df = self.get_df_from_sql_meanwhile_insert_cache()
                print(key_error)
                print("Columns of self.df:")
                print(self.df.columns)
                self.df = self.df.rename(columns={'slug_x': 'post_slug'})
                self.df = self.df[
                    ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title',
                     'description',
                     'all_features_preprocessed', 'body_preprocessed', 'doc2vec_representation', 'full_text',
                     'trigrams_full_text']]
        else:
            self.df = self.df[
                ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title',
                 'description',
                 'all_features_preprocessed', 'body_preprocessed', 'doc2vec_representation', 'trigrams_full_text']]
        return self.df

    def find_post_by_slug(self, searched_slug):
        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a string.")
        else:
            if searched_slug == "":
                raise ValueError("Entered string is empty.")
            else:
                pass

        return self.get_posts_dataframe().loc[self.get_posts_dataframe()['slug'] == searched_slug]

    def get_posts_categories_dataframe(self, force_update=False):
        posts_df = self.get_posts_dataframe(force_update=True)
        categories_df = self.get_categories_dataframe()

        posts_df = posts_df.rename(columns={'title': 'post_title'})
        categories_df = categories_df.rename(columns={'title': 'category_title'})
        if 'slug_y' in categories_df.columns:
            categories_df = categories_df.rename(columns={'slug_y': 'category_slug'})
        elif 'slug' in categories_df.columns:
            categories_df = categories_df.rename(columns={'slug': 'category_slug'})
        print("posts_df")
        print(posts_df)
        print(posts_df.columns)
        print("categories_df")
        print(categories_df)
        print(categories_df.columns)
        self.df = pd.merge(posts_df, categories_df, left_on='category_id', right_on='id')
        return self.df

    def get_posts_categories_full_text(self):
        posts_df = self.get_posts_dataframe()
        posts_df = posts_df.rename(columns={'title': 'post_title'})
        print("self.posts_df.columns")
        print(posts_df.columns)
        posts_df = posts_df.rename(columns={'slug': 'post_slug'})
        print("self.posts_df.columns")
        print(posts_df.columns)
        categories_df = self.get_categories_dataframe()
        categories_df = categories_df.rename(columns={'title': 'category_title'})

        self.df = self.posts_df.merge(categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        print("df.columns")
        print(self.df.columns)
        if 'post_title' in self.df.columns:
            self.df = self.df.rename({'title': 'post_title', 'slug': 'post_slug'})
        self.df = self.df[
            ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
             'all_features_preprocessed', 'body_preprocessed', 'full_text', 'category_id']]
        return self.df

    def get_all_posts(self):
        self.database.connect()
        all_posts_df = self.database.get_all_posts()
        self.database.disconnect()
        return all_posts_df

    def get_posts_users_categories_ratings_df(self):
        self.database.connect()
        posts_users_categories_ratings_df = self.database.get_posts_users_categories_ratings()
        self.database.disconnect()
        return posts_users_categories_ratings_df


def get_cleaned_text(row):
    return row


class TfIdfDataHandlers:

    def __init__(self, df=None):
        self.df = df
        self.tfidf_vectorizer = None
        self.cosine_sim_df = None
        self.tfidf_tuples = None

    def set_df(self, df):
        self.df = df

    def get_fit_by_feature_(self, feature_name, second_feature=None):
        fit_by_feature = self.get_tfidf_vectorizer(feature_name, second_feature)
        return fit_by_feature

    def most_similar_by_keywords(self, keywords, tupple_of_fitted_matrices, number_of_recommended_posts=20):
        # combining results of all feature types to sparse matrix
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices, dtype=np.float16)

        # computing cosine similarity using matrix with combined features
        print("Computing cosine similarity using matrix with combined features...")
        cosine_transform = CosineTransformer()
        self.cosine_sim_df = cosine_transform.get_cosine_sim_use_own_matrix(combined_matrix1, self.df)
        combined_all = self.get_recommended_posts_for_keywords(keywords=keywords,
                                                               data_frame=self.df,
                                                               k=number_of_recommended_posts)

        df_renamed = combined_all.rename(columns={'post_slug': 'slug'})
        recommended_posts_in_json = convert_to_json_keyword_based(df_renamed)

        return recommended_posts_in_json

    def get_recommended_posts_for_keywords(self, keywords, data_frame, k=10):

        keywords_list = [keywords]
        txt_cleaned = get_cleaned_text(self.df['post_title'] + self.df['category_title'] + self.df['keywords'] +
                                       self.df[
                                           'excerpt'])
        tfidf = self.tfidf_vectorizer.fit_transform(txt_cleaned)
        tfidf_keywords_input = self.tfidf_vectorizer.transform(keywords_list)
        cosine_similarities = cosine_similarity(tfidf_keywords_input, tfidf).flatten()
        # cosine_similarities = linear_kernel(tfidf_keywords_input, tfidf).flatten()

        data_frame['coefficient'] = cosine_similarities

        # related_docs_indices = cosine_similarities.argsort()[:-(number+1):-1]
        related_docs_indices = cosine_similarities.argsort()[::-1][:k]

        closest = data_frame.sort_values('coefficient', ascending=False)[:k]

        closest.reset_index(inplace=True)
        # closest = closest.set_index('index1')
        closest['index1'] = closest.index
        # closest.index.name = 'index1'
        closest.columns.name = 'index'

        print("closest")
        print(closest.to_string())
        print(closest.columns)

        return closest[["slug", "coefficient"]]
        # return pd.DataFrame(closest).merge(items).head(k)

    def get_tfidf_vectorizer(self, fit_by, fit_by_2=None):
        """
        Metoda fit: výpočet průměru a rozptylu jednotlivých sloupců z dat.
        Metoda transformace: # transformuje všechny prvky pomocí příslušného průměru a rozptylu.
        """

        self.set_tfid_vectorizer()
        if fit_by_2 is None:
            print("self.df")
            print(self.df)
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[
                                                                        fit_by])
        else:
            self.df[fit_by] = self.df[fit_by_2] + " " + self.df[fit_by]
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[
                                                                        fit_by])

        return self.tfidf_tuples  # tuples of (document_id, token_id) and tf-idf score for it

    def set_tfid_vectorizer(self):
        # load_texts czech stopwords from file
        filename = Path("src/prefillers/preprocessing/stopwords/czech_stopwords.txt")
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

        filename = Path("src/prefillers/preprocessing/stopwords/general_stopwords.txt")
        with open(filename, encoding="utf-8") as file:
            general_stopwords = file.readlines()
            general_stopwords = [line.rstrip() for line in general_stopwords]
        # print(cz_stopwords)
        stopwords = cz_stopwords + general_stopwords

        # transforms text to feature vectors that can be used as input to estimator
        self.tfidf_vectorizer = TfidfVectorizer(dtype=np.float32,
                                                stop_words=stopwords)

    # # @profile
    def recommend_by_more_features(self, slug, tupple_of_fitted_matrices, num_of_recommendations=20):
        """
        # combining results of all feature types
        # combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)
        # creating sparse matrix containing mostly zeroes from combined feature tupples

        Example 1: solving linear system A*x=b where A is 5000x5000 but is block diagonal matrix constructed of 500 5x5
        blocks. Setup code:

        As = sparse(rand(5, 5));
        for(i=1:999)
           As = blkdiag(As, sparse(rand(5,5)));
        end;                         %As is made up of 500 5x5 blocks along diagonal
        Af = full(As); b = rand(5000, 1);

        Then you can tests speed difference:

        As \ b % operation on sparse As takes .0012 seconds
        Af \ b % solving with full Af takes about 2.3 seconds

        """
        print("tupple_of_fitted_matrices:")
        print(tupple_of_fitted_matrices)
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)
        print("combined_matrix1:")
        print(combined_matrix1)

        cosine_transform = CosineTransformer()
        self.cosine_sim_df = cosine_transform.get_cosine_sim_use_own_matrix(combined_matrix1, self.df)

        # getting posts with the highest similarity
        combined_all = self.get_closest_posts(slug, self.cosine_sim_df,
                                              self.df[['slug']], k=num_of_recommendations)

        recommended_posts_in_json = convert_dataframe_posts_to_json(combined_all, slug, self.cosine_sim_df)
        return recommended_posts_in_json

    def get_closest_posts(self, find_by_string, data_frame, items, k=20):
        print("self.cosine_sim_df:")
        print(self.cosine_sim_df)
        ix = data_frame.loc[:, find_by_string].to_numpy().argpartition(range(-1, -k, -1))
        closest = data_frame.columns[ix[-1:-(k + 2):-1]]
        # print("closest")
        # print(closest)
        # drop post itself
        closest = closest.drop(find_by_string, errors='ignore')

        # print("pd.DataFrame(closest).merge(items).head(k)")
        # print(pd.DataFrame(closest).merge(items).head(k))
        return pd.DataFrame(closest).merge(items).head(k)
