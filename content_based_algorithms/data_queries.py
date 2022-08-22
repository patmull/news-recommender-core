import json
import pickle
from pathlib import Path
from threading import Thread

import dropbox
import gensim
import numpy as np
import pandas as pd
import smart_open
from gensim.utils import deaccent
from nltk import FreqDist
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_connection import Database
import os
dirname = os.path.dirname(__file__)


def load_cz_stopwords(remove_punct=True):
    filename = "preprocessing/stopwords/czech_stopwords.txt"
    with open(filename, encoding="utf-8") as file:
        cz_stopwords = file.readlines()
        if remove_punct is False:
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        else:
            cz_stopwords = [gensim.utils.simple_preprocess(line.rstrip()) for line in cz_stopwords]
        return cz_stopwords


def load_general_stopwords():
    filename = "preprocessing/stopwords/general_stopwords.txt"
    with open(filename, encoding="utf-8") as file:
        general_stopwords = file.readlines()
        general_stopwords = [line.rstrip() for line in general_stopwords]
        return general_stopwords


def remove_stopwords(texts):
    stopwords_cz = load_cz_stopwords()
    stopwords_general = load_general_stopwords()
    stopwords = stopwords_cz + stopwords_general
    stopwords = flatten(stopwords)
    joined_stopwords = ' '.join(str(x) for x in stopwords)
    stopwords = deaccent(joined_stopwords)
    stopwords = stopwords.split(' ')
    return [[word for word in gensim.utils.simple_preprocess(doc) if word not in stopwords] for doc in texts]


def flatten(t):
    return [item for sublist in t for item in sublist]


class RecommenderMethods:

    def __init__(self):
        self.database = Database()

    def get_posts_dataframe(self, force_update=False):
        print("4.1.1")
        if force_update is True:
            self.posts_df = self.database.insert_posts_dataframe_to_cache()
        else:
            print("Trying reading from cache as default...")
            cached_file_path = "db_cache/cached_posts_dataframe.pkl"
            if os.path.isfile(cached_file_path):
                try:
                    print("Reading from cache...")
                    self.posts_df = self.database.get_posts_dataframe_from_cache()
                except Exception as e:
                    print(e)
                    print(e.with_traceback())
                    self.posts_df = self.get_df_from_sql_meanwhile_insert_cache()
            else:
                self.posts_df = self.get_df_from_sql_meanwhile_insert_cache()

        print("4.1.2")
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        print("4.1.3")
        return self.posts_df

    def get_df_from_sql_meanwhile_insert_cache(self):
        def update_cache(self):
            print("Inserting file to cache in the background...")
            self.database.insert_posts_dataframe_to_cache()

        print("Posts not found on cache. Will use PgSQL command.")
        posts_df = self.database.get_posts_dataframe_from_sql(pd)
        thread = Thread(target=update_cache, kwargs={'self': self})
        thread.start()
        return posts_df

    def get_users_dataframe(self):
        self.posts_df = self.database.get_users_dataframe()
        return self.posts_df

    def get_ratings_dataframe(self):
        self.posts_df = self.database.get_ratings_dataframe(pd)
        return self.posts_df

    def get_categories_dataframe(self, rename_title=True):
        # rename_title (defaul=False): for ensuring that category title does not collide with post title
        self.categories_df = self.database.get_categories_dataframe(pd)
        return self.categories_df

    def get_user_posts_ratings(self):
        database = Database()
        ##Step 1
        # database.set_row_var()
        # EXTRACT RESULTS FROM CURSOR

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id AS user_id, u.name, r.value AS rating_value
                            FROM posts p
                            JOIN ratings r ON r.post_id = p.id
                            JOIN users u ON r.user_id = u.id;"""
        # LOAD INTO A DATAFRAME
        df_ratings = pd.read_sql_query(sql_rating, database.get_cnx())
        return df_ratings

    def get_results_dataframe(self):
        results_df = self.database.get_results_dataframe(pd)
        results_df.reset_index(inplace=True)
        print("self.results_df:")
        print(results_df)
        results_df_ = results_df[['id', 'query_slug', 'results_part_1', 'results_part_2', 'user_id', 'model_name', 'model_variant', 'created_at']]
        return results_df_

    def refresh_cached_db_file(self):
        print("Refreshing DB cache...")
        self.database.insert_posts_dataframe_to_cache()
        print("New DB cache file saved to local storage.")

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
                    ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
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
                    ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
                     'all_features_preprocessed', 'body_preprocessed', 'doc2vec_representation', 'full_text', 'trigrams_full_text']]
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
                ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
                 'all_features_preprocessed', 'body_preprocessed', 'doc2vec_representation', 'trigrams_full_text']]
        return self.df

    def join_posts_ratings_categories_full_text(self):
        self.posts_df = self.posts_df.rename(columns = {'title': 'post_title'})
        print("self.posts_df.columns")
        print(self.posts_df.columns)
        self.posts_df = self.posts_df.rename(columns= {'slug': 'post_slug'})
        print("self.posts_df.columns")
        print(self.posts_df.columns)
        self.categories_df = self.categories_df.rename(columns = {'title': 'category_title'})

        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        print("self.df.columns")
        print(self.df.columns)
        self.df = self.df[
            ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
             'all_features_preprocessed', 'body_preprocessed', 'full_text', 'category_id']]
        return self.df


    #### Above are data queries ####

    def get_fit_by_feature_(self, feature_name, second_feature=None):
        fit_by_feature = self.get_tfIdfVectorizer(feature_name, second_feature)
        return fit_by_feature

    def recommend_by_keywords(self, keywords, tupple_of_fitted_matrices, number_of_recommended_posts=20):
        # combining results of all feature types to sparse matrix
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices, dtype=np.float16)

        # computing cosine similarity using matrix with combined features
        print("Computing cosine similarity using matrix with combined features...")
        self.set_cosine_sim_use_own_matrix(combined_matrix1)
        combined_all = self.get_recommended_posts_for_keywords(keywords, self.cosine_sim_df,
                                                               self.df[['keywords']], k=number_of_recommended_posts)
        # print("combined_all:")
        # print(combined_all)
        df_renamed = combined_all.rename(columns={'post_slug': 'slug'})
        # print("DF RENAMED:")
        # print(df_renamed)

        json = self.convert_to_json_keyword_based(df_renamed)

        return json

    def convert_to_json_keyword_based(self, post_recommendations):

        list_of_article_slugs = []

        # post_recommendations['coefficient'] = list_of_coefficients

        dict = post_recommendations.to_dict('records')

        list_of_article_slugs.append(dict.copy())
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs[0])
        return list_of_article_slugs[0]

    def get_recommended_posts_for_keywords(self, keywords, data_frame, items, k=10):

        keywords_list = [keywords]
        txt_cleaned = self.get_cleaned_text(self.df,
                                            self.df['post_title'] + self.df['category_title'] + self.df['keywords'] + self.df[
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

        return closest[["post_slug", "coefficient"]]
        # return pd.DataFrame(closest).merge(items).head(k)

    def find_post_by_slug(self, slug, force_update=False):
        recommenderMethods = RecommenderMethods()
        print("recommenderMethods.get_posts_dataframe():")
        print(recommenderMethods.get_posts_dataframe()['slug'])
        print("slug:")
        print(slug)
        return recommenderMethods.get_posts_dataframe().loc[recommenderMethods.get_posts_dataframe()['slug'] == slug]

    def get_cleaned_text(self, df, row):
        return row

    def get_tfIdfVectorizer(self, fit_by, fit_by_2=None):

        self.set_tfIdfVectorizer()

        if fit_by_2 is None:
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[
                                                                        fit_by])
        else:
            self.df[fit_by] = self.df[fit_by_2] + " " + self.df[fit_by]
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[
                                                                        fit_by])  # Metoda fit: výpočet průměru a rozptylu jednotlivých sloupců z dat. Metoda transformace: # transformuje všechny prvky pomocí příslušného průměru a rozptylu.

        return self.tfidf_tuples  # tuples of (document_id, token_id) and tf-idf score for it

    def set_tfIdfVectorizer(self):
        # load_texts czech stopwords from file
        filename = "preprocessing/stopwords/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]


        filename = "preprocessing/stopwords/general_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            general_stopwords = file.readlines()
            general_stopwords = [line.rstrip() for line in general_stopwords]
        # print(cz_stopwords)
        stopwords = cz_stopwords + general_stopwords

        tfidf_vectorizer = TfidfVectorizer(dtype=np.float32,
                                           stop_words=stopwords)  # transforms text to feature vectors that can be used as input to estimator
        print("tfidf_vectorizer")
        print(tfidf_vectorizer)
        self.tfidf_vectorizer = tfidf_vectorizer

    # # @profile
    def recommend_by_more_features(self, slug, tupple_of_fitted_matrices, num_of_recommendations=20):
        # combining results of all feature types
        # combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices) # creating sparse matrix containing mostly zeroes from combined feature tupples
        print("tupple_of_fitted_matrices:")
        print(tupple_of_fitted_matrices)
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)
        print("combined_matrix1:")
        print(combined_matrix1)
        """
        Example 1: solving linear system A*x=b where A is 5000x5000 but is block diagonal matrix constructed of 500 5x5 blocks. Setup code:

        As = sparse(rand(5, 5));
        for(i=1:999)
           As = blkdiag(As, sparse(rand(5,5)));
        end;                         %As is made up of 500 5x5 blocks along diagonal
        Af = full(As); b = rand(5000, 1);

        Then you can test speed difference:

        As \ b % operation on sparse As takes .0012 seconds
        Af \ b % solving with full Af takes about 2.3 seconds

        """
        # # print(combined_matrix1.shape)
        # computing cosine similarity
        self.set_cosine_sim_use_own_matrix(combined_matrix1)

        # getting posts with highest similarity
        combined_all = self.get_recommended_posts(slug, self.cosine_sim_df,
                                                  self.df[['slug']], k=num_of_recommendations)

        # json conversion
        json = self.convert_datframe_posts_to_json(combined_all, slug)
        return json

    def recommend_by_more_features_with_full_text(self, slug, tupple_of_fitted_matrices):
        """
        Example 1: solving linear system A*x=b where A is 5000x5000 but is block diagonal matrix constructed of 500 5x5 blocks. Setup code:

        As = sparse(rand(5, 5));
        for(i=1:999)
           As = blkdiag(As, sparse(rand(5,5)));
        end;                         %As is made up of 500 5x5 blocks along diagonal
        Af = full(As); b = rand(5000, 1);

        Then you can test speed difference:

        As \ b % operation on sparse As takes .0012 seconds
        Af \ b % solving with full Af takes about 2.3 seconds

        """
        # combining results of all feature types
        # combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices) # creating sparse matrix containing mostly zeroes from combined feature tupples
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)

        # computing cosine similarity
        print("Computing cosine simialirity")
        self.set_cosine_sim_use_own_matrix(combined_matrix1)

        # getting posts with highest similarity
        combined_all = self.get_recommended_posts(slug, self.cosine_sim_df,
                                                  self.df[['slug']])

        # json conversion
        json = self.convert_datframe_posts_to_json(combined_all, slug)

        return json

    def set_cosine_sim_use_own_matrix(self, own_tfidf_matrix):
        own_tfidf_matrix_csr = sparse.csr_matrix(own_tfidf_matrix.astype(dtype=np.float16)).astype(dtype=np.float16)
        cosine_sim = self.cosine_similarity_n_space(own_tfidf_matrix_csr, own_tfidf_matrix_csr)
        print("cosine_sim:")
        print(cosine_sim)
        # cosine_sim = cosine_similarity(own_tfidf_matrix_csr) # computing cosine similarity
        cosine_sim_df = pd.DataFrame(cosine_sim, index=self.df['slug'],
                                     columns=self.df['slug'])  # finding original record of post belonging to slug
        del cosine_sim
        self.cosine_sim_df = cosine_sim_df

    def cosine_similarity_n_space(self, m1, m2=None, batch_size=100):
        assert m1.shape[1] == m2.shape[1] and isinstance(batch_size, int) == True

        ret = np.ndarray((m1.shape[0], m2.shape[0]), dtype=np.float16)

        # iterating through dataframe by batches
        batches = m1.shape[0] // batch_size

        if m1.shape[0] % batch_size != 0:
            batches = batches + 1

        for row_i in range(0, batches):
            start = row_i * batch_size
            end = min([(row_i + 1) * batch_size, m1.shape[0]])
            rows = m1[start: end]
            sim = cosine_similarity(rows, m2)
            ret[start: end] = sim

        return ret

    def get_recommended_posts(self, find_by_string, data_frame, items, k=20):
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

    def convert_datframe_posts_to_json(self, post_recommendations, slug):
        list_of_article_slugs = []
        list_of_coefficients = []

        # finding coefficient belonging to recommended posts compared to original post (for which we want to find recommendations)
        if 'post_slug' in post_recommendations:
            post_recommendations = post_recommendations.rename(columns={'post_slug': 'slug'})

        for index, row in post_recommendations.iterrows():

            list_of_coefficients.append(self.cosine_sim_df.at[row['slug'], slug])

        post_recommendations['coefficient'] = list_of_coefficients
        dict = post_recommendations.to_dict('records')
        list_of_article_slugs.append(dict.copy())
        return list_of_article_slugs[0]

    def convert_df_to_json(self, dataframe):
        result = dataframe[["title", "excerpt", "body"]].to_json(orient="records", lines=True)
        parsed = json.loads(result)
        return parsed

    @DeprecationWarning
    def download_from_amazon(self, amazon_bucket_url):
        amazon_model = pickle.load(smart_open.smart_open(self.amazon_bucket_url))
        return amazon_model

    def dropbox_file_download(self, access_token, dropbox_file_path, local_folder_name):
        try:
            dbx = dropbox.Dropbox(access_token)

            """
            print("Dropbox Files:")
            for entry in dbx.files_list_folder('').entries:
                print(entry.path_lower)
            """
            dbx.files_download_to_file(dropbox_file_path, local_folder_name)

        except Exception as e:
            print(e)
            return False

    @staticmethod
    def most_common_words(all_words):
        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)
        k = 150
        return zip(*fdist.most_common(k))

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def get_posts_categories_dataframe(self):
        posts_df = self.get_posts_dataframe()
        categories_df = self.get_categories_dataframe()
        posts_df = posts_df.rename(columns={'title': 'post_title'})
        categories_df = categories_df.rename(columns={'title': 'category_title', 'slug': 'category_slug'})
        print("self.posts_df")
        print(self.posts_df)
        print("self.categories_df")
        print(self.categories_df)
        self.df = pd.merge(posts_df, categories_df, left_on='category_id', right_on='id')
        return self.df


def dropbox_file_download(access_token, dropbox_file_path, local_folder_name):
    try:
        dbx = dropbox.Dropbox(access_token)

        """
        print("Dropbox Files:")
        for entry in dbx.files_list_folder('').entries:
            print(entry.path_lower)
        """
        dbx.files_download_to_file(dropbox_file_path, local_folder_name)

    except Exception as e:
        print(e)
        return False