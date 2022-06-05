import json
import pickle
from pathlib import Path

import dropbox
import gensim
import numpy as np
import pandas as pd
import smart_open
from nltk import FreqDist
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_conenction import Database
import os
dirname = os.path.dirname(__file__)


def load_cz_stopwords():
    filename = "preprocessing/stopwords/czech_stopwords.txt"
    with open(filename, encoding="utf-8") as file:
        cz_stopwords = file.readlines()
        cz_stopwords = [line.rstrip() for line in cz_stopwords]
        return cz_stopwords


def load_general_stopwords():
    filename = "preprocessing/stopwords/general_stopwords.txt"
    with open(filename, encoding="utf-8") as file:
        general_stopwords = file.readlines()
        general_stopwords = [line.rstrip() for line in general_stopwords]
        return general_stopwords


def remove_stopwords(texts):
    stopwords = load_cz_stopwords()
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]


class RecommenderMethods:

    def __init__(self):
        self.database = Database()

    def get_posts_dataframe(self):
        self.database.connect()
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        self.database.disconnect()
        return self.posts_df

    def get_users_dataframe(self):
        self.database.connect()
        self.posts_df = self.database.get_all_users()
        self.database.disconnect()
        return self.posts_df

    def get_ratings_dataframe(self):
        self.database.connect()
        self.posts_df = self.database.get_ratings_dataframe(pd)
        self.database.disconnect()
        return self.posts_df

    def get_categories_dataframe(self):
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe(pd)
        self.database.disconnect()
        return self.categories_df

    def get_user_posts_ratings(self):
        database = Database()
        database.connect()
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

    def join_posts_ratings_categories(self, include_prefilled=False):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        if include_prefilled is False:
            self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
            # clean up from unnecessary columns
            self.df = self.df[
                ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
                 'all_features_preprocessed', 'body_preprocessed']]
        else:
            self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
            # clean up from unnecessary columns
            try:
                self.df = self.df[
                    ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
                     'all_features_preprocessed', 'body_preprocessed',
                     'recommended_tfidf_full_text']]
            except KeyError:
                self.df = self.database.insert_posts_dataframe_to_cache()
                self.posts_df.drop_duplicates(subset=['title'], inplace=True)
                print(self.df.columns.values)
                self.df = self.df[
                    ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
                     'all_features_preprocessed', 'body_preprocessed',
                     'recommended_tfidf_full_text']]
        return self.df


    #### Above are data queries ####

    def get_fit_by_feature(self, feature_name, second_feature=None):
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
        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})
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
                                            self.df['title_x'] + self.df['title_y'] + self.df['keywords'] + self.df[
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

        return closest[["slug_x", "coefficient"]]
        # return pd.DataFrame(closest).merge(items).head(k)

    def find_post_by_slug(self, slug):
        recommenderMethods = RecommenderMethods()
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
        # load czech stopwords from file
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
                                                  self.df[['slug_x']], k=num_of_recommendations)

        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})
        # json conversion
        json = self.convert_datframe_posts_to_json(df_renamed, slug)

        return json


    def recommend_by_more_features_with_full_text(self, slug, tupple_of_fitted_matrices):
        # combining results of all feature types
        # combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices) # creating sparse matrix containing mostly zeroes from combined feature tupples
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)
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
        # computing cosine similarity
        print("Computing cosine simialirity")
        self.set_cosine_sim_use_own_matrix(combined_matrix1)

        # getting posts with highest similarity
        combined_all = self.get_recommended_posts(slug, self.cosine_sim_df,
                                                  self.df[['slug_x']])

        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})

        # json conversion
        json = self.convert_datframe_posts_to_json(df_renamed, slug)

        return json

    def set_cosine_sim_use_own_matrix(self, own_tfidf_matrix):
        own_tfidf_matrix_csr = sparse.csr_matrix(own_tfidf_matrix.astype(dtype=np.float16)).astype(dtype=np.float16)
        cosine_sim = self.cosine_similarity_n_space(own_tfidf_matrix_csr, own_tfidf_matrix_csr)
        print("cosine_sim:")
        print(cosine_sim)
        # cosine_sim = cosine_similarity(own_tfidf_matrix_csr) # computing cosine similarity
        cosine_sim_df = pd.DataFrame(cosine_sim, index=self.df['slug_x'],
                                     columns=self.df['slug_x'])  # finding original record of post belonging to slug
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