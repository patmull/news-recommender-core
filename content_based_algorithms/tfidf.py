import gc
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from content_based_algorithms.data_queries import RecommenderMethods
from data_connection import Database


class TfIdf:

    def __init__(self):
        self.posts_df = None
        self.ratings_df = None
        self.categories_df = None
        self.df = None
        self.database = Database()
        self.user_categories_df = None
        self.tfidf_tuples = None
        self.tfidf_vectorizer = None
        self.cosine_sim_df = None

    # @profile
    def keyword_based_comparison(self, keywords, number_of_recommended_posts=20, all_posts=False):

        if keywords == "":
            return {}

        keywords_splitted_1 = keywords.split(" ")  # splitting sentence into list of keywords by space

        # creating dictionary of words
        wordDictA = dict.fromkeys(keywords_splitted_1, 0)
        for word in keywords_splitted_1:
            wordDictA[word] += 1

        recommender_methods = RecommenderMethods()
        recommender_methods.database.connect()
        recommender_methods.get_posts_categories_dataframe()
        recommender_methods.database.disconnect()

        # same as "classic" tf-idf
        """
        fit_by_title_matrix = recommenderMethods.get_fit_by_feature_('title_x', 'category_title')  # prepended by category
        print("6")
        """
        cached_file_path = "precalculated/all_features_preprocessed_vectorizer.pickle"
        if os.path.isfile(cached_file_path):
            print("Cached file already exists.")
            fit_by_all_features_preprocessed = self.load_tfidf_vectorizer(path=cached_file_path)
        else:
            print("Not found .pickle file with precalculated vectors. Calculating fresh.")
            fit_by_all_features_preprocessed = recommender_methods.get_fit_by_feature_('all_features_preprocessed')
            self.save_tfidf_vectorizer(fit_by_all_features_preprocessed, path=cached_file_path)

        print("7")

        fit_by_keywords_matrix = recommender_methods.get_fit_by_feature_('keywords')
        print("8")

        tuple_of_fitted_matrices = (fit_by_all_features_preprocessed, fit_by_keywords_matrix)
        print("9")
        del fit_by_all_features_preprocessed, fit_by_keywords_matrix
        gc.collect()
        if all_posts is False:
            post_recommendations = recommender_methods.recommend_by_keywords(keywords, tuple_of_fitted_matrices,
                                                                            number_of_recommended_posts=number_of_recommended_posts)
            print("10")

        if all_posts is True:
            post_recommendations = recommender_methods.recommend_by_keywords(keywords, tuple_of_fitted_matrices,
                                                                            number_of_recommended_posts=len(recommender_methods.posts_df.index))
        del recommender_methods
        return post_recommendations

    # https://datascience.stackexchange.com/questions/18581/same-tf-idf-vectorizer-for-2-data-inputs
    def set_tfidf_vectorizer_combine_features(self):
        tfidf_vectorizer = TfidfVectorizer()
        self.df.drop_duplicates(subset=['title_x'], inplace=True)
        tf_train_data = pd.concat([self.df['category_title'], self.df['keywords'], self.df['title_x'], self.df['excerpt']])
        tfidf_vectorizer.fit_transform(tf_train_data)

        tf_idf_title_x = tfidf_vectorizer.transform(self.df['title_x'])
        tf_idf_category_title = tfidf_vectorizer.transform(self.df['category_title'])  # category title
        tf_idf_keywords = tfidf_vectorizer.transform(self.df['keywords'])
        tf_idf_excerpt = tfidf_vectorizer.transform(self.df['excerpt'])

        model = LogisticRegression()
        model.fit([tf_idf_title_x.shape, tf_idf_category_title.shape, tf_idf_keywords.shape, tf_idf_excerpt.shape],
                  self.df['excerpt'])

    def set_cosine_sim(self):
        cosine_sim = cosine_similarity(self.tfidf_tuples)
        cosine_sim_df = pd.DataFrame(cosine_sim, index=self.df['slug_x'], columns=self.df['slug_x'])
        self.cosine_sim_df = cosine_sim_df

    # # @profile

    # @profile
    def get_cleaned_text(self, df, row):
        return row

    def get_recommended_posts_for_keywords(self, keywords, data_frame, k=10):

        keywords_list = []
        keywords_list.append(keywords)
        txt_cleaned = self.get_cleaned_text(self.df,
                                            self.df['title_x'] + self.df['category_title'] + self.df['keywords'] + self.df[
                                                'excerpt'])
        tfidf = self.tfidf_vectorizer.fit_transform(txt_cleaned)
        tfidf_keywords_input = self.tfidf_vectorizer.transform(keywords_list)
        cosine_similarities = cosine_similarity(tfidf_keywords_input, tfidf).flatten()

        data_frame['coefficient'] = cosine_similarities

        closest = data_frame.sort_values('coefficient', ascending=False)[:k]

        closest.reset_index(inplace=True)
        closest['index1'] = closest.index
        closest.columns.name = 'index'

        return closest[["slug_x", "coefficient"]]
        # return pd.DataFrame(closest).merge(items).head(k)

    def prepare_dataframes(self):
        recommender_methods = RecommenderMethods()
        recommender_methods.database.connect()
        recommender_methods.get_posts_categories_dataframe()
        recommender_methods.database.disconnect()
        self.df = recommender_methods.join_posts_ratings_categories(include_prefilled=True)  # joining posts and categories into one table

    def get_prefilled_full_text(self, slug):
        recommender_methods = RecommenderMethods()
        recommender_methods.database.connect()
        recommender_methods.get_posts_categories_dataframe()
        recommender_methods.database.disconnect()

    def save_sparse_matrix(self, recommenderMethods):
        print("Loading posts.")
        fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature_('all_features_preprocessed')
        print("Saving sparse matrix into file...")
        self.save_sparse_csr(filename="models/tfidf_all_features_preprocessed.npz", array=fit_by_all_features_matrix)
        return fit_by_all_features_matrix

    # @profile
    def recommend_posts_by_all_features_preprocessed(self, slug, num_of_recommendations=20, force_update=False):
        """
        This method differs from Fresh API module's method. This method is more optimized for "offline" use among
        prefillers
        """

        recommender_methods = RecommenderMethods()
        recommender_methods.database.connect()
        recommender_methods.get_posts_categories_dataframe()
        recommender_methods.database.disconnect()

        my_file = Path("models/tfidf_all_features_preprocessed.npz")
        if my_file.exists() is False:
            fit_by_all_features_matrix = self.save_sparse_matrix(recommender_methods)
        else:
            fit_by_all_features_matrix = self.load_sparse_csr(filename="models/tfidf_all_features_preprocessed.npz")

        my_file = Path("models/tfidf_category_title.npz")
        if my_file.exists() is False:
            # category_title = category
            fit_by_title = recommender_methods.get_fit_by_feature_('category_title')
            self.save_sparse_csr(filename="models/tfidf_category_title.npz", array=fit_by_title)
        else:
            fit_by_title = self.load_sparse_csr(filename="models/tfidf_category_title.npz")

        tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title) # join feature tuples into one matrix

        gc.collect()

        print("tuple_of_fitted_matrices")
        print(tuple_of_fitted_matrices)

        try:
            post_recommendations = recommender_methods.recommend_by_more_features(slug, tuple_of_fitted_matrices, num_of_recommendations=num_of_recommendations)
        except ValueError:
            fit_by_all_features_matrix = self.save_sparse_matrix(recommender_methods)
            fit_by_title = recommender_methods.get_fit_by_feature_('category_title')
            self.save_sparse_csr(filename="models/tfidf_category_title.npz", array=fit_by_title)
            fit_by_title = self.load_sparse_csr(filename="models/tfidf_category_title.npz")
            tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title)
            post_recommendations = recommender_methods.recommend_by_more_features(slug, tuple_of_fitted_matrices, num_of_recommendations=num_of_recommendations)

        del recommender_methods, tuple_of_fitted_matrices
        return post_recommendations

    # @profile
    def recommend_posts_by_all_features_preprocessed_with_full_text(self, slug):

        recommender_methods = RecommenderMethods()
        print("Loading posts")
        recommender_methods.database.connect()
        recommender_methods.get_posts_categories_dataframe()
        recommender_methods.database.disconnect()
        gc.collect()

        # replacing None values with empty strings
        recommender_methods.df['full_text'] = recommender_methods.df['full_text'].replace([None], '')

        fit_by_all_features_matrix = recommender_methods.get_fit_by_feature_('all_features_preprocessed')
        fit_by_title = recommender_methods.get_fit_by_feature_('category_title')
        fit_by_full_text = recommender_methods.get_fit_by_feature_('full_text')

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_title, fit_by_all_features_matrix, fit_by_full_text)
        del fit_by_title
        del fit_by_all_features_matrix
        del fit_by_full_text
        gc.collect()

        post_recommendations = recommender_methods.recommend_by_more_features_with_full_text(slug,
                                                                                            tuple_of_fitted_matrices)
        del recommender_methods
        return post_recommendations

    # # @profile
    def recommend_posts_by_all_features(self, slug):

        recommender_methods = RecommenderMethods()
        recommender_methods.database.connect()
        recommender_methods.get_posts_categories_dataframe()
        recommender_methods.database.disconnect()

        # preprocessing

        # feature tuples of (document_id, token_id) and coefficient
        fit_by_post_title_matrix = recommender_methods.get_fit_by_feature_('post_title', 'category_title')
        print("fit_by_post_title_matrix")
        print(fit_by_post_title_matrix)
        # fit_by_category_matrix = recommenderMethods.get_fit_by_feature_('category_title')
        fit_by_excerpt_matrix = recommender_methods.get_fit_by_feature_('excerpt')
        print("fit_by_excerpt_matrix")
        print(fit_by_excerpt_matrix)
        fit_by_keywords_matrix = recommender_methods.get_fit_by_feature_('keywords')
        print("fit_by_keywords_matrix")
        print(fit_by_keywords_matrix)

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_post_title_matrix, fit_by_excerpt_matrix, fit_by_keywords_matrix)
        post_recommendations = recommender_methods.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del recommender_methods
        return post_recommendations

    def convert_to_json_one_row(self, key, value):
        list_for_json = []
        dict_for_json = {key: value}
        list_for_json.append(dict_for_json)
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_for_json)
        return list_for_json

    def convert_to_json_keyword_based(self, post_recommendations):

        list_of_article_slugs = []
        dict = post_recommendations.to_dict('records')
        list_of_article_slugs.append(dict.copy())
        return list_of_article_slugs[0]

    def save_sparse_csr(self, filename, array):
        np.savez(filename, data=array.data, indices=array.indices,
                 indptr=array.indptr, shape=array.shape)

    def update_saved_matrix(self):
        recommenderMethods = RecommenderMethods()
        fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature_('all_features_preprocessed')
        self.save_sparse_csr(filename="models/tfidf_all_features_preprocessed.npz", array=fit_by_all_features_matrix)

    def save_tfidf_vectorizer(self, vector, path="precalculated/all_features_preprocessed_vectorizer.pickle"):
        print("Saving TfIdf vectorizer of posts")
        pickle.dump(vector, open(path, "wb"))

    def load_tfidf_vectorizer(self, path="precalculated/all_features_preprocessed_vectorizer.pickle"):
        # TODO: refreshing when new posts added
        print("Loading TfIdf vectorizer of posts")
        vectorizer = pickle.load(open(path, "rb"))
        return vectorizer

    def load_sparse_csr(self, filename):
        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])

    def analyze(self, slug):
        """
        Specificaly part of this module in order to perform data visualizations for the case studies
        :param slug:
        :return:
        """

        recommenderMethods = RecommenderMethods()
        found_posts = []
        queried_post = recommenderMethods.find_post_by_slug(slug)
        path_to_saved_results = "research/tfidf/presaved_results.pkl"
        path_to_saved_ids = "research/tfidf/presaved_ids.pkl"
        ids = []
        print("Index of queried:")
        print(queried_post.index[0])
        ids.append(queried_post.index[0])

        if os.path.exists(path_to_saved_results):
            print("Results found. Loading...")
            with open(path_to_saved_results, 'rb') as opened_file:
                found_posts = pickle.load(opened_file)
        else:
            recommended_posts = self.recommend_posts_by_all_features_preprocessed(slug)
            # queried doc
            print("Results not found. Querying results...")
            found_posts.append(queried_post.iloc[0]['all_features_preprocessed'])
            for post in recommended_posts:
                found_post = recommenderMethods.find_post_by_slug(post['slug'])
                found_posts.append(found_post.iloc[0]['all_features_preprocessed'])
                ids.append(found_post.index[0])
                print("found_posts")
                print(found_posts)

            with open(path_to_saved_results, 'wb') as opened_file:
                pickle.dump(found_posts, opened_file)

            with open(path_to_saved_ids, 'wb') as opened_file:
                pickle.dump(ids, opened_file)

        # TODO: Save to pickle
        filename = "preprocessing/stopwords/general_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            general_stopwords = file.readlines()
            general_stopwords = [line.rstrip() for line in general_stopwords]

        filename = "preprocessing/stopwords/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in general_stopwords]
        # print(cz_stopwords)
        stopwords = cz_stopwords + general_stopwords

        cv = CountVectorizer(stop_words=stopwords)
        for found_post in found_posts:
            cv_fit = cv.fit_transform([found_post])
            word_list = cv.get_feature_names()
            count_list = cv_fit.toarray().sum(axis=0)
            print("found_post")
            print(found_post)
            print("word_list:")
            print(word_list)
            print("dict:")
            print(dict(zip(word_list, count_list)))

        matched_words_results = []
        queried_post = recommenderMethods.find_post_by_slug(slug)

        posts = recommenderMethods.get_posts_dataframe()
        for found_post in found_posts:
            print(found_post)
            queried_post_stopwords_free = remove_stopwords([queried_post.iloc[0]['all_features_preprocessed']])[0]
            found_post_stopwords_free = remove_stopwords([found_post])[0]

            list_tmp = set(queried_post_stopwords_free) & set(found_post_stopwords_free)  # we don't need to list3 to actually be a list
            list_tmp = sorted(list_tmp, key=lambda k: queried_post_stopwords_free.index(k))
            print(list_tmp)
            if len(list_tmp) > 0:
                matched_words_results.append(list_tmp[0])
                print("----------------")

                print("IDF values:")
                print(found_post)
                print("----------------")
                print("Simple example:")
                word_count = cv.fit_transform(found_posts)
                print(word_count.shape)
                print(word_count)
                print(word_count.toarray())
                tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
                tfidf_transformer.fit(word_count)
                df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
                print(df_idf.sort_values(by=['idf_weights']).head(40))
            else:
                print("No matches found")

        print("---------------")
        print("From whole dataset:")
        word_count = cv.fit_transform(posts['all_features_preprocessed'])
        print(word_count.shape)
        print(word_count)
        print(word_count.toarray())
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count)
        df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
        print(df_idf.sort_values(by=['idf_weights']).head(40))

        print("----------------")
        print("TF-IDF values:")
        # dataframe_number =
        print("----------------")
        print("Simple example:")
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_vectors = tfidf_vectorizer.fit_transform(found_posts)
        # dataframe number
        first_vector_tfidfvectorizer = tfidf_vectors[0]
        df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                          columns=["tfidf"])
        print(df.sort_values(by=["tfidf"], ascending=False).head(45))

        print("---------------")
        print("From whole dataset:")
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_vectors = tfidf_vectorizer.fit_transform(posts['all_features_preprocessed'])
        # dataframe number
        # TODO: loop through recommended posts
        for id in ids:
            first_vector_tfidfvectorizer = tfidf_vectors[id]
            df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                              columns=["tfidf"])
            print(df.sort_values(by=["tfidf"], ascending=False).head(45))

        text_titles = posts['all_features_preprocessed'].to_list()

        tfidf_visualizer = TfIdfVisualizer()
        tfidf_visualizer.prepare_for_heatmap(tfidf_vectors, text_titles, tfidf_vectorizer)
        tfidf_visualizer.plot_tfidf_heatmap()


