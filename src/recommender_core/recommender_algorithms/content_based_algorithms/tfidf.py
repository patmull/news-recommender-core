import gc
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

import project_config.trials_counter

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import TfIdfDataHandlers, RecommenderMethods
from src.prefillers.preprocessing.stopwords_loading import remove_stopwords
from research.visualisation.tfidf_visualisation import TfIdfVisualizer

import logging

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from tfidf module.")


def get_cleaned_text(row):
    return row


def get_prefilled_full_text():
    recommender_methods = RecommenderMethods()
    recommender_methods.get_posts_categories_dataframe()


def convert_to_json_one_row(key, value):
    list_for_json = []
    dict_for_json = {key: value}
    list_for_json.append(dict_for_json)
    # print("------------------------------------")
    # print("JSON:")
    # print("------------------------------------")
    # print(list_for_json)
    return list_for_json


def convert_to_json_keyword_based(post_recommendations):
    list_of_article_slugs = []
    recommended_posts_dictionary = post_recommendations.to_dict('records')
    list_of_article_slugs.append(recommended_posts_dictionary.copy())
    return list_of_article_slugs[0]


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def save_tfidf_vectorizer(vector, path):
    print("Saving TfIdf vectorizer of posts")
    pickle.dump(vector, open(path, "wb"))


def load_tfidf_vectorizer(path=Path("precalc_vectors/all_features_preprocessed_vectorizer.pickle")):
    # TODO: refreshing when new posts added
    print("Loading TfIdf vectorizer of posts")
    vectorizer = pickle.load(open(path, "rb"))
    return vectorizer


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def display_top(word_count, cv):
    print(word_count.shape)
    print(word_count)
    print(word_count.toarray())
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
    print(df_idf.sort_values(by=['idf_weights']).head(40))


class TfIdf:

    def __init__(self):
        self.posts_df = pd.DataFrame()
        self.ratings_df = pd.DataFrame()
        self.categories_df = pd.DataFrame()
        self.df = pd.DataFrame()
        self.database = DatabaseMethods()
        self.user_categories_df = pd.DataFrame()
        self.tfidf_tuples = None
        self.tfidf_vectorizer = None
        self.cosine_sim_df = pd.DataFrame()

    # @profile
    def keyword_based_comparison(self, keywords, number_of_recommended_posts=20, all_posts=False):
        if type(keywords) is not str:
            raise ValueError("Entered keywords must be a input_string.")
        else:
            if keywords == "":
                raise ValueError("Entered input_string is empty.")
            else:
                pass

        post_recommendations = None
        if keywords == "":
            return {}

        keywords_splitted_1 = keywords.split(" ")  # splitting sentence into list of keywords by space

        # creating dictionary of words
        word_dict_a = dict.fromkeys(keywords_splitted_1, 0)
        for word in keywords_splitted_1:
            word_dict_a[word] += 1

        recommender_methods = RecommenderMethods()
        self.df = recommender_methods.get_posts_categories_dataframe()
        tfidf_data_handlers = TfIdfDataHandlers(self.df)

        cached_file_path = Path("precalc_vectors/all_features_preprocessed_vectorizer.pickle")
        if os.path.isfile(cached_file_path):
            print("Cached file already exists.")
            fit_by_all_features_preprocessed = load_tfidf_vectorizer(path=cached_file_path)
        else:
            print("Not found .pickle file with precalculated vectors. Calculating fresh.")
            fit_by_all_features_preprocessed = tfidf_data_handlers.get_fit_by_feature_('all_features_preprocessed')
            save_tfidf_vectorizer(fit_by_all_features_preprocessed, path=cached_file_path)

        print("7")

        fit_by_keywords_matrix = tfidf_data_handlers.get_fit_by_feature_('keywords')
        print("8")

        tuple_of_fitted_matrices = (fit_by_all_features_preprocessed, fit_by_keywords_matrix)
        print("9")
        del fit_by_all_features_preprocessed, fit_by_keywords_matrix
        gc.collect()
        if all_posts is False:
            try:
                post_recommendations = tfidf_data_handlers \
                    .most_similar_by_keywords(keywords, tuple_of_fitted_matrices,
                                              number_of_recommended_posts=number_of_recommended_posts)
            except ValueError as e:
                print("Value error had occurred while computing most similar posts by keywords. Error:")
                print(e)
                print("Trying to solve this by updating the tfidf vectorizer file")
                fit_by_all_features_preprocessed = tfidf_data_handlers.get_fit_by_feature_('all_features_preprocessed')
                save_tfidf_vectorizer(fit_by_all_features_preprocessed, path=cached_file_path)
                post_recommendations = tfidf_data_handlers \
                    .most_similar_by_keywords(keywords, tuple_of_fitted_matrices,
                                              number_of_recommended_posts=number_of_recommended_posts)
            print("10")

        if all_posts is True:
            if self.posts_df.index is not None:
                try:
                    post_recommendations = tfidf_data_handlers \
                        .most_similar_by_keywords(keywords, tuple_of_fitted_matrices,
                                                  number_of_recommended_posts=len(self.posts_df.index))
                except ValueError as e:
                    print("Value error had occurred while computing most similar posts by keywords. Error:")
                    print(e)
                    print("Trying to solve this by updating the tfidf vectorizer file")
                    fit_by_all_features_preprocessed = tfidf_data_handlers.get_fit_by_feature_(
                        'all_features_preprocessed')
                    save_tfidf_vectorizer(fit_by_all_features_preprocessed, path=cached_file_path)
                    post_recommendations = tfidf_data_handlers \
                        .most_similar_by_keywords(keywords, tuple_of_fitted_matrices,
                                                  number_of_recommended_posts=len(self.posts_df.index))
            else:
                raise ValueError("Dataframe of posts is None. Cannot continue with next operation.")
        del tfidf_data_handlers
        return post_recommendations

    # https://datascience.stackexchange.com/questions/18581/same-tf-idf-vectorizer-for-2-data-inputs
    def set_tfidf_vectorizer_combine_features(self):
        tfidf_vectorizer = TfidfVectorizer()
        if self.df is None:
            raise ValueError("self.df is set to None. Cannot continue to next operation.")
        self.df = self.df.drop_duplicates(subset=['title_x'])
        tf_train_data = pd.concat([self.df['category_title'], self.df['keywords'], self.df['title_x'],
                                   self.df['excerpt']])
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

    def prepare_dataframes(self):
        recommender_methods = RecommenderMethods()
        recommender_methods.get_posts_categories_dataframe()
        # joining posts and categories into one table
        self.df = recommender_methods.join_posts_ratings_categories(include_prefilled=True)

    def save_sparse_matrix(self, for_hybrid=False):
        logging.info("Loading sparse matrix.")

        if for_hybrid is False:
            path = Path("models/tfidf_all_features_preprocessed.npz")
        else:
            path = Path("models/for_hybrid/tfidf_all_features_preprocessed.npz")

        tfidf_data_handlers = TfIdfDataHandlers(self.df)
        fit_by_all_features_matrix = tfidf_data_handlers.get_fit_by_feature_('all_features_preprocessed')
        logging.info("Saving sparse matrix into file...")
        save_sparse_csr(filename=path, array=fit_by_all_features_matrix)
        return fit_by_all_features_matrix

    def recommend_posts_by_all_features_preprocessed(self, searched_slug, num_of_recommendations=20):
        """
        This method differs from Fresh API module's method. This method is more optimized for "offline" use among
        prefillers
        """
        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a input_string.")
        else:
            if searched_slug == "":
                raise ValueError("Entered input_string is empty.")
            else:
                pass

        recommender_methods = RecommenderMethods()
        self.df = recommender_methods.get_posts_categories_dataframe()

        fit_by_all_features_matrix = self.load_matrix()

        fit_by_title = self.get_fit_by_title()
        tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title)

        gc.collect()

        print("tuple_of_fitted_matrices")
        print(tuple_of_fitted_matrices)

        if searched_slug not in self.df['slug'].to_list():
            print('Slug does not appear in dataframe. Refreshing datafreme of posts.')
            recommender_methods = RecommenderMethods()
            recommender_methods.get_posts_dataframe(force_update=True)
            self.df = recommender_methods.get_posts_categories_dataframe(from_cache=True)

        try:
            tfidf_data_handlers = TfIdfDataHandlers(self.df)
            recommended_post_recommendations = tfidf_data_handlers \
                .recommend_by_more_features(slug=searched_slug, tupple_of_fitted_matrices=tuple_of_fitted_matrices,
                                            num_of_recommendations=num_of_recommendations)
        except ValueError:
            fit_by_all_features_matrix = self.save_sparse_matrix()
            tfidf_data_handlers = TfIdfDataHandlers(self.df)
            fit_by_title = tfidf_data_handlers.get_fit_by_feature_('category_title')
            save_sparse_csr(filename="models/tfidf_category_title.npz", array=fit_by_title)
            fit_by_title = load_sparse_csr(filename="models/tfidf_category_title.npz")
            tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title)
            recommended_post_recommendations = tfidf_data_handlers \
                .recommend_by_more_features(slug=searched_slug, tupple_of_fitted_matrices=tuple_of_fitted_matrices,
                                            num_of_recommendations=num_of_recommendations)

        del recommender_methods, tuple_of_fitted_matrices
        return recommended_post_recommendations

    # @profile
    # TODO: Merge ful text and short text method into one method. Distinguish only with parameter.
    def recommend_posts_by_all_features_preprocessed_with_full_text(self, searched_slug, posts_from_cache=True):

        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a input_string.")
        else:
            if searched_slug == "":
                raise ValueError("Entered input_string is empty.")
            else:
                pass

        recommender_methods = RecommenderMethods()
        print("Loading posts")
        self.df = recommender_methods.get_posts_categories_dataframe(from_cache=posts_from_cache)
        gc.collect()

        if searched_slug not in self.df['slug'].to_list():
            if project_config.trials_counter.NUM_OF_TRIALS < 1:
                print('Slug does not appear in dataframe. Refreshing datafreme of posts.')
                recommender_methods = RecommenderMethods()
                recommender_methods.get_posts_dataframe(force_update=True)
                self.df = recommender_methods.get_posts_categories_dataframe(from_cache=True)

                project_config.trials_counter.NUM_OF_TRIALS += 1

                self.recommend_posts_by_all_features_preprocessed_with_full_text(searched_slug, posts_from_cache)
            else:
                project_config.trials_counter.NUM_OF_TRIALS = 0
                raise ValueError("searched_slug not in dataframe. Tried to deal with this by updating posts_categories "
                                 "df but didn't helped")
        # replacing None values with empty strings
        recommender_methods.df['full_text'] = recommender_methods.df['full_text'].replace([None], '')

        tf_idf_data_handlers = TfIdfDataHandlers(self.df)

        fit_by_all_features_matrix = tf_idf_data_handlers.get_fit_by_feature_('all_features_preprocessed')
        fit_by_title = tf_idf_data_handlers.get_fit_by_feature_('category_title')
        fit_by_full_text = tf_idf_data_handlers.get_fit_by_feature_('full_text')

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_title, fit_by_all_features_matrix, fit_by_full_text)
        del fit_by_title
        del fit_by_all_features_matrix
        del fit_by_full_text
        gc.collect()

        tf_idf_data_handlers = TfIdfDataHandlers(self.df)
        recommended_post_recommendations = tf_idf_data_handlers \
            .recommend_by_more_features(searched_slug, tuple_of_fitted_matrices)
        del recommender_methods
        return recommended_post_recommendations

    # # @profile
    def recommend_posts_by_all_features(self, slug):

        recommender_methods = RecommenderMethods()
        recommender_methods.get_posts_categories_dataframe()

        # preprocessing

        # feature tuples of (document_id, token_id) and coefficient
        tf_idf_data_handlers = TfIdfDataHandlers(self.df)
        fit_by_post_title_matrix = tf_idf_data_handlers.get_fit_by_feature_('post_title', 'category_title')
        tuple_of_fitted_matrices = tf_idf_data_handlers.get_tupple_of_fitted_matrices(fit_by_post_title_matrix)
        tf_idf_data_handlers = TfIdfDataHandlers(self.df)
        recommended_posts = tf_idf_data_handlers.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del recommender_methods
        return recommended_posts

    def update_saved_matrix(self):
        tf_idf_data_handlers = TfIdfDataHandlers(self.df)
        fit_by_all_features_matrix = tf_idf_data_handlers.get_fit_by_feature_('all_features_preprocessed')
        save_sparse_csr(filename="models/tfidf_all_features_preprocessed.npz", array=fit_by_all_features_matrix)

    def analyze(self, slug):
        """
        Specificaly part of this module in order to perform data visualizations for the case studies
        :param slug:
        :return:
        """

        recommender_methods = RecommenderMethods()
        found_posts = []
        queried_post = recommender_methods.find_post_by_slug(slug)
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
                found_post = recommender_methods.find_post_by_slug(post['slug'])
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
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
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
        queried_post = recommender_methods.find_post_by_slug(slug)

        posts = recommender_methods.get_posts_dataframe()
        for found_post in found_posts:
            print(found_post)
            queried_post_stopwords_free = remove_stopwords([queried_post.iloc[0]['all_features_preprocessed']])[0]
            found_post_stopwords_free = remove_stopwords([found_post])[0]

            list_tmp = set(queried_post_stopwords_free) & set(
                found_post_stopwords_free)  # we don'multi_dimensional_list need to list3 to actually be a list
            # TODO:  error: Returning Any from function declared to return "SupportsLessThan". Prirotity: LOW
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
                display_top(word_count, cv)
            else:
                print("No matches found")

        print("---------------")
        print("From whole dataset:")
        word_count = cv.fit_transform(posts['all_features_preprocessed'])
        display_top(word_count, cv)

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
        for item_id in ids:
            first_vector_tfidfvectorizer = tfidf_vectors[item_id]
            df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                              columns=["tfidf"])
            print(df.sort_values(by=["tfidf"], ascending=False).head(45))

        text_titles = posts['all_features_preprocessed'].to_list()

        tfidf_visualizer = TfIdfVisualizer()
        tfidf_visualizer.prepare_for_heatmap(tfidf_vectors, text_titles, tfidf_vectorizer)
        tfidf_visualizer.plot_tfidf_heatmap()

    @staticmethod
    def get_prefilled_full_text():
        recommender_methods = RecommenderMethods()
        recommender_methods.get_posts_categories_dataframe()

    def get_similarity_matrix(self, list_of_slugs, save_to_csv=False):

        recommender_methods = RecommenderMethods()
        list_of_posts_series = []
        i = 0
        for slug in list_of_slugs:
            logging.debug("Searching for features of post %d:" % i)
            logging.debug(slug)
            i += 1
            found_post = recommender_methods.find_post_by_slug(slug)
            list_of_posts_series.append(found_post)
        self.df = pd.concat(list_of_posts_series, ignore_index=True)
        category_df = recommender_methods.get_categories_dataframe()
        self.df = self.df.merge(category_df, left_on='category_id', right_on='id')
        if 'title_y' in self.df.columns:
            self.df = self.df.rename(columns={'title_y': 'category_title'})

        fit_by_all_features_matrix = self.load_matrix()

        fit_by_title = self.get_fit_by_title()
        tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title)

        gc.collect()

        logging.debug("tuple_of_fitted_matrices")
        logging.debug(tuple_of_fitted_matrices)

        logging.debug("self.df")
        logging.debug(self.df.columns)

        try:
            tfidf_data_handlers = TfIdfDataHandlers(self.df)
            sim_matrix = tfidf_data_handlers.calculate_cosine_sim_matrix(tupple_of_fitted_matrices
                                                                         =tuple_of_fitted_matrices)
        except ValueError as e:
            print("Value error occurred:")
            print(e)
            my_file = Path("models/for_hybrid/tfidf_category_title_from_n_posts.npz")
            fit_by_all_features_matrix = self.save_sparse_matrix(for_hybrid=True)
            tfidf_data_handlers = TfIdfDataHandlers(self.df)
            fit_by_title = tfidf_data_handlers.get_fit_by_feature_('category_title')
            save_sparse_csr(filename=my_file, array=fit_by_title)
            fit_by_title = load_sparse_csr(filename=my_file)
            tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title)
            sim_matrix = tfidf_data_handlers.calculate_cosine_sim_matrix(tupple_of_fitted_matrices
                                                                         =tuple_of_fitted_matrices)
        print("sim_matrix:")
        print(sim_matrix)

        if save_to_csv is True:
            print(sim_matrix.to_csv("research/tfidf/tfidf_matrix.csv"))

        return sim_matrix

    def get_fit_by_title(self):
        my_file = Path("models/for_hybrid/tfidf_category_title.npz")
        if my_file.exists() is False:
            # category_title = category
            tf_idf_data_handlers = TfIdfDataHandlers(self.df)
            fit_by_title = tf_idf_data_handlers.get_fit_by_feature_('category_title')
            save_sparse_csr(filename=my_file, array=fit_by_title)
        else:
            fit_by_title = load_sparse_csr(filename=my_file)
        #  join feature tuples into one matrix
        return fit_by_title

    def load_matrix(self, test_call=False):
        file_path_to_load = "models/tfidf_all_features_preprocessed.npz"
        my_file = Path(file_path_to_load)
        if my_file.exists() is False:
            logging.debug('File with matrix not found in path: ' + file_path_to_load)
            fit_by_all_features_matrix = self.save_sparse_matrix()
            saved_again = True
        else:
            fit_by_all_features_matrix = load_sparse_csr(filename=my_file)
            saved_again = False

        if test_call is True:
            return fit_by_all_features_matrix, saved_again
        else:
            return fit_by_all_features_matrix


if __name__ == '__main__':
    logging.info("TF-IDF Module")
