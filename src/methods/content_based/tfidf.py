import gc
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

import config.trials_counter
from src.data_handling.data_manipulation import DatabaseMethods
from src.data_handling.data_queries import RecommenderMethods, TfIdfDataHandlers

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from terms_frequencies module.")

CATEGORY_FILENAME = "models/tfidf_category_title.npz"

EMPTY_INPUT_STRING_MSG = "Entered input_string is empty."


def get_cleaned_text(row):
    return row


def get_prefilled_full_text():
    recommender_methods = RecommenderMethods()
    recommender_methods.get_posts_categories_dataframe()


def convert_to_json_one_row(key, value):
    list_for_json = []
    dict_for_json = {key: value}
    list_for_json.append(dict_for_json)
    return list_for_json


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def save_tfidf_vectorizer(vector, path):
    pickle.dump(vector, open(path, "wb"))


def load_tfidf_vectorizer(path=Path("precalc_vectors/all_features_preprocessed_vectorizer.pickle")):
    vectorizer = pickle.load(open(path, "rb"))
    return vectorizer


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def print_top(word_count):
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count)


def most_similar_by_keywords(tuple_of_fitted_matrices, keywords,
                             cached_file_path, number_of_recommended_posts=20):
    tfidf_data_handlers = TfIdfDataHandlers()
    try:
        post_recommendations = tfidf_data_handlers \
            .most_similar_by_keywords(keywords, tuple_of_fitted_matrices,
                                      number_of_recommended_posts=number_of_recommended_posts)
    except ValueError as e:
        # Value error had occurred while computing the most similar posts by keywords.
        logging.warning(F"Value error had occurred while computing most similar posts by keywords: {e}")
        logging.info("Trying to solve Value error by updating the terms_frequencies vectorizer file")
        fit_by_all_features_preprocessed = tfidf_data_handlers.get_fit_by_feature_('all_features_preprocessed')
        save_tfidf_vectorizer(fit_by_all_features_preprocessed, path=cached_file_path)
        post_recommendations = tfidf_data_handlers \
            .most_similar_by_keywords(keywords, tuple_of_fitted_matrices,
                                      number_of_recommended_posts=number_of_recommended_posts)

    return post_recommendations


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

    def keyword_based_comparison(self, keywords, number_of_recommended_posts=20, all_posts=False):
        if type(keywords) is not str:
            raise ValueError("Entered keywords must be a input_string.")
        else:
            if keywords == "":
                raise ValueError(EMPTY_INPUT_STRING_MSG)

        if keywords == "":
            return {}

        keywords_split_1 = keywords.split(" ")  # splitting sentence into list of keywords by space

        # creating _dictionary of words
        word_dict_a = dict.fromkeys(keywords_split_1, 0)
        for word in keywords_split_1:
            word_dict_a[word] += 1

        recommender_methods = RecommenderMethods()
        self.df = recommender_methods.get_posts_categories_dataframe()
        tfidf_data_handlers = TfIdfDataHandlers(self.df)

        cached_file_path = Path("precalc_vectors/all_features_preprocessed_vectorizer.pickle")
        fit_by_all_features_preprocessed = load_tfidf_vectorizer(cached_file_path)
        fit_by_keywords_matrix = tfidf_data_handlers.get_fit_by_feature_('keywords')
        tuple_of_fitted_matrices = (fit_by_all_features_preprocessed, fit_by_keywords_matrix)
        del fit_by_keywords_matrix
        gc.collect()
        if all_posts is False:
            post_recommendations = tfidf_data_handlers.most_similar_by_keywords(keywords,
                                                                                tuple_of_fitted_matrices,
                                                                                number_of_recommended_posts)
        else:
            if self.posts_df.index is not None:
                try:
                    post_recommendations = tfidf_data_handlers \
                        .most_similar_by_keywords(keywords, tuple_of_fitted_matrices,
                                                  number_of_recommended_posts=len(self.posts_df.index))
                except ValueError as e:
                    (logging
                     .warning(f"Value error had occurred while computing most similar posts by keywords: {e}")
                     )
                    logging.info("Trying to solve Value error by updating the terms_frequencies vectorizer file")
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

    def set_tfidf_vectorizer_combine_features(self):
        """
        See more:
        # https://datascience.stackexchange.com/questions/18581/same-tf-idf-vectorizer-for-2-data-inputs

        """
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

    def save_sparse_matrix(self, for_hybrid=False):
        logging.info("Loading sparse matrix.")

        if for_hybrid is True:
            path = Path("models/for_hybrid/tfidf_all_features_preprocessed.npz")
        else:
            path = Path("models/tfidf_all_features_preprocessed.npz")

        if len(self.df.index) == 0:
            logging.debug("Dataframe is Empty. Getting posts from DB.")
            recommender_methods = RecommenderMethods()
            self.df = recommender_methods.get_posts_categories_dataframe(from_cache=True)
        logging.debug("self.df 1")
        logging.debug(self.df)
        tfidf_data_handlers = TfIdfDataHandlers(self.df)
        logging.debug("self.df 2")
        logging.debug(tfidf_data_handlers.df)
        fit_by_all_features_matrix = tfidf_data_handlers.get_fit_by_feature_('all_features_preprocessed')
        logging.info("Saving sparse matrix into file...")
        save_sparse_csr(filename=path, array=fit_by_all_features_matrix)
        return fit_by_all_features_matrix

    def recommend_posts_by_all_features_preprocessed(self, searched_slug, num_of_recommendations=20):
        """
        This method differs from Fresh API module's method.
        This method is more optimized for "offline" use among prefillers.
        """
        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a input_string.")
        else:
            if searched_slug == "":
                raise ValueError(EMPTY_INPUT_STRING_MSG)

        recommender_methods = RecommenderMethods()
        self.df = recommender_methods.get_posts_categories_dataframe()

        fit_by_all_features_matrix = self.load_matrix()

        fit_by_title = self.get_fit_by_title()
        tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title)

        gc.collect()

        if searched_slug not in self.df['slug'].to_list():
            # Slug does not appear in dataframe. Refreshing dataframe of posts.
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
            save_sparse_csr(filename=CATEGORY_FILENAME, array=fit_by_title)
            fit_by_title = load_sparse_csr(filename=CATEGORY_FILENAME)
            tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title)
            recommended_post_recommendations = tfidf_data_handlers \
                .recommend_by_more_features(slug=searched_slug, tupple_of_fitted_matrices=tuple_of_fitted_matrices,
                                            num_of_recommendations=num_of_recommendations)

        del recommender_methods, tuple_of_fitted_matrices
        return recommended_post_recommendations

    def recommend_posts_by_all_features_preprocessed_with_full_text(self, searched_slug, posts_from_cache=True,
                                                                    tf_idf_data_handlers=None,
                                                                    fit_by_all_features_matrix=None,
                                                                    fit_by_title=None,
                                                                    fit_by_full_text=None):

        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a input_string.")
        else:
            if searched_slug == "":
                raise ValueError(EMPTY_INPUT_STRING_MSG)

        recommender_methods = RecommenderMethods()
        self.df = recommender_methods.get_posts_categories_dataframe(from_cache=posts_from_cache)
        gc.collect()

        if searched_slug not in self.df['slug'].to_list():
            if config.trials_counter.NUM_OF_TRIALS < 1:
                # Slug does not appear in dataframe. Refreshing datafreme of posts.
                recommender_methods = RecommenderMethods()
                recommender_methods.get_posts_dataframe(force_update=True)
                self.df = recommender_methods.get_posts_categories_dataframe(from_cache=True)

                config.trials_counter.NUM_OF_TRIALS += 1

                self.recommend_posts_by_all_features_preprocessed_with_full_text(searched_slug, posts_from_cache)
            else:
                config.trials_counter.NUM_OF_TRIALS = 0
                raise ValueError("searched_slug not in dataframe. Tried to deal with this by updating posts_categories "
                                 "df but didn't helped")
        # replacing None values with empty strings
        recommender_methods.df['full_text'] = recommender_methods.df['full_text'].replace([None], '')

        if tf_idf_data_handlers is None:
            tf_idf_data_handlers = TfIdfDataHandlers(self.df)
        if fit_by_all_features_matrix is None:
            fit_by_all_features_matrix = tf_idf_data_handlers.get_fit_by_feature_('all_features_preprocessed')
        if fit_by_title is None:
            fit_by_title = tf_idf_data_handlers.get_fit_by_feature_('category_title')
        if fit_by_full_text is None:
            fit_by_full_text = tf_idf_data_handlers.get_fit_by_feature_('full_text')

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_title, fit_by_all_features_matrix, fit_by_full_text)
        del fit_by_title
        del fit_by_all_features_matrix
        del fit_by_full_text
        gc.collect()

        recommended_post_recommendations = tf_idf_data_handlers \
            .recommend_by_more_features(searched_slug, tuple_of_fitted_matrices)
        del recommender_methods
        return recommended_post_recommendations

    def get_similarity_matrix(self, list_of_slugs, for_hybrid=False):

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
            logging.error(f"Value error occurred: {e}")
            if for_hybrid:
                my_file = Path("models/for_hybrid/tfidf_category_title_from_n_posts.npz")
            else:
                my_file = Path("models/tfidf_category_title_from_n_posts.npz")
            fit_by_all_features_matrix = self.save_sparse_matrix(for_hybrid)
            tfidf_data_handlers = TfIdfDataHandlers(self.df)
            fit_by_title = tfidf_data_handlers.get_fit_by_feature_('category_title')
            save_sparse_csr(filename=my_file, array=fit_by_title)
            fit_by_title = load_sparse_csr(filename=my_file)
            tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title)
            sim_matrix = tfidf_data_handlers.calculate_cosine_sim_matrix(tupple_of_fitted_matrices
                                                                         =tuple_of_fitted_matrices)

        return sim_matrix

    def get_fit_by_title(self, for_hybrid=False):
        if for_hybrid:
            my_file = Path("models/for_hybrid/tfidf_category_title.npz")
        else:
            my_file = Path(CATEGORY_FILENAME)
        if my_file.exists() is False:
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
