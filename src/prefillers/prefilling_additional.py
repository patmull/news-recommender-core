import logging
import time
import random

import gensim

from src.prefillers.preprocessing.bigrams_phrases import PhrasesCreator
from src.prefillers.preprocessing.cz_preprocessing import CzPreprocess, preprocess
from src.prefillers.preprocessing.keyword_extraction import SingleDocKeywordExtractor
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods


def shuffle_and_reverse(posts, random_order, reversed_order=True):

    if reversed_order is True:
        logging.debug("Reversing list of posts...")
        posts.reverse()

    if random_order is True:
        logging.debug("Starting random_order iteration...")
        time.sleep(5)
        random.shuffle(posts)


def get_post_columns(post):
    post_id = post[0]
    slug = post[3]
    article_title = post[2]
    article_excerpt = post[4]
    article_full_text = post[20]
    current_body_preprocessed = post[21]

    return post_id, slug, article_title, article_excerpt, article_full_text, current_body_preprocessed


def extract_keywords(string_for_extraction):

    # ** HERE WAS ALSO LINK FOR PREPROCESSING API. Abandoned for not being used.
    # keywords extraction
    logging.debug("Extracting keywords...")
    singleDocKeywordExtractor = SingleDocKeywordExtractor()
    singleDocKeywordExtractor.set_text(string_for_extraction)
    singleDocKeywordExtractor.clean_text()
    return singleDocKeywordExtractor.get_keywords_combine_all_methods(string_for_extraction=singleDocKeywordExtractor
                                                                      .text_raw)


def prepare_filling(skip_already_filled, random_order):
    database = DatabaseMethods()
    recommender_methods = RecommenderMethods()
    if skip_already_filled is False:
        recommender_methods.database.connect()
        posts = recommender_methods.get_all_posts()
        recommender_methods.database.disconnect()
    else:
        database.connect()
        posts = database.get_not_preprocessed_posts()
        database.disconnect()

    number_of_inserted_rows = 0

    shuffle_and_reverse(posts=posts, random_order=random_order)

    cz_lemma = CzPreprocess()

    return posts, database, cz_lemma, number_of_inserted_rows


class PreFillerAdditional:

    # universal common method
    def fill_preprocessed(self, skip_already_filled, random_order, db="pgsql"):
        recommender_methods = RecommenderMethods()
        database_methods = DatabaseMethods()

        if skip_already_filled is False:
            recommender_methods.database.connect()
            posts = recommender_methods.get_all_posts()
            recommender_methods.database.disconnect()
            posts_categories = database_methods.join_posts_ratings_categories()
            shuffle_and_reverse(posts=posts, random_order=random_order)
        else:
            posts_categories = database_methods.get_not_preprocessed_posts()
            shuffle_and_reverse(posts=posts_categories, random_order=random_order)

        number_of_inserted_rows = 0

        cz_lemma = CzPreprocess()

        for post in posts_categories:
            if len(posts_categories) < 1:
                break
            post_id = post[0]
            slug = post[3]
            """
            article_category = post[2]
            article_title = post[1]
            article_excerpt = post[3]
            article_body = post[4]
            article_keywords = post[5]
            article_all_features_preprocessed = post[6]
            """
            article_full_text = post[7]
            article_body_preprocessed = post[8]

            logging.debug("Prefilling body preprocessd in article: " + slug)

            if skip_already_filled is True:
                if article_body_preprocessed is None:
                    if db == "redis":
                        database_methods.insert_preprocessed_body(preprocessed_body=article_body_preprocessed,
                                                                  article_id=post_id)
                    elif db == "pgsql":
                        preprocessed_text = preprocess(article_full_text)
                        logging.debug("article_full_text:")
                        logging.debug(article_full_text)
                        logging.debug("preprocessed_text:")
                        logging.debug(preprocessed_text)

                        database_methods.insert_preprocessed_body(preprocessed_body=preprocessed_text,
                                                                  article_id=post_id)
                    else:
                        raise Exception
                else:
                    logging.debug("Skipping.")
            else:
                self.start_preprocessed_features_prefilling(db, cz_lemma, article_full_text, database_methods, post_id,
                                                            number_of_inserted_rows, random_order)

    def fill_body_preprocessed(self, skip_already_filled, random_order, db="pgsql"):
        posts, database, cz_lemma, number_of_inserted_rows = prepare_filling(skip_already_filled, random_order)

        for post in posts:
            if len(posts) < 1:
                break

            # noinspection PyPep8
            post_id, slug, article_title, article_excerpt, article_full_text, current_body_preprocessed \
                = get_post_columns(post)

            logging.debug("Prefilling body preprocessd in article: " + slug)

            if skip_already_filled is True:
                if current_body_preprocessed is None:
                    if db == "pgsql":
                        preprocessed_text = preprocess(article_full_text)
                        logging.debug("article_full_text:")
                        logging.debug(article_full_text)
                        logging.debug("preprocessed_text:")
                        logging.debug(preprocessed_text)

                        database.connect()
                        database.insert_preprocessed_body(preprocessed_body=preprocessed_text, article_id=post_id)
                        database.disconnect()
                    else:
                        raise NotImplementedError
                else:
                    logging.debug("Skipping.")
            else:
                self.start_preprocessed_features_prefilling(db, cz_lemma, article_full_text,
                                                            database, post_id,
                                                            number_of_inserted_rows, random_order)

    def fill_keywords(self, skip_already_filled, random_order):
        database = DatabaseMethods()
        database.connect()
        if skip_already_filled is False:
            posts = database.get_all_posts()
        else:
            posts = database.get_posts_with_no_keywords()

        number_of_inserted_rows = 0

        if reversed is True:  # type: ignore
            logging.debug("Reversing list of posts...")  # type: ignore
            posts.reverse()

        if random_order is True:
            logging.debug("Starting random iteration...")
            time.sleep(5)
            random.shuffle(posts)

        for post in posts:
            if len(posts) < 1:
                break
            post_id = post[0]
            slug = post[3]
            article_title = post[2]
            article_excerpt = post[4]
            article_full_text = post[20]
            features = article_title + ' ' + article_excerpt + ' ' + article_full_text

            logging.debug("Prefilling body preprocessd in article: " + slug)

            if skip_already_filled is True:
                preprocessed_keywords = extract_keywords(string_for_extraction=features)
                logging.debug("article_full_text:")
                logging.debug(features)
                logging.debug("preprocessed_keywords:")
                logging.debug(preprocessed_keywords)

                database.insert_keywords(keyword_all_types_splitted=preprocessed_keywords, article_id=post_id)
            else:
                preprocessed_keywords = extract_keywords(string_for_extraction=features)
                logging.debug("article_full_text")
                logging.debug(features)
                logging.debug("preprocessed_keywords:")
                logging.debug(preprocessed_keywords)

                database.insert_keywords(keyword_all_types_splitted=preprocessed_keywords, article_id=post_id)

                number_of_inserted_rows += 1
                if number_of_inserted_rows > 20:
                    logging.debug("Refreshing list of posts for finding only not prefilled posts.")
                    self.fill_keywords(skip_already_filled=True,
                                       random_order=random_order)

    def fill_all_features_preprocessed(self, skip_already_filled, random_order, db="pgsql"):
        posts, database, cz_lemma, number_of_inserted_rows = prepare_filling(skip_already_filled=skip_already_filled,
                                                                             random_order=random_order)

        for post in posts:
            if len(posts) < 1:
                break
            # TODO: Category should be there too
            # noinspection PyPep8
            post_id, slug, article_title, article_excerpt, article_full_text, current_body_preprocessed\
                = get_post_columns(post)

            logging.debug("Prefilling body preprocessd in article: " + slug)

            if skip_already_filled is True:
                if current_body_preprocessed is None:
                    if db == "pgsql":
                        preprocessed_text = preprocess(article_full_text)
                        logging.debug("article_full_text:")
                        logging.debug(article_full_text)
                        logging.debug("preprocessed_text:")
                        logging.debug(preprocessed_text)
                        database.connect()
                        database.insert_preprocessed_combined(preprocessed_text, post_id)
                        database.disconnect()
                    else:
                        raise NotImplementedError
                else:
                    logging.debug("Skipping.")
            else:
                self.start_preprocessed_features_prefilling(self, db, cz_lemma, article_full_text,
                                                            database, post_id,
                                                            number_of_inserted_rows)

    def start_preprocessed_features_prefilling(self, db, cz_lemma, article_full_text, database, post_id,
                                               number_of_inserted_rows, random_order):
        if db == "pgsql":
            preprocessed_text = cz_lemma.preprocess(article_full_text)
            logging.debug("article_full_text")
            logging.debug(article_full_text)
            logging.debug("preprocessed_text:")
            logging.debug(preprocessed_text)

            database.insert_preprocessed_body(preprocessed_body=preprocessed_text, article_id=post_id)
        else:
            raise Exception
        number_of_inserted_rows += 1
        if number_of_inserted_rows > 20:
            logging.debug("Refreshing list of posts for finding only not prefilled posts.")
            self.fill_all_features_preprocessed(db=db, skip_already_filled=True,
                                                random_order=random_order)

    def fill_ngrams_for_all_posts(self, skip_already_filled, random_order, full_text):
        database = DatabaseMethods()
        logging.debug("Beginning prefiling of bigrams, variant full_text=" + str(full_text))
        if skip_already_filled is False:
            posts = database.get_all_posts()
        else:
            posts = database.get_posts_with_not_prefilled_ngrams_text(full_text)

        if len(posts) == 0:
            logging.debug("All posts full_text=" + str(full_text) + " prefilled. Skipping.")
            return

        number_of_inserted_rows = 0

        if random_order is True:
            logging.debug("Starting random iteration...")
            time.sleep(5)
            random.shuffle(posts)

        phrases_creator = PhrasesCreator()

        for post in posts:
            if len(posts) < 1:
                break
            post_id = post[0]
            slug = post[3]
            short_text_preprocessed = post[19]  # == all_features_preprocessed
            article_full_text = post[20]
            current_body_preprocessed = post[21]
            if full_text is False:
                current_bigrams = post[31]
            else:
                current_bigrams = post[32]
            # TODO: Category should be there too
            # second part: checking whether current body preprocessed is not none
            if full_text is True and isinstance(article_full_text, str):
                input_text = short_text_preprocessed + " " + current_body_preprocessed
            else:
                input_text = short_text_preprocessed

            logging.debug("Prefilling body preprocessd in article: " + slug)
            if skip_already_filled is True:
                if current_bigrams is None:
                    input_text_splitted = input_text.split()
                    preprocessed_text = gensim.utils. \
                        deaccent(preprocess(phrases_creator.create_trigrams(input_text_splitted)))
                    logging.debug("input_text:")
                    logging.debug(input_text)
                    logging.debug("preprocessed_text double preprocessed:")
                    logging.debug(preprocessed_text)

                    database.insert_phrases_text(bigram_text=preprocessed_text, article_id=post_id,
                                                 full_text=full_text)
                else:
                    logging.debug("Skipping.")
            else:
                input_text_splitted = input_text.split()
                preprocessed_text = gensim.utils.deaccent(
                    preprocess(phrases_creator.create_trigrams(input_text_splitted)))
                logging.debug("input_text:")
                logging.debug(input_text)
                logging.debug("preprocessed_text:")
                logging.debug(preprocessed_text)

                database.insert_phrases_text(bigram_text=preprocessed_text, article_id=post_id,
                                             full_text=full_text)

                number_of_inserted_rows += 1
                if number_of_inserted_rows > 300:
                    logging.debug("Refreshing list of posts for finding only not prefilled posts.")
                    self.fill_ngrams_for_all_posts(skip_already_filled=skip_already_filled, random_order=random_order,
                                                   full_text=full_text)
