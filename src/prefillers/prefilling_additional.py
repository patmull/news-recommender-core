import logging
import os
import time
import random

import gensim

from src.prefillers.preprocessing.bigrams_phrases import PhrasesCreator
from src.prefillers.preprocessing.cz_preprocessing import preprocess
from src.prefillers.preprocessing.keyword_extraction import SingleDocKeywordExtractor
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
# WARNING: Not tested log file:
# handler = logging.FileHandler('tests/logs/prefilling_testing_logging.logs', 'w+')
logging.debug("Testing logging from %s." % os.path.basename(__file__))


def shuffle_and_reverse(posts, random_order, reversed_order=True):
    """
    Sometimes may be beneficial to run prefilling in the random order. This is rather experimental method.

    @param posts: list of post slugs
    @param random_order: nomen omen
    @param reversed_order: nomen omen
    @return:
    """

    if reversed_order is True:
        logging.debug("Reversing list of posts...")
        posts.reverse()

    if random_order is True:
        logging.debug("Starting random_order iteration...")
        time.sleep(5)
        random.shuffle(posts)

    return posts


def get_post_columns(post):
    """
    Post columns extractor. Contains the column later used in preprocessing of the text.

    @param post: string of the slug to use

    @return: post_id, slug, article_title, article_excerpt, article_full_text, current_body_preprocess
    """
    post_id = post[0]
    slug = post[3]
    article_title = post[2]
    article_excerpt = post[4]
    article_full_text = post[20]
    current_body_preprocessed = post[21]

    return post_id, slug, article_title, article_excerpt, article_full_text, current_body_preprocessed


def extract_keywords(string_for_extraction):
    """
    Handles the call of the keyword extraction methods.

    @param string_for_extraction:
    @return:
    """

    # ** HERE WAS ALSO LINK FOR PREPROCESSING API. Abandoned for not being used.
    # keywords extraction
    logging.debug("Extracting keywords...")
    singleDocKeywordExtractor = SingleDocKeywordExtractor()
    singleDocKeywordExtractor.set_text(string_for_extraction)
    singleDocKeywordExtractor.clean_text()
    return singleDocKeywordExtractor.get_keywords_combine_all_methods(string_for_extraction=singleDocKeywordExtractor
                                                                      .text_raw)


def prepare_filling(skip_already_filled, random_order, method):
    recommender_methods = RecommenderMethods()
    if skip_already_filled is False:
        recommender_methods.database.connect()
        posts = recommender_methods.get_all_posts()
        recommender_methods.database.disconnect()
    else:
        posts = recommender_methods.get_posts_with_no_features_preprocessed(method=method)

    posts = shuffle_and_reverse(posts=posts, random_order=random_order)

    return posts


class PreFillerAdditional:

    # universal common method
    @PendingDeprecationWarning
    def fill_preprocessed(self, skip_already_filled, random_order):
        recommender_methods = RecommenderMethods()
        database_methods = DatabaseMethods()

        if skip_already_filled is False:
            posts_categories = recommender_methods.get_posts_with_not_prefilled_ngrams_text()
            posts_categories = shuffle_and_reverse(posts=posts_categories, random_order=random_order)
        else:
            posts_categories = recommender_methods.get_not_preprocessed_posts_all_features_column_and_body_preprocessed()
            posts_categories = shuffle_and_reverse(posts=posts_categories, random_order=random_order)

        for post in posts_categories:
            if len(posts_categories) < 1:
                break
            post_id = post[0]
            slug = post[3]
            article_all_features_preprocessed = post[19]
            article_full_text = post[20]
            article_body_preprocessed = post[21]
            # NOTICE: Here can be also other methods.

            logging.debug("Prefilling body preprocessd in article: " + slug)

            if skip_already_filled is True:
                if article_body_preprocessed is None or article_all_features_preprocessed is None:
                    preprocessed_text = preprocess(article_full_text)
                    logging.debug("article_full_text:")
                    logging.debug(article_full_text)
                    logging.debug("preprocessed_text:")
                    logging.debug(preprocessed_text)

                    recommender_methods = RecommenderMethods()
                    self.start_preprocessed_columns_prefilling(article_full_text=preprocessed_text,
                                                               post_id=post_id)
                else:
                    logging.debug("Skipping.")
            else:
                self.start_preprocessed_columns_prefilling(article_full_text, post_id)

    def fill_body_preprocessed(self, skip_already_filled, random_order):
        posts = prepare_filling(skip_already_filled, random_order, method='body_preprocessed')
        for post in posts:
            if len(posts) < 1:
                break

            # noinspection PyPep8
            post_id, slug, article_title, article_excerpt, article_full_text, current_body_preprocessed \
                = get_post_columns(post)

            logging.debug("Prefilling body preprocessed in article: " + slug)

            if skip_already_filled is True:
                if current_body_preprocessed is None:
                    preprocessed_text = preprocess(article_full_text)
                    logging.debug("article_full_text:")
                    logging.debug(article_full_text)
                    logging.debug("preprocessed_text:")
                    logging.debug(preprocessed_text)

                    recommender_methods = RecommenderMethods()
                    recommender_methods.insert_preprocessed_body(preprocessed_body=preprocessed_text,
                                                                 article_id=post_id)

                else:
                    logging.debug("Skipping.")
            else:
                self.start_preprocessed_columns_prefilling(article_full_text, post_id)

    def fill_keywords(self, skip_already_filled, random_order):
        recommender_methods = RecommenderMethods()
        if skip_already_filled is False:
            posts = recommender_methods.get_all_posts()
        else:
            posts = recommender_methods.get_posts_with_no_features_preprocessed(method='keywords')

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
            features = str(article_title or '') + ' ' + str(article_excerpt or '') + ' ' + str(article_full_text or '')

            logging.debug("Prefilling body preprocessd in article: " + slug)

            recommender_methods = RecommenderMethods()
            if skip_already_filled is True:
                preprocessed_keywords = extract_keywords(string_for_extraction=features)
                logging.debug("article_full_text:")
                logging.debug(features)
                logging.debug("preprocessed_keywords:")
                logging.debug(preprocessed_keywords)

                recommender_methods.insert_keywords(keyword_all_types_splitted=preprocessed_keywords, article_id=post_id)
            else:
                preprocessed_keywords = extract_keywords(string_for_extraction=features)
                logging.debug("article_full_text")
                logging.debug(features)
                logging.debug("preprocessed_keywords:")
                logging.debug(preprocessed_keywords)

                recommender_methods.insert_keywords(keyword_all_types_splitted=preprocessed_keywords, article_id=post_id)

                number_of_inserted_rows += 1
                if number_of_inserted_rows > 20:
                    logging.debug("Refreshing list of posts for finding only not prefilled posts.")
                    self.fill_keywords(skip_already_filled=True,
                                       random_order=random_order)

    def fill_all_features_preprocessed(self, skip_already_filled, random_order):
        posts = prepare_filling(skip_already_filled=skip_already_filled, random_order=random_order,
                                method='all_features_preprocessed')

        for post in posts:
            if len(posts) < 1:
                break
            # TODO: Category should be there too
            # noinspection PyPep8
            post_id, slug, article_title, article_excerpt, article_full_text, current_body_preprocessed\
                = get_post_columns(post)
            current_all_features_preprocessed = post[19]

            logging.debug("post:")
            logging.debug(post)

            logging.debug("Prefilling body preprocessd in article: " + slug)

            recommender_methods = RecommenderMethods()
            if skip_already_filled is True:
                if current_all_features_preprocessed is None:
                    preprocessed_text = preprocess(article_full_text)
                    logging.debug("article_full_text:")
                    logging.debug(article_full_text)
                    logging.debug("preprocessed_text:")
                    logging.debug(preprocessed_text)
                    recommender_methods.insert_all_features_preprocessed_combined(preprocessed_text, post_id)
                else:
                    logging.debug("Skipping.")
            else:
                self.start_preprocessed_columns_prefilling(article_full_text=article_full_text, post_id=post_id)

    def start_preprocessed_columns_prefilling(self, article_full_text, post_id):

        preprocessed_text = preprocess(article_full_text)
        logging.debug("article_full_text")
        logging.debug(article_full_text)
        logging.debug("preprocessed_text:")
        logging.debug(preprocessed_text)

        recommender_methods = RecommenderMethods()
        recommender_methods.insert_preprocessed_body(preprocessed_body=preprocessed_text, article_id=post_id)
        number_of_inserted_rows = 0

        number_of_inserted_rows += 1

    def fill_ngrams_for_all_posts(self, skip_already_filled, random_order, full_text):
        recommender_methods = RecommenderMethods()
        logging.debug("Beginning prefiling of bigrams, variant full_text=" + str(full_text))
        if skip_already_filled is False:
            posts = recommender_methods.get_all_posts()
        else:
            posts = recommender_methods.get_posts_with_not_prefilled_ngrams_text(full_text)

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
                if short_text_preprocessed is not None and current_body_preprocessed is not None:
                    input_text = short_text_preprocessed + " " + current_body_preprocessed
                else:
                    # TODO: Almost already fixed this. Just review the logic.
                    if current_body_preprocessed is None and short_text_preprocessed is not None:
                        # Using only the short text (= all_features_preprocessed column)
                        input_text = short_text_preprocessed
                        logging.warning("body_preprocessed is None. Trigrams are created only from short text."
                                        "Prefill the body_preprocessed column first to use it for trigrams.")
                    elif current_body_preprocessed is None and short_text_preprocessed is None:
                        raise ValueError("Either all_features_preprocessed or "
                                         "body_preprocessed needs to be filled in,")

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

                    recommender_methods.insert_phrases_text(bigram_text=preprocessed_text, article_id=post_id,
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

                recommender_methods.insert_phrases_text(bigram_text=preprocessed_text, article_id=post_id,
                                                        full_text=full_text)

                number_of_inserted_rows += 1
                if number_of_inserted_rows > 300:
                    logging.debug("Refreshing list of posts for finding only not prefilled posts.")
                    self.fill_ngrams_for_all_posts(skip_already_filled=skip_already_filled, random_order=random_order,
                                                   full_text=full_text)
