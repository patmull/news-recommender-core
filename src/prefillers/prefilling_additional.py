import logging
import os
import time
import random

import gensim

from src.prefillers.preprocessing.bigrams_phrases import PhrasesCreator
from src.prefillers.preprocessing.czech_preprocessing import preprocess
from src.prefillers.preprocessing.keyword_extraction import SingleDocKeywordExtractor
from src.recommender_core.data_handling.data_queries import RecommenderMethods

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

SKIP_RECORD_MESSAGE = "Skipping."
PREFILLING_PREPROCESSING_MESSAGE = "Prefilling body preprocessed in article: "

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
    post_id = post['post-id']
    slug = post['slug']
    article_full_text = post['full_text']

    return post_id, slug, article_full_text


def extract_keywords(string_for_extraction):
    """
    Handles the call of the keyword extraction methods.
    @param string_for_extraction:
    @return:
    """

    # ** HERE WAS ALSO LINK FOR PREPROCESSING API. Abandoned for not being used.
    # keywords extraction
    logging.debug("Extracting keywords...")
    single_doc_keyword_extractor = SingleDocKeywordExtractor()
    single_doc_keyword_extractor.set_text(string_for_extraction)
    single_doc_keyword_extractor.clean_text()
    return single_doc_keyword_extractor.get_keywords_combine_all_methods(
        string_for_extraction=single_doc_keyword_extractor
        .text_raw)


def prepare_filling(skip_already_filled, random_order, method):
    """
    Handles the data loading and shuffling/reversing if chosen for the
    prefilling methods.

    @param skip_already_filled:
    @param random_order:
    @param method:
    @return:
    """
    recommender_methods = RecommenderMethods()
    if skip_already_filled is False:
        recommender_methods.database.connect()
        posts = recommender_methods.get_all_posts()
        recommender_methods.database.disconnect()
    else:
        posts = recommender_methods.get_posts_with_no_features_preprocessed(method=method)

    posts = shuffle_and_reverse(posts=posts, random_order=random_order)

    return posts


def start_preprocessed_columns_prefilling(article_full_text, post_id):
    """
    Handles the 'body_preprocessed' data loading and actual pre-processing.

    @param article_full_text:
    @param post_id:
    @return:
    """

    preprocessed_text = preprocess(article_full_text)

    recommender_methods = RecommenderMethods()
    recommender_methods.insert_preprocessed_body(preprocessed_body=preprocessed_text, article_id=post_id)


class PreFillerAdditional:

    # universal common method
    @PendingDeprecationWarning
    def fill_preprocessed(self, skip_already_filled, random_order):
        recommender_methods = RecommenderMethods()

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

            logging.debug(PREFILLING_PREPROCESSING_MESSAGE + slug)

            if skip_already_filled is True:
                if article_body_preprocessed is None or article_all_features_preprocessed is None:
                    preprocessed_text = preprocess(article_full_text)
                    start_preprocessed_columns_prefilling(article_full_text=preprocessed_text,
                                                          post_id=post_id)
                else:
                    logging.debug(SKIP_RECORD_MESSAGE)
            else:
                start_preprocessed_columns_prefilling(article_full_text, post_id)

    def fill_body_preprocessed(self, skip_already_filled, random_order):
        posts = prepare_filling(skip_already_filled, random_order, method='body_preprocessed')
        for post in posts:
            if len(posts) < 1:
                break

            (post_id, slug, article_full_text) = get_post_columns(post)

            logging.debug(PREFILLING_PREPROCESSING_MESSAGE + slug)

            if skip_already_filled is True:
                preprocessed_text = preprocess(article_full_text)

                recommender_methods = RecommenderMethods()
                recommender_methods.insert_preprocessed_body(preprocessed_body=preprocessed_text,
                                                             article_id=post_id)

            else:
                start_preprocessed_columns_prefilling(article_full_text, post_id)

    def fill_keywords(self, skip_already_filled, random_order):
        """
        Hanldes the keywords extraction methods and calls the inserting methods.

        @param skip_already_filled:
        @param random_order:
        @return:
        """
        recommender_methods = RecommenderMethods()
        if skip_already_filled is False:
            posts = recommender_methods.get_all_posts()
        else:
            posts = recommender_methods.get_posts_with_no_features_preprocessed(method='keywords')

        number_of_inserted_rows = 0

        if random_order is True:
            logging.debug("Starting random iteration...")
            time.sleep(5)
            random.shuffle(posts)

        for post in posts:
            if len(posts) < 1:
                break
            post_id = post['id']
            slug = post['slug']
            article_title = post['title']
            article_excerpt = post['excerpt']
            article_full_text = post['full_text']
            features = str(article_title or '') + ' ' + str(article_excerpt or '') + ' ' + str(article_full_text or '')

            logging.debug(PREFILLING_PREPROCESSING_MESSAGE + slug)

            recommender_methods = RecommenderMethods()
            if skip_already_filled is True:
                preprocessed_keywords = extract_keywords(string_for_extraction=features)
                logging.debug("article_full_text:")
                logging.debug(features)
                logging.debug("preprocessed_keywords:")
                logging.debug(preprocessed_keywords)

                recommender_methods.insert_keywords(keyword_all_types_split=preprocessed_keywords, article_id=post_id)
            else:
                preprocessed_keywords = extract_keywords(string_for_extraction=features)
                logging.debug("article_full_text")
                logging.debug(features)
                logging.debug("preprocessed_keywords:")
                logging.debug(preprocessed_keywords)

                recommender_methods.insert_keywords(keyword_all_types_split=preprocessed_keywords, article_id=post_id)

                number_of_inserted_rows += 1
                if number_of_inserted_rows > 20:
                    logging.debug("Refreshing list of posts for finding only not prefilled posts.")
                    self.fill_keywords(skip_already_filled=True,
                                       random_order=random_order)

    def fill_all_features_preprocessed(self, skip_already_filled, random_order):
        """
        Handles the data loaders and inserts fot the 'all_features_preprocessed' method.

        @param skip_already_filled:
        @param random_order:
        @return:
        """
        posts = prepare_filling(skip_already_filled=skip_already_filled, random_order=random_order,
                                method='all_features_preprocessed')

        for post in posts:
            if len(posts) < 1:
                break
            # noinspection PyPep8
            post_id, slug, article_full_text = get_post_columns(post)
            current_all_features_preprocessed = post['all_features_preprocessed']

            logging.debug("post:")
            logging.debug(post)

            logging.debug(PREFILLING_PREPROCESSING_MESSAGE + slug)

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
                    logging.debug(SKIP_RECORD_MESSAGE)
            else:
                start_preprocessed_columns_prefilling(article_full_text=article_full_text, post_id=post_id)

    def fill_ngrams_for_all_posts(self, skip_already_filled, random_order, full_text):
        """
        Bigrams and trigrams data loading, data handling, extracting methods call and inserts.

        @param skip_already_filled:
        @param random_order:
        @param full_text:
        @return:
        """
        global input_text
        recommender_methods = RecommenderMethods()
        logging.debug("Beginning prefiling of bigrams, variant full_text=" + str(full_text))
        if skip_already_filled is False:
            posts = recommender_methods.get_all_posts()
        else:
            posts = recommender_methods.get_posts_with_not_prefilled_ngrams_text(full_text)

        if len(posts) == 0:
            logging.debug("All posts full_text=" + str(full_text) + " prefilled. Skipping.")
            return

        if random_order is True:
            logging.debug("Starting random iteration...")
            time.sleep(5)
            random.shuffle(posts)

        iterate_and_fill_ngrams(posts, full_text)


def load_data_for_ngrams_filling(post, full_text=False):
    post_id = post['id']
    slug = post['slug']
    short_text_preprocessed = post['short_text']
    article_full_text = post['full_text']
    current_body_preprocessed = post['body_preprocessed']
    current_ngrams = post['trigrans'] if full_text else post['bigrams']

    if full_text and isinstance(article_full_text, str):
        if short_text_preprocessed and current_body_preprocessed:
            _input_text = f"{short_text_preprocessed} {current_body_preprocessed}"
        else:
            if not current_body_preprocessed and short_text_preprocessed:
                _input_text = short_text_preprocessed
                logging.warning("body_preprocessed is None. Trigrams are created only from short text. "
                                "Prefill the body_preprocessed column first to use it for trigrams.")
            elif not current_body_preprocessed and not short_text_preprocessed:
                raise ValueError("Either all_features_preprocessed or body_preprocessed needs to be filled in")
    else:
        _input_text = short_text_preprocessed

    if not _input_text:
        raise ValueError("input_text is None. This should not happen.")

    loaded_data = {
        'post_id': post_id,
        'slug': slug,
        'input_text': _input_text,
        'current_ngrams': current_ngrams
    }

    return loaded_data


def iterate_and_fill_ngrams(posts, full_text=False, skip_already_filled=True, random_order=True):
    phrases_creator = PhrasesCreator()
    recommender_methods = RecommenderMethods()
    prefiller_additonal = PreFillerAdditional()
    number_of_inserted_rows = 0
    for post in posts:
        loaded_post_data = load_data_for_ngrams_filling(post, full_text)

        if len(posts) < 1:
            break

        logging.debug(PREFILLING_PREPROCESSING_MESSAGE + loaded_post_data['slug'])

        if skip_already_filled is True:
            if loaded_post_data['current_ngrams'] is None:
                input_text_split = input_text.split()
                preprocessed_text = gensim.utils. \
                    deaccent(preprocess(phrases_creator.create_trigrams(input_text_split)))

                recommender_methods.insert_phrases_text(bigram_text=preprocessed_text,
                                                        article_id=loaded_post_data['post_id'],
                                                        full_text=full_text)
            else:
                logging.debug(SKIP_RECORD_MESSAGE)
        else:
            input_text_split = input_text.split()
            preprocessed_text = gensim.utils.deaccent(
                preprocess(phrases_creator.create_trigrams(input_text_split)))

            recommender_methods.insert_phrases_text(bigram_text=preprocessed_text,
                                                    article_id=loaded_post_data['post_id'],
                                                    full_text=full_text)

            number_of_inserted_rows += 1
            if number_of_inserted_rows > 300:
                logging.debug("Refreshing list of posts for finding only not prefilled posts.")
                prefiller_additonal.fill_ngrams_for_all_posts(skip_already_filled=skip_already_filled,
                                                              random_order=random_order, full_text=full_text)
