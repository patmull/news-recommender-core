import logging
import pickle
import random
import time

import spacy_sentence_bert

from src.recommender_core.data_handling.model_methods.user_methods import UserMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    Classifier
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from data?manipulation.")


def predict_ratings_for_user_store_to_redis(user_id, force_retrain=False):
    """
    Predictign relevant articles from user base on classifier model (SVC / Random Forrest methods) using BERT
    multi-lingual model.
    @param user_id:
    @param force_retrain:
    @return:
    """
    classifier = Classifier()
    print("Loading BERT multilingual model...")
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    try:
        classifier.predict_relevance_for_user(user_id=user_id, relevance_by='thumbs', bert_model=bert,
                                              force_retraining=force_retrain)
    except ValueError as ve:
        print("Value error occurred when trying to get relevant thumbs for user. Skipping "
              "this user.")
        logging.warning(ve)
        logging.warning("This is probably caused by insufficient number of examples for thumbs."
                        "User also needs to rate some posts both by thumbs up and down in order "
                        "to provide sufficient number of examples.")
        pass
    try:
        classifier.predict_relevance_for_user(user_id=user_id, relevance_by='stars', bert_model=bert,
                                              force_retraining=force_retrain)
    except ValueError as ve:
        print("Value error occurred when trying to get relevant thumbs for user. Skipping "
              "this user.")
        logging.warning(ve)
        pass


def predict_ratings_for_all_users_store_to_redis():
    """
    This method handles the prediction and storing of the users to Redis

    @return:
    """
    user_methods = UserMethods()
    all_users_df = user_methods.get_users_dataframe()
    classifier = Classifier()
    print("Loading BERT multilingual model...")
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    for user_row in zip(*all_users_df.to_dict("list").values()):
        print("user_id:")
        print(user_row[0])
        try:
            classifier.predict_relevance_for_user(user_id=user_row[0], relevance_by='thumbs', bert_model=bert)
        except ValueError as ve:
            print("Value error occurred when trying to get relevant thumbs for user. Skipping "
                  "this user.")
            logging.warning(ve)
            pass
        try:
            classifier.predict_relevance_for_user(user_id=user_row[0], relevance_by='stars', bert_model=bert)
        except ValueError as ve:
            print("Value error occurred when trying to get relevant thumbs for user. Skipping "
                  "this user.")
            logging.warning(ve)
            pass


def retrain_models_for_all_users():
    user_methods = UserMethods()
    all_users_df = user_methods.get_users_dataframe()
    classifier = Classifier()
    print("Loading BERT multilingual model...")
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    for user_row in zip(*all_users_df.to_dict("list").values()):
        print("user_id:")
        print(user_row[0])
        try:
            classifier.predict_relevance_for_user(user_id=user_row[0], relevance_by='thumbs', bert_model=bert,
                                                  store_to_redis=False, force_retraining=True)
        except ValueError as ve:
            print("Value error occurred when trying to get relevant thumbs for user. Skipping "
                  "this user.")
            logging.warning(ve)
            pass
        try:
            classifier.predict_relevance_for_user(user_id=user_row[0], relevance_by='stars', bert_model=bert,
                                                  store_to_redis=False, force_retraining=True)
        except ValueError as ve:
            print("Value error occurred when trying to get relevant thumbs for user. Skipping "
                  "this user.")
            logging.warning(ve)
            pass


def fill_bert_vector_representation(skip_already_filled=True, reversed_order=False, random_order=False, db="pgsql"):
    """
    This method is used for prefilling the BERT vector test doc representation to RDBS. Currently not used.

    @param skip_already_filled:
    @param reversed_order:
    @param random_order:
    @param db:
    @return:
    """
    print("Loading sentence bert multilingual model...")
    bert_model = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')

    database = DatabaseMethods()
    if skip_already_filled is False:
        database.connect()
        posts = database.get_all_posts()
        database.disconnect()
    else:
        database.connect()
        posts = database.get_not_bert_vectors_filled_posts()
        database.disconnect()

    number_of_inserted_rows = 0

    if reversed_order is True:
        logging.debug("Reversing list of posts...")
        posts.reverse()

    if random_order is True:
        print("Starting random_order iteration...")
        time.sleep(5)
        random.shuffle(posts)

    for post in posts:
        if len(posts) < 1:
            break
        post_id = post[0]
        slug = post[3]
        article_title = post[2]
        article_full_text = post[20]
        current_bert_vector_representation = post[41]

        print("Prefilling BERT vector reprenentation in article: " + slug)

        if skip_already_filled is True:
            if current_bert_vector_representation is None:
                if db == "pgsql":
                    if article_full_text is not None:
                        bert_vector_representation_of_current_post = bert_model(article_full_text).vector.reshape(1, -1)
                    elif article_title is not None:
                        bert_vector_representation_of_current_post = bert_model(article_title).vector.reshape(1, -1)
                    else:
                        raise ValueError("Post has not full text or even title text. Probably defected post."
                                         "Remove from DB.")
                    bert_vector_representation_of_current_post = (pickle
                                                                  .dumps(bert_vector_representation_of_current_post))
                    database.connect()
                    database.insert_bert_vector_representation(
                        bert_vector_representation=bert_vector_representation_of_current_post,
                        article_id=post_id)
                    database.disconnect()
                else:
                    raise NotImplementedError
            else:
                print("Skipping.")
        else:
            if db == "pgsql":
                bert_vector_representation_of_current_post = bert_model(article_full_text).vector.reshape(1, -1)
                print("article_title")
                print(article_title)
                print("bert_vector_representation_of_current_post (full_text)")
                print(bert_vector_representation_of_current_post)
                database.connect()
                database.insert_bert_vector_representation(
                    bert_vector_representation=bert_vector_representation_of_current_post,
                    article_id=post_id)
                database.disconnect()
            else:
                raise NotImplementedError
            number_of_inserted_rows += 1
            if number_of_inserted_rows > 20:
                print("Refreshing list of posts for finding only not prefilled posts.")
                fill_bert_vector_representation(db=db, skip_already_filled=skip_already_filled,
                                                reversed_order=reversed_order,
                                                random_order=random_order)


"""
** prefill_for_user(user_id) METHOD WAS HERE. REMOVED DUE TO UNCLEAR USE CASE. **
"""