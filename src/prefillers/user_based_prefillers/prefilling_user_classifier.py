import logging

import spacy_sentence_bert
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    Classifier

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from data?manipulation.")

bert_classifier_error_text = ("Value error occurred when trying to get relevant thumbs for evalutation. "
                              "Skipping this evalutation.")


def predict_ratings_for_user_store_to_redis(user_id, force_retrain=False):
    """
    Predictign relevant articles from evalutation base on classifier model (SVC / Random Forrest methods) using BERT
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
        logging.error(bert_classifier_error_text)
        logging.warning(ve)
        logging.warning("This is probably caused by insufficient number of examples for thumbs."
                        "User also needs to rate some posts both by thumbs up and down in order "
                        "to provide sufficient number of examples.")

    try:
        classifier.predict_relevance_for_user(user_id=user_id, relevance_by='stars', bert_model=bert,
                                              force_retraining=force_retrain)
    except ValueError as ve:
        logging.error(bert_classifier_error_text)
        logging.warning(ve)


def predict_ratings_for_all_users_store_to_redis():
    """
    This method handles the prediction and storing of the users to Redis

    @return:
    """
    user_methods = UserMethods()
    all_users_df = user_methods.get_users_dataframe()
    classifier = Classifier()
    logging.info("Loading BERT multilingual model...")
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    for user_row in zip(*all_users_df.to_dict("list").values()):
        try:
            classifier.predict_relevance_for_user(user_id=user_row[0], relevance_by='thumbs', bert_model=bert)
        except ValueError as ve:
            logging.error("Value error occurred when trying to get relevant thumbs for evalutation. Skipping "
                          "this evalutation.")
            logging.warning(ve)
        try:
            classifier.predict_relevance_for_user(user_id=user_row[0], relevance_by='stars', bert_model=bert)
        except ValueError as ve:
            print("Value error occurred when trying to get relevant thumbs for evalutation. Skipping "
                  "this evalutation.")
            logging.warning(ve)

# *** HERE were experimental code of BERT models *** ABANDONED DUE TO: not satisfying results
