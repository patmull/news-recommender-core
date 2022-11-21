import logging

from rabbitmq_receive import call_collaborative_prefillers
from research.user_based.user_relevance_eval import user_relevance_asessment
from src.prefillers.user_based_prefillers.prefilling_user_classifier import retrain_models_for_all_users

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from try_hybrid_methods.")

if __name__ == '__main__':
    """
    print("Loading BERT multilingual model...")
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    svm = Classifier()
    print(svm.predict_relevance_for_user(user_id=431, relevance_by='thumbs', only_with_prefilled_bert_vectors=False,
                                         bert_model=bert, force_retraining=True))
    print(svm.predict_relevance_for_user(user_id=431, relevance_by='stars', only_with_prefilled_bert_vectors=False,
                                         bert_model=bert))
    """
    # predict_ratings_for_user_store_to_redis(3118)
    call_collaborative_prefillers(method='classifier', msg_body='{"user_id":"3146"}', retrain_classifier=True)

    # retrain_models_for_all_users()
    