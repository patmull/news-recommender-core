import logging

from rabbitmq_receive import call_collaborative_prefillers

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from try_hybrid_methods.")

if __name__ == '__main__':
    # predict_ratings_for_user_store_to_redis(3118)
    call_collaborative_prefillers(method='classifier', msg_body='{"user_id":"3150"}', retrain_classifier=True)

    # retrain_models_for_all_users()
    