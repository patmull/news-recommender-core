import json
import logging
import traceback

import pika.exceptions

from mail_sender import send_error_email
from src.messaging.init_channels import publish_rabbitmq_channel, ChannelConstants
from src.prefillers.prefilling_all import run_prefilling
from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative
from src.prefillers.user_based_prefillers.prefilling_user_classifier import predict_ratings_for_user_store_to_redis
from src.recommender_core.data_handling.data_connection import init_rabbitmq

from src.recommender_core.data_handling.model_methods.user_methods import UserMethods

for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging.")

rabbit_connection = init_rabbitmq()

channel = rabbit_connection.channel()

logging.info('[*] Waiting for messages. To exit press CTRL+C')


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def is_init_or_test(decoded_body):
    if decoded_body == ChannelConstants.MESSAGE:
        logging.info("Received queue INIT message. Waiting for another messages.")
        is_init_or_test_value = True
    elif decoded_body == ChannelConstants.TEST_MESSAGE:
        logging.info("Received queue TEST message. Waiting for another messages.")
        is_init_or_test_value = True
    else:
        is_init_or_test_value = False

    if is_init_or_test_value is True:
        logging.info("Successfully received. Not doing any action since this was init or test.")

    return is_init_or_test_value


def new_post_scrapped_callback(ch, method, body):
    logging.info("[x] Received %r" % body.decode())
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode() == "new_articles_scrapped":
        logging.info("Received message that new posts were scrapped.")
        logging.info("I'm calling prefilling db_columns...")
        if not is_init_or_test(body.decode()):
            try:
                # TODO: Remove debugging parameters
                run_prefilling(skip_cache_refresh=False)
            except Exception as e:
                logging.warning("Exception occurred" + str(e))
                traceback.print_exception(None, e, e.__traceback__)
                send_error_email(traceback.format_exc())


def user_rated_by_stars_callback(ch, method, properties, body):
    logging.info("[x] Received %r" % body.decode())
    logging.info("Properties:")
    logging.info(properties)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if not is_init_or_test(body.decode()):
            try:
                logging.debug(ChannelConstants.USER_PRINT_CALLING_PREFILLERS)
                method = 'svd'
                call_collaborative_prefillers(method, body)
                method = 'hybrid'
                call_collaborative_prefillers(method, body)
            except Exception as e:
                logging.warning(str(e))
                send_error_email(traceback.format_exc())
            """
            Classifier was commented out for now to make SVD and hybrid faster.
            Classifier of both thumbs and ratings should be updated in thumbs_rating_queue.
            """
            # method = 'classifier'
            # call_collaborative_prefillers(method, body)


def user_rated_by_thumb_callback(ch, method, properties, body):
    logging.info("[x] Received %r" % body.decode())
    logging.info("Properties:")
    logging.info(properties)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if not is_init_or_test(body.decode()):
            logging.debug(ChannelConstants.USER_PRINT_CALLING_PREFILLERS)
            try:
                # User classifier update
                method = 'classifier'
                call_collaborative_prefillers(method, body, retrain_classifier=True)
                method = 'hybrid'
                call_collaborative_prefillers(method, body)
            except Exception as e:
                logging.warning(str(e))
                send_error_email(traceback.format_exc())


def test_callback(ch, method, properties, body):
    logging.info("[x] Received %r" % body.decode())
    logging.info("Properties:")
    logging.info(properties)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if not is_init_or_test(body.decode()):
            logging.debug("Received message from testing queue.")
            logging.debug("Not doing anything, just consuming this message.")


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def user_added_keywords(ch, method, properties, body):
    logging.info("[x] Received %r" % body.decode())
    logging.info("Properties:")
    logging.info(properties)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if not is_init_or_test(body.decode()):
            try:
                logging.debug(ChannelConstants.USER_PRINT_CALLING_PREFILLERS)
                method = 'user_keywords'
                call_collaborative_prefillers(method, body)
                method = 'hybrid'
                call_collaborative_prefillers(method, body)
            except Exception as e:
                logging.warning(str(e))
                send_error_email(traceback.format_exc())


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def user_added_categories(ch, method, properties, body):
    logging.info("[x] Received %r" % body.decode())
    logging.info("Properties:")
    logging.info(properties)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if not is_init_or_test(body.decode()):
            try:
                logging.debug(ChannelConstants.USER_PRINT_CALLING_PREFILLERS)
                method = 'best_rated_by_others_in_user_categories'
                call_collaborative_prefillers(method, body)
                method = 'hybrid'
                call_collaborative_prefillers(method, body)
            except Exception as e:
                logging.warning(str(e))
                send_error_email(traceback.format_exc())


def insert_testing_json(received_user_id, method, heroku_testing_db=False):
    if method == "classifier":
        logging.warning("Storing classifier to DB is not implemented yet.")

    user_methods = UserMethods()
    logging.debug("Inserting testing JSON for testing user.")

    if method == 'user_keywords':
        test_dict = [{"slug": "test",
                      "coefficient": 1.0},
                     {"slug": "test2",
                      "coefficient": 1.0}]
    else:
        test_dict = dict(columns=["post_id", "slug", "ratings_values"], index=[1, 2], data=[
            [999999, "test", 1.0],
            [9999999, "test2", 1.0],
        ])
    actual_json = json.dumps(test_dict)
    logging.debug("_actual_json:")
    logging.debug(str(actual_json))
    logging.debug(type(actual_json))

    if heroku_testing_db:
        db = "pgsql_heroku_testing"
    else:
        db = "pgsql"

    user_methods.insert_recommended_json_user_based(recommended_json=actual_json,
                                                    user_id=received_user_id, db=db,
                                                    method=method)


def decode_msg_body_to_user_id(msg_body):
    received_data = json.loads(msg_body)
    received_user_id = int(received_data['user_id'])
    return received_user_id


def call_collaborative_prefillers(method, msg_body, retrain_classifier=False):
    logging.debug("I'm calling method for updating of " + method + " prefilled recommendation...")
    try:
        logging.debug("Received JSON")
        received_user_id = decode_msg_body_to_user_id(msg_body)

        logging.info("Checking whether user is not test user...")
        user_methods = UserMethods()
        user = user_methods.get_user_dataframe(received_user_id)
        try:
            if len(user.index) == 0:
                logging.warning("User's data are empty. User is probably not presented in DB")
                insert_testing_json(received_user_id, method, heroku_testing_db=True)
                test_user_name = True
            else:
                test_user_name = user['name'].values[0].startswith('test-user-dusk')
        except IndexError as ie:
            logging.warning("Index error occurred while trying to fetch information about the user. "
                            "User is probably not longer in database.")
            logging.warning("SEE FULL EXCEPTION MESSAGE:")
            raise ie

        if test_user_name:
            insert_testing_json(received_user_id, method)
        else:
            logging.info("Recommender Core Prefilling class will be run for the user of ID:")
            logging.info(received_user_id)
            if method == "classifier":
                """
                NOTICE: re-training influences also the hybrid recommendations but it's handled here to provide better
                user flow.
                """
                predict_ratings_for_user_store_to_redis(received_user_id, force_retrain=retrain_classifier)
            else:
                run_prefilling_collaborative(methods=[method], user_id=received_user_id, test_run=False)

    except Exception as ie:
        logging.warning("Exception occurred" + str(ie))
        traceback.print_exception(None, ie, ie.__traceback__)


"""
** HERE WAS A DECLARATION OF QUEUE ACTIVATED AFTER POST PREFILLING CALLING new_post_scrapped_callback() method.
Abandoned due to unclear use case. **
"""


# WARNING! This does not work. It consumes only the first queue in list!!!
@DeprecationWarning
def init_all_consuming_channels():
    queues = ['user-post-star_rating-updated-queue',
              'user-keywords-updated-queue',
              'user-categories-updated-queue',
              'post-features-updated-queue',
              'user-post-thumb_rating-updated-queue']
    for queue in queues:
        init_consuming(queue)


class Callback:
    event = None

    def __init__(self, event):
        self.event = event


def init_consuming(queue_name):
    if queue_name == 'user-post-star_rating-updated-queue':
        called_function = user_rated_by_stars_callback
    elif queue_name == 'user-keywords-updated-queue':
        called_function = user_added_keywords
    elif queue_name == 'user-categories-updated-queue':
        called_function = user_added_categories
    elif queue_name == 'post-features-updated-queue':
        called_function = new_post_scrapped_callback
    elif queue_name == 'user-post-thumb_rating-updated-queue':
        called_function = user_rated_by_thumb_callback
    elif queue_name == 'test-queue':
        called_function = test_callback
    else:
        raise ValueError('Bad queue_name supplied.')

    try:
        channel.basic_consume(queue=queue_name, on_message_callback=called_function)
    except pika.exceptions.ChannelClosedByBroker as ie:
        logging.warning(ie)
        publish_rabbitmq_channel(queue_name)
        channel.basic_consume(queue=queue_name, on_message_callback=user_rated_by_stars_callback)
        send_error_email(traceback.format_exc())

    channel.start_consuming()


def restart_channel(queue_name):
    publish_rabbitmq_channel(queue_name)
    channel.basic_consume(queue=queue_name, on_message_callback=user_rated_by_stars_callback)
    channel.start_consuming()
