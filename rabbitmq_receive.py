import json
import time
import traceback

import pika.exceptions

from src.messaging.init_channels import publish_rabbitmq_channel, ChannelConstants
from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative
from src.recommender_core.data_handling.data_connection import init_rabbitmq

from src.recommender_core.data_handling.model_methods.user_methods import UserMethods

rabbit_connection = init_rabbitmq()

channel = rabbit_connection.channel()

channel.queue_declare(queue='new_articles_alert', durable=True)
print('[*] Waiting for messages. To exit press CTRL+C')
"""
** HERE WAS A DECLARATION OF new_post_scrapped_callback() method.
Abandoned due to unclear use case. **
"""


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def user_rated_by_stars_callback(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    print("Properties:")
    print(properties)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if body.decode() == ChannelConstants.MESSAGE:
            print("Received queue INIT message. Waiting for another messages.")
        else:
            method = 'svd'
            print("Received message o queue.")
            call_collaborative_prefillers(method, body)


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def user_added_keywords(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    print("Properties:")
    print(properties)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if body.decode() == ChannelConstants.MESSAGE:
            print("Received queue INIT message. Waiting for another messages.")
        else:
            method = 'user_keywords'
            call_collaborative_prefillers(method, body)


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def user_added_categories(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    print("Properties:")
    print(properties)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if body.decode() == ChannelConstants.MESSAGE:
            print("Received queue INIT message. Waiting for another messages.")
        else:
            method = 'best_rated_by_others_in_user_categories'
            print("Received message to queue.")
            call_collaborative_prefillers(method, body)


def insert_testing_json(received_user_id, method):

    user_methods = UserMethods()
    print("Inserting testing JSON for testing user.")

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
    print("actual_json:")
    print(str(actual_json))
    print(type(actual_json))
    user_methods.insert_recommended_json_user_based(recommended_json=actual_json,
                                                    user_id=received_user_id, db="pgsql",
                                                    method=method)


def call_collaborative_prefillers(method, msg_body):
    print("I'm calling method for updating of " + method + " prefilled recommendation...")
    try:
        print("Received JSON")
        received_data = json.loads(msg_body)
        received_user_id = received_data['user_id']

        print("Checking whether user is not test user...")
        user_methods = UserMethods()
        user = user_methods.get_user_dataframe(received_user_id)

        try:
            user_name = user['name'].values[0].startswith('test-user-dusk')
        except IndexError as ie:
            print("Index error occurred while trying to fetch information about the user. "
                  "User is probably not longer in database.")
            print("SEE FULL EXCEPTION MESSAGE:")
            raise ie

        if user_name:
            insert_testing_json(received_user_id, method)
        else:
            print("Recommender Core Prefilling class will be run for the user of ID:")
            print(received_user_id)
            run_prefilling_collaborative(methods=[method], user_id=received_user_id, test_run=False)
    except Exception as ie:
        print("Exception occurred" + str(ie))
        traceback.print_exception(None, ie, ie.__traceback__)


"""
** HERE WAS A DECLARATION OF QUEUE ACTIVATED AFTER POST PREFILLING CALLING new_post_scrapped_callback() method.
Abandoned due to unclear use case. **
"""


def init_consuming():
    # convention: [object/subject]-[action]-queue
    queue_name = 'user-post-star_rating-updated-queue'
    try:
        channel.basic_consume(queue=queue_name, on_message_callback=user_rated_by_stars_callback)
    except pika.exceptions.ChannelClosedByBroker as ie:
        print(ie)
        publish_rabbitmq_channel(queue_name)
        channel.basic_consume(queue=queue_name, on_message_callback=user_rated_by_stars_callback)

    queue_name = 'user-keywords-updated-queue'
    try:
        channel.basic_consume(queue=queue_name, on_message_callback=user_added_keywords)
    except pika.exceptions.ChannelClosedByBroker as ie:
        print(ie)
        publish_rabbitmq_channel(queue_name)
        channel.basic_consume(queue=queue_name, on_message_callback=user_added_keywords)

    queue_name = 'user-categories-updated-queue'
    try:
        channel.basic_consume(queue=queue_name, on_message_callback=user_added_categories)
    except pika.exceptions.ChannelClosedByBroker as ie:
        print(ie)
        publish_rabbitmq_channel(queue_name)
        channel.basic_consume(queue=queue_name, on_message_callback=user_added_categories)

    channel.start_consuming()


while True:
    try:
        init_consuming()
    except Exception as e:
        print("EXCEPTION OCCURRED WHEN RUNNING PIKA:")
        print(e)
    except (RuntimeError, TypeError, NameError) as e:
        print("ERROR OCCURRED WHEN RUNNING PIKA:")
        print(e)
    time.sleep(15)
