import json
import traceback

import pika.exceptions

from src.messaging.init_channels import publish_rabbitmq_channel, ChannelConstants
from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative
from src.recommender_core.data_handling.data_connection import init_rabbitmq

from src.recommender_core.data_handling.model_methods.user_methods import UserMethods

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.

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
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if body.decode() == ChannelConstants.MESSAGE:
            print("Received queue INIT message. Waiting for another messages.")
        else:
            methods = ['user_keywords']
            call_collaborative_prefillers(methods, body)


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def user_added_categories(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if body.decode() == ChannelConstants.MESSAGE:
            print("Received queue INIT message. Waiting for another messages.")
        else:
            method = 'best_rated_by_others_in_user_categories'
            print("Received message to queue.")
            call_collaborative_prefillers(method, body)


def call_collaborative_prefillers(method, msg_body):
    print("I'm calling method for updating of " + method + " prefilled recommendation...")
    try:
        print("Received JSON")
        received_data = json.loads(msg_body)
        received_user_id = received_data['user_id']

        print("Checking whether user is not test user...")
        user_methods = UserMethods()
        user = user_methods.get_user_dataframe(received_user_id)

        if user['name'].iloc[0].startswith('test-user'):

            if method == 'user_keywords':
                test_dict = [{"slug": "test",
                        "coefficient": 1.0},
                        {"slug": "test2",
                        "coefficient": 1.0}]
            else:
                test_dict = {"columns": ["post_id", "slug", "ratings_values"],
                        "index": [1, 2],
                        "data": [
                            [999999, "test", 1.0],
                            [9999999, "test2", 1.0],
                        ]}
            actual_json = json.dumps(test_dict)
            user_methods.insert_recommended_json_user_based(recommended_json=actual_json,
                                                            user_id=received_user_id, db="pgsql",
                                                            method=method)
        else:
            print("Recommender Core Prefilling class will be run for the user of ID:")
            print(received_user_id)
            methods = [method]
            run_prefilling_collaborative(methods=methods, user_id=received_user_id, test_run=False)
    except Exception as e:
        print("Exception occurred" + str(e))
        traceback.print_exception(None, e, e.__traceback__)


""" 
** HERE WAS A DECLARATION OF QUEUE ACTIVATED AFTER POST PREFILLING CALLING new_post_scrapped_callback() method.
Abandoned due to unclear use case. **
"""

# convention: [object/subject]-[action]-queue
queue_name = 'user-post-star_rating-updated-queue'
try:
    channel.basic_consume(queue=queue_name, on_message_callback=user_rated_by_stars_callback)
except pika.exceptions.ChannelClosedByBroker as e:
    print(e)
    publish_rabbitmq_channel(queue_name)
    channel.basic_consume(queue=queue_name, on_message_callback=user_rated_by_stars_callback)

queue_name = 'user-keywords-updated-queue'
try:
    channel.basic_consume(queue=queue_name, on_message_callback=user_added_keywords)
except pika.exceptions.ChannelClosedByBroker as e:
    print(e)
    publish_rabbitmq_channel(queue_name)
    channel.basic_consume(queue=queue_name, on_message_callback=user_added_keywords)

queue_name = 'user-categories-updated-queue'
try:
    channel.basic_consume(queue=queue_name, on_message_callback=user_added_categories)
except pika.exceptions.ChannelClosedByBroker as e:
    print(e)
    publish_rabbitmq_channel(queue_name)
    channel.basic_consume(queue=queue_name, on_message_callback=user_added_categories)

channel.start_consuming()
