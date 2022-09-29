import json
import traceback

import pika.exceptions

from src.messaging.init_channels import publish_rabbitmq_channel, ChannelConstants
from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative
from src.recommender_core.data_handling.data_connection import init_rabbitmq
from src.prefillers.prefilling_all import run_prefilling

# NOTICE: Logging didn't work really well for Pika so far...

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
            methods = ['svd']
            print("Received message o queue.")
            print("I'm calling method for updating of " + methods[0] + " prefilled recommendation...")
            try:
                print("Received JSON")
                received_data = json.loads(body)
                received_user_id = received_data['user_id']
                print("Recommender Core Prefilling class will be run for the user of ID:")
                print(received_user_id)
                run_prefilling_collaborative(methods=methods, user_id=received_user_id, test_run=False)
            except Exception as e:
                print("Exception occurred" + str(e))
                traceback.print_exception(None, e, e.__traceback__)


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def user_added_keywords(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if body.decode() == ChannelConstants.MESSAGE:
            print("Received queue INIT message. Waiting for another messages.")
        else:
            methods = ['user_keywords']
            print("Received message o queue.")
            print("I'm calling method for updating of " + methods[0] + " prefilled recommendation...")
            try:
                print("Received JSON")
                received_data = json.loads(body)
                received_user_id = received_data['user_id']
                print("Recommender Core Prefilling class will be run for the user of ID:")
                print(received_user_id)
                run_prefilling_collaborative(methods=methods, user_id=received_user_id, test_run=False)
            except Exception as e:
                print("Exception occurred" + str(e))
                traceback.print_exception(None, e, e.__traceback__)


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def user_added_categories(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode():
        if body.decode() == ChannelConstants.MESSAGE:
            print("Received queue INIT message. Waiting for another messages.")
        else:
            methods = ['best_rated_by_others_in_user_categories']
            print("Received message o queue.")
            print("I'm calling method for updating of " + methods[0] + " prefilled recommendation...")
            try:
                print("Received JSON")
                received_data = json.loads(body)
                received_user_id = received_data['user_id']
                print("Recommender Core Prefilling class will be run for the user of ID:")
                print(received_user_id)
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
