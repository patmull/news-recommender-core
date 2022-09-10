# publish.py
import pika

# This file is there for the purposes of manual invocation of prefilling by notify_prefillers.py file
# Normally, this is used in news-parser module for notification of this module (rabbitmq_recieve.py file)
from src.recommender_core.data_handling.data_connection import init_rabbitmq


def notify_prefiller():
    rabbit_connection = init_rabbitmq()
    channel = rabbit_connection.channel()

    channel.queue_declare(queue='new_articles_alert', durable=True)

    message = b"new_articles_scrapped"
    channel.basic_publish(
        exchange='',
        routing_key='new_articles_alert',
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2  # make message persistent
        )
    )

    print("[x] Sent %r" % message)
    rabbit_connection.close()