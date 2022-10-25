# publish.py
import pika

# This file is there for the purposes of manual invocation of prefilling by notify_prefillers.py file
# Normally, this is used in news-parser module for notification of this module (rabbitmq_receive.py file)
from src.recommender_core.data_handling.data_connection import init_rabbitmq

@PendingDeprecationWarning
def publish_channel(queue, message, routing_key, exchange=''):
    rabbit_connection = init_rabbitmq()
    channel = rabbit_connection.channel()

    channel.queue_declare(queue=queue, durable=True)
    channel.queue_bind(queue=queue, exchange=exchange, routing_key=routing_key)

    message = message
    channel.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2  # make message persistent
        )
    )

    print("[x] Sent %r" % message)
    rabbit_connection.close()
