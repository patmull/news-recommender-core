import pika

from src.recommender_core.data_handling.data_connection import init_rabbitmq


def notify_scrapper_prefiller():
    rabbit_connection = init_rabbitmq()
    channel = rabbit_connection.channel()
    message = b'{"test_json":"test"}'
    channel.basic_publish(
        exchange='post',
        routing_key='post.features.updated.queue',
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2  # make message persistent
        )
    )

    print("[x] Sent %r" % message)
    rabbit_connection.close()


if __name__ == '__main__':
    notify_scrapper_prefiller()
