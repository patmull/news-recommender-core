import logging
import time
import traceback

from mail_sender import send_error_email
from src.messaging.amqp.rabbitmq_receive import init_consuming, restart_channel


def consume_queue(queue_name):
    """
    Global RabbitMQ init consuming class which purpose is to not crash the program on Exception.
    :param queue_name:
    :return:
    """
    while True:
        try:
            init_consuming(queue_name)
        except Exception as e:
            logging.warning(f"EXCEPTION OCCURRED WHEN RUNNING PIKA: {e}")
            send_error_email(traceback.format_exc())
            logging.warning("Trying to restart the _channel")
            restart_channel(queue_name)

        time.sleep(15)

