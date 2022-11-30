import logging
import time
import traceback

from mail_sender import send_error_email
from rabbitmq_receive import init_consuming, restart_channel


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
            logging.warning("EXCEPTION OCCURRED WHEN RUNNING PIKA:")
            logging.warning(e)
            send_error_email(traceback.format_exc())
            logging.warning("Trying to restart the channel")
            restart_channel(queue_name)
        except (RuntimeError, TypeError, NameError) as e:
            logging.warning("ERROR OCCURRED WHEN RUNNING PIKA:")
            logging.warning(type(e).__name__)
            logging.warning(e)
            send_error_email(traceback.format_exc())
            logging.warning("Trying to restart the channel")
            restart_channel(queue_name)
        time.sleep(15)
