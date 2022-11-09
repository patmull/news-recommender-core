import logging
import time

from rabbitmq_receive import init_consuming


def consume_queue(queue_name):
    while True:
        try:
            init_consuming(queue_name)
        except Exception as e:
            logging.warning("EXCEPTION OCCURRED WHEN RUNNING PIKA:")
            logging.warning(e)
        except (RuntimeError, TypeError, NameError) as e:
            logging.warning("ERROR OCCURRED WHEN RUNNING PIKA:")
            logging.warning(e)
        time.sleep(15)
