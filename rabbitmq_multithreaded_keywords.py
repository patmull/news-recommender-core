import functools
import logging
import os
import traceback

import pika
import threading
import time

from mail_sender import send_error_email
from rabbitmq_receive import is_init_or_test, call_collaborative_prefillers
from src.messaging.init_channels import ChannelConstants

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def ack_message(channel, delivery_tag):
    """Note that `channel` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    if channel.is_open:
        channel.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def do_work_keywords(connection, channel, delivery_tag, body):
    thread_id = threading.get_ident()
    fmt1 = 'Thread id: {} Delivery tag: {} Message body: {}'
    LOGGER.info(fmt1.format(thread_id, delivery_tag, body))
    # Sleeping to simulate 10 seconds of work

    try:
        # User classifier update
        logging.debug(ChannelConstants.USER_PRINT_CALLING_PREFILLERS)
        method = 'user_keywords'
        call_collaborative_prefillers(method, body)
        method = 'hybrid'
        call_collaborative_prefillers(method, body)
    except Exception as e:
        logging.warning(str(e))
        raise e
        # send_error_email(traceback.format_exc())

    cb = functools.partial(ack_message, channel, delivery_tag)
    connection.add_callback_threadsafe(cb)


def on_message(channel, method_frame, header_frame, body, args):
    logging.info("[x] Received %r" % body.decode())
    logging.info("Properties:")
    logging.info(header_frame)
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    if body.decode():
        if not is_init_or_test(body.decode()):
            logging.debug(ChannelConstants.USER_PRINT_CALLING_PREFILLERS)

            (connection, threads) = args
            delivery_tag = method_frame.delivery_tag
            t = threading.Thread(target=do_work_keywords, args=(connection, channel, delivery_tag, body))
            t.start()
            threads.append(t)


rabbitmq_user = os.environ.get('RABBITMQ_USER')
rabbitmq_password = os.environ.get('RABBITMQ_PASSWORD')
rabbitmq_host = os.environ.get('RABBITMQ_HOST')
rabbitmq_vhost = os.environ.get('RABBITMQ_VHOST', rabbitmq_user)
port = os.environ.get("port", 5672)

credentials = pika.credentials.PlainCredentials(
    username=rabbitmq_user, password=rabbitmq_password  # type: ignore
)
# Note: sending a short heartbeat to prove that heartbeats are still
# sent even though the worker simulates long-running work
connection_params = pika.ConnectionParameters(
    host=rabbitmq_host,
    port=port,
    credentials=credentials,
    virtual_host=rabbitmq_vhost,
    heartbeat=600  # This was initially set to 600
    # ** Here was the blocked connection timeout. Removed due to possible cause of the channel close problem.
)

connection = pika.BlockingConnection(connection_params)

queue_name = 'user-keywords-updated-queue'
routing_key = 'user.keywords.event.updated'

channel = connection.channel()
channel.exchange_declare(exchange='user', exchange_type="direct", passive=False, durable=True, auto_delete=False)
channel.queue_declare(queue=queue_name, auto_delete=False, durable=True)
channel.queue_bind(queue=queue_name, exchange='user',
                   routing_key=routing_key)
# Note: prefetch is set to 1 here as an example only and to keep the number of threads created
# to a reasonable amount. In production you will want to test with different prefetch values
# to find which one provides the best performance and usability for your solution
channel.basic_qos(prefetch_count=1)

threads = []
on_message_callback = functools.partial(on_message, args=(connection, threads))
channel.basic_consume(on_message_callback=on_message_callback, queue=queue_name)  # type: ignore

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()

# Wait for all to complete
for thread in threads:
    thread.join()

connection.close()
