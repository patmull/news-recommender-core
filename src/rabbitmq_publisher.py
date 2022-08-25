# publish.py
import ssl
import pika, os

# This file is there for the purposes of manual invocation of prefilling by notify_prefillers.py file
# Normally, this is used in news-parser module for notification of this module (rabbitmq_recieve.py file)

ssl_enabled = os.environ.get("ssl", False)

rabbitmq_user = os.environ.get('CLOUDAMQP_USERNAME', '***REMOVED***')
rabbitmq_password = os.environ.get('CLOUDAMQP_PASSWORD', '***REMOVED***')
rabbitmq_host = os.environ.get('CLOUDAMQP_HOST', '***REMOVED***')
rabbitmq_vhost = os.environ.get('CLOUDAMQP_VHOST', rabbitmq_user)


if ssl_enabled:
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(os.environ.get("ca_bundle", '/etc/pki/tls/certs/ca-bundle.crt'))
    ssl_options = pika.SSLOptions(context)
    port = os.environ.get("port", 5671)
else:
    ssl_options = None
    port = os.environ.get("port", 5672)

credentials = pika.credentials.PlainCredentials(
    username=rabbitmq_user, password=rabbitmq_password
)
connection_params = pika.ConnectionParameters(
    host=rabbitmq_host,
    ssl_options=ssl_options,
    port=port,
    credentials=credentials,
    virtual_host=rabbitmq_vhost,
    heartbeat=600,
    blocked_connection_timeout=300
)


def notify_prefiller():
    rabbit_connection = pika.BlockingConnection(connection_params)
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