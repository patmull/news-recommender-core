import os
import ssl
import pika


def init_rabbitmq():
    ssl_enabled = os.environ.get("ssl", False)

    rabbitmq_user = os.environ.get('CLOUDAMQP_USERNAME', 'grzxhywg')
    rabbitmq_password = os.environ.get('CLOUDAMQP_PASSWORD', 'HEXls0RSd0nQLvpWcF5lhguRhR4JKWj4')
    rabbitmq_host = os.environ.get('CLOUDAMQP_HOST', 'stingray.rmq.cloudamqp.com')
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

    return pika.BlockingConnection(connection_params)
