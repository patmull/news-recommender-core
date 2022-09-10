import traceback

from src.prefillers.prefilling_all import run_prefilling

rabbit_connection = init_rabbitmq()

channel = rabbit_connection.channel()

channel.queue_declare(queue='new_articles_alert', durable=True)
print('[*] Waiting for messages. To exit press CTRL+C')


def callback(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode() == "new_articles_scrapped":
        print("Recieved message that new posts were scrapped.")
        print("I'm calling prefilling methods...")
        try:
            run_prefilling()
        except Exception as e:
            print("Exception occured" + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='new_articles_alert', on_message_callback=callback)

channel.start_consuming()