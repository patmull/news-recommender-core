import traceback

from src.recommender_core.data_handling.data_connection import init_rabbitmq
from src.prefillers.prefilling_all import run_prefilling

rabbit_connection = init_rabbitmq()

channel = rabbit_connection.channel()

channel.queue_declare(queue='new_articles_alert', durable=True)
print('[*] Waiting for messages. To exit press CTRL+C')


# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def post_updated_callback(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode() == "new_articles_scrapped":
        print("Recieved message that new posts were scrapped.")
        print("I'm calling prefilling db_columns...")
        try:
            run_prefilling()
        except Exception as e:
            print("Exception occurred" + str(e))
            traceback.print_exception(None, e, e.__traceback__)

# NOTICE: properties needs to stay here even if PyCharm says it's not used!
def post_star_rating_updated_callback(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode() == "This is testing message from Laravel":
        print("Received message that new posts were scrapped.")
        print("I'm calling method for updating of prefilled SVD recommendation...")
        try:
            methods = ["svd"]
            print("Prefilling class will be run the next time...")
            # TODO: Enable below...
            # run_prefilling_collaborative(methods=methods, test_run=False)
        except Exception as e:
            print("Exception occurred" + str(e))
            traceback.print_exception(None, e, e.__traceback__)


# convention: [object/subject]-[action]-queue
channel.basic_qos(prefetch_count=1)
# channel.basic_consume(queue='posts-updated-queue', on_message_callback=post_updated_callback)
# TODO: Somehow send also user_id...
channel.basic_consume(queue='user-post-star_rating-updated-queue', on_message_callback=post_star_rating_updated_callback)

channel.start_consuming()
