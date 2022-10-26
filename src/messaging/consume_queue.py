import time

from rabbitmq_receive import init_consuming


def consume_queue(queue_name):
    while True:
        try:
            init_consuming(queue_name)
        except Exception as e:
            print("EXCEPTION OCCURRED WHEN RUNNING PIKA:")
            print(e)
        except (RuntimeError, TypeError, NameError) as e:
            print("ERROR OCCURRED WHEN RUNNING PIKA:")
            print(e)
        time.sleep(15)
