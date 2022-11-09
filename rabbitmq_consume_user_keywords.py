import logging
import os

from src.messaging.consume_queue import consume_queue

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from " + os.path.basename(__file__))

if __name__ == '__main__':
    consume_queue('user-keywords-updated-queue')
