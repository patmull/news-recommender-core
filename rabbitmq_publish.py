# publish.py
import pika

# This file is there for the purposes of manual invocation of prefilling by notify_prefillers.py file
# Normally, this is used in news-parser module for notification of this module (rabbitmq_receive.py file)
from rabbitmq_publisher import notify_prefiller
from src.recommender_core.data_handling.data_connection import init_rabbitmq

queue = 'user-post-star_rating-updated-queue'
message = "Contacting queue for SVD from MC Core"
routing_key = 'user.post.star_rating.event.updated'
exchange='user'

notify_prefiller(queue, message, routing_key, exchange)
