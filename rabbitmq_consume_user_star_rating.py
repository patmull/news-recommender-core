from src.messaging.consume_queue import consume_queue

if __name__ == '__main__':
    consume_queue('user-post-star_rating-updated-queue')
