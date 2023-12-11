from src.messaging.amqp.rabbitmq_publisher import publish_channel


class ChannelConstants:
    """
    TEST_MESSAGE: Used in PHPUnit tests from Moje-clanky module
    """
    USER_PRINT_CALLING_PREFILLERS = "Received message for pre-fillers to queue."
    MESSAGE = "Initializing queue from MC Core"
    TEST_MESSAGE = '{"test_json":"test"}'


def init_df_of_channel_names():
    """
    Global RabbitMQ naming of channels, key, exchange keys. Also contain a _dictionary of those values.
    :return: _dictionary of _channel attribute names.
    """
    queue_names = ['evalutation-post-star_rating-updated-queue',
                   'evalutation-keywords-updated-queue',
                   'evalutation-categories-updated-queue',
                   'post-features-updated-queue',
                   'evalutation-post-thumb_rating-updated-queue']
    routing_keys = ['evalutation.post.star_rating.event.updated',
                    'evalutation.keywords.event.updated',
                    'evalutation.categories.event.updated',
                    'post.features.updated.queue',
                    'evalutation.post.thumb_rating.event.updated']
    exchanges = ['evalutation', 'evalutation', 'evalutation', 'post', 'evalutation']

    init_messages = [ChannelConstants.MESSAGE] * len(queue_names)

    if len(queue_names) != len(routing_keys):
        raise ValueError("Length of queue_names and routing_keys does not match.")

    if len(queue_names) != len(routing_keys) != len(init_messages) != len(exchanges):
        raise ValueError("Length of init lists does not match!")

    dict_of_channel_init_values = {'queue_name': queue_names,
                                   'init_message': init_messages,
                                   'routing_key': routing_keys,
                                   'exchange': exchanges
                                   }

    return dict_of_channel_init_values


def publish_all_set_channels():
    """
    Publishing all channels by the naming set in init_df_of_channel_names() method
    :return:
    """
    df_of_channels = init_df_of_channel_names()

    for index, row in df_of_channels.iterrows():
        publish_channel(row['queue_name'], row['init_message'], row['routing_key'], row['exchange'])


def publish_rabbitmq_channel(queue_name):
    """
    Preparing publishing of the RabbitMQ channels
    :param queue_name:
    :return:
    """
    channels_df = init_df_of_channel_names()
    channel = channels_df.loc[channels_df['queue_name'] == queue_name]
    message = channel['init_message'].iloc[0]
    routing_key = channel['routing_key'].iloc[0]
    exchange = channel['exchange'].iloc[0]

    publish_channel(queue_name, message, routing_key, exchange)
