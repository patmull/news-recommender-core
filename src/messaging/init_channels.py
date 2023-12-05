import pandas as pd
from rabbitmq_publisher import publish_channel


class ChannelConstants:
    """
    TEST_MESSAGE: Used in PHPUnit tests from Moje-clanky module
    """
    USER_PRINT_CALLING_PREFILLERS = "Received message for pre-fillers to queue."
    MESSAGE = "Initializing queue from MC Core"
    TEST_MESSAGE = '{"test_json":"test"}'


def init_df_of_channel_names():
    """
    Global RabbitMQ naming of channels, key, exchange keys. Also contain a dictionary of those values.
    :return: Dataframe of channel attribute names.
    """
    list_of_queues = ['user-post-star_rating-updated-queue',
                      'user-keywords-updated-queue',
                      'user-categories-updated-queue',
                      'post-features-updated-queue',
                      'user-post-thumb_rating-updated-queue']
    list_of_routing_keys = ['user.post.star_rating.event.updated',
                            'user.keywords.event.updated',
                            'user.categories.event.updated',
                            'post.features.updated.queue',
                            'user.post.thumb_rating.event.updated']
    exchange = ['user', 'user', 'user', 'post', 'user']

    init_messages = []

    if len(list_of_queues) == len(list_of_routing_keys):
        for i in range(len(list_of_queues)):
            init_messages.append(ChannelConstants.MESSAGE)

    else:
        raise ValueError("Length of list_of_queues and list_of_routing_keys does not macth.")

    if len(list_of_queues) != len(list_of_routing_keys) != len(init_messages) != len(exchange):
        raise ValueError("Length of init lists does not macth!")

    dict_of_channel_init_values = {'queue_name': list_of_queues,
                                   'init_message': init_messages,
                                   'routing_key': list_of_routing_keys,
                                   'exchange': exchange
                                   }

    df_of_channels = pd.DataFrame.from_dict(dict_of_channel_init_values)
    print("df_of_channels:")
    print(df_of_channels)

    return df_of_channels


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
    df_of_channels = init_df_of_channel_names()
    queue = queue_name
    message = df_of_channels.loc[df_of_channels['queue_name'] == queue_name, 'init_message'].iloc[0]
    routing_key = df_of_channels.loc[df_of_channels['queue_name'] == queue_name, 'routing_key'].iloc[0]
    exchange = df_of_channels.loc[df_of_channels['queue_name'] == queue_name, 'exchange'].iloc[0]

    publish_channel(queue, message, routing_key, exchange)
