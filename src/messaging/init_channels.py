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
    LIST_OF_QUEUES = ['user-post-star_rating-updated-queue',
                      'user-keywords-updated-queue',
                      'user-categories-updated-queue',
                      ]
    LIST_OF_ROUTING_KEYS = ['user.post.star_rating.event.updated',
                            'user.keywords.event.updated',
                            'user.categories.event.updated']
    EXCHANGE = ['user', 'user', 'user']

    init_messages = []

    if len(LIST_OF_QUEUES) == len(LIST_OF_ROUTING_KEYS):
        for i in range(len(LIST_OF_QUEUES)):
            init_messages.append(ChannelConstants.MESSAGE)

    else:
        raise ValueError("Length of LIST_OF_QUEUES and LIST_OF_ROUTING_KEYS does not macth.")

    if len(LIST_OF_QUEUES) != len(LIST_OF_ROUTING_KEYS) != len(init_messages) != len(EXCHANGE):
        raise ValueError("Length of init lists does not macth!")

    dict_of_channel_init_values = {'queue_name': LIST_OF_QUEUES,
                                   'init_message': init_messages,
                                   'routing_key': LIST_OF_ROUTING_KEYS,
                                   'exchange': EXCHANGE
                                   }

    df_of_channels = pd.DataFrame.from_dict(dict_of_channel_init_values)
    print("df_of_channels:")
    print(df_of_channels)

    return df_of_channels


def publish_all_set_channels():
    df_of_channels = init_df_of_channel_names()

    for index, row in df_of_channels.iterrows():
        publish_channel(row['queue_name'], row['init_message'], row['routing_key'], row['exchange'])


def publish_rabbitmq_channel(queue_name):
    df_of_channels = init_df_of_channel_names()
    queue = queue_name
    message = df_of_channels.loc[df_of_channels['queue_name'] == queue_name, 'init_message'].iloc[0]
    routing_key = df_of_channels.loc[df_of_channels['queue_name'] == queue_name, 'routing_key'].iloc[0]
    exchange = df_of_channels.loc[df_of_channels['queue_name'] == queue_name, 'exchange'].iloc[0]

    publish_channel(queue, message, routing_key, exchange)
