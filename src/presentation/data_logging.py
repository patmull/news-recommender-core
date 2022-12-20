import logging


def get_logger(logging: object) -> object:
    """

    :param logging:
    :return:
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Enabling Word2Vec logging
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.NOTSET)
    logger = logging.getLogger()  # get the root logger
    logger.info("Testing file write")
    return logger


def log_dataframe_info(df):
    """
    This is the global Dataframe info ogigng method. Should be defiitely used more
    than it is currently. It would improve the code readability and would save time.

    @param df:
    @return:
    """
    logging.debug("-------------------------------")
    logging.debug("Dataframe info:")
    logging.debug("-------------------------------")

    logging.debug("df info:")
    logging.debug(df.info(verbose=True))
