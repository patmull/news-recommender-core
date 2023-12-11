import logging


def log_dataframe_info(df):
    """
    This is the global Dataframe info ogigng method_name. Should be defiitely used more
    than it is currently. It would improve the code readability and would save time.

    @param df:
    @return:
    """
    logging.debug("-------------------------------")
    logging.debug("Dataframe info:")
    logging.debug("-------------------------------")

    logging.debug("df info:")
    logging.debug(df.info(verbose=True))
