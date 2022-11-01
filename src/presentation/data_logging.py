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
