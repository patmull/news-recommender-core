import logging

import numpy as np
from tabulate import tabulate

from src.constants import naming
import pandas as pd


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


def save_dataframe(df):
    if isinstance(df, pd.DataFrame):
        csv_file_path = f"logs/results_df_fuzzy_{naming.SAVE_COUNTER}.csv"
        tex_file_path = f"logs/results_df_fuzzy_{naming.SAVE_COUNTER}.tex"
        df.to_csv(csv_file_path)
        df.to_latex(tex_file_path)
        naming.SAVE_COUNTER += 1
    else:
        logging.warning("Not a Pandas Dataframe! Skipping...")
        logging.warning(f"Type: {type(df)}")
        logging.debug(df)


def refer_to_file():
    return f"See the file num. {naming.SAVE_COUNTER} in dataframes folder"


def display_tabulate(numpy_array):
    # Select the first 3 rows and columns
    first_part = numpy_array[:3, :3]
    # Select the last 3 rows and columns
    last_part = numpy_array[-3:, -3:]

    shortened_array = np.concatenate((first_part, last_part), axis=1)

    try:
        return tabulate(shortened_array, tablefmt="latex", floatfmt=".2f")
    except TypeError as te:
        logging.warning("Could not tabulate, saving just as sigma.")
        logging.error(te)
        return str(numpy_array)
