import logging


def combine_features_from_single_df_row(single_row_df, list_of_features):
    logging.debug("single_row_df")
    logging.debug(single_row_df)
    logging.debug("list_of_features")
    logging.debug(list_of_features)
    return (single_row_df[list_of_features] + ' ').sum(axis=1).str.strip()
