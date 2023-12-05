import datetime
import logging
import os

from src.constants.file_paths import get_cached_posts_file_path
from src.recommender_core.data_handling.data_queries import RecommenderMethods
# TODO: Replace logging.debugs with debug logging. Priority: MEDIUM


def check_if_cache_exists_and_fresh():
    if os.path.exists(get_cached_posts_file_path()):
        today = datetime.datetime.today()
        modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(get_cached_posts_file_path()))
        duration = today - modified_date
        # if file older than 1 day
        if duration.total_seconds() / (24 * 60 * 60) > 1:
            return False
        else:
            recommender_methods = RecommenderMethods()
            cached_df = recommender_methods.get_posts_dataframe(force_update=False, from_cache=True)
            sql_columns = recommender_methods.get_sql_columns().tolist()  # tolist() for converting Pandas index to list
            num_of_sql_rows = recommender_methods.get_sql_num_of_rows()
            logging.debug("sql_columns:")
            logging.debug(sql_columns)
            sql_columns.remove('bert_vector_representation')
            # -1 because bert_vector_representation needs to be excluded from cache
            if len(cached_df.columns) == (len(sql_columns) - 1):
                if set(cached_df.columns) == set(sql_columns):
                    if len(cached_df.index) == len(num_of_sql_rows):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
    else:
        return False


def create_app():
    # initializing files needed for the start of application
    # checking needed parts...

    if not check_if_cache_exists_and_fresh():
        logging.info("Posts cache file does not exists, older than 1 day or columns do not match PostgreSQL "
                     "columns and rows.")
        logging.debug("Creating posts cache file...")
        recommender_methods = RecommenderMethods()
        recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)


if __name__ == "__main__":
    create_app()
