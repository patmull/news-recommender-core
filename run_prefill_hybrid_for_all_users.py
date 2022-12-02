import logging
import time

import pandas as pd
import schedule

from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods

for handler in logging.root.handlers[:]:
   logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging.")


def job_prefill_hybrid_for_all_users():
    prefill_hybrid_for_all_users()


def prefill_hybrid_for_all_users(only_with_prefilled_bert_vectors=False, only_user=None):
    user_methods = UserMethods()
    recommender_methods = RecommenderMethods()

    all_users_df = user_methods.get_users_dataframe()
    for user_row in zip(*all_users_df.to_dict("list").values()):
        user_id = user_row[0]
        if only_user is not None:
            if user_id != only_user:
                continue

        logging.debug("Checking user %s." % user_id)
        try:
            df_posts_users_categories_relevance = recommender_methods \
                .get_posts_users_categories_ratings_df(user_id=user_id,
                                                       only_with_bert_vectors=only_with_prefilled_bert_vectors)
            df_user_history = user_methods.get_user_read_history(user_id=user_id)
            val_err = False
        except ValueError as e:
            if "There are no thumbs for a given user" in str(e):
                logging.debug("User has no thumbs, skipping this user.")
                continue
            df_posts_users_categories_relevance = None
            df_user_history = None
            val_err = True

        if val_err is False:
            if len(df_posts_users_categories_relevance.index) < 10 or len(df_user_history.index) < 3:
                logging.debug("User has not enough of ratings or seen articles. Skipping.")
                pass
            else:
                user = user_methods.get_user_dataframe(user_id)
                if pd.isnull(user['recommended_by_hybrid'].iloc[0]) or user['recommended_by_hybrid'].iloc[0] == "":
                    logging.debug("User has empty hybrid recommendations. Starting to "
                                  "compute for him...")
                    run_prefilling_collaborative(methods=["hybrid"], user_id=user_id)
                else:
                    logging.debug("User's hybrid recommendations are not empty. Skipping.")
                    pass
        else:
            logging.debug("Value error occurred before, skipping this user")
            pass


def main():
    # for testing purposes (immediately triggers the method)
    job_prefill_hybrid_for_all_users()

    schedule.every(4).minutes.do(job_prefill_hybrid_for_all_users)

    while 1:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__": main()