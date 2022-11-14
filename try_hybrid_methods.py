import logging

from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import \
    precalculate_and_save_sim_matrix_for_all_posts
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from try_hybrid_methods.")


def main():
    # precalculate_and_save_sim_matrix_for_all_posts(['doc2vec', 'word2vec'])
    # TODO: Here were some other tryings. Those can be later used for testing.
    # body_message = f"{user_id: 3118}"
    recommender_methods = RecommenderMethods()
    # recommender_methods.update_cache_of_posts_df()
    print(run_prefilling_collaborative(methods=["hybrid"], user_id=3134))


if __name__ == "__main__": main()
