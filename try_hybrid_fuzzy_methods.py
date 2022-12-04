import logging

from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import \
    precalculate_and_save_sim_matrix_for_all_posts, get_most_similar_by_hybrid
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from try_hybrid_methods.")


def main():
    hybrid_results, fuzzy_hybrid_results = get_most_similar_by_hybrid(431, load_from_precalc_sim_matrix=False,
                                                                      use_fuzzy=True)
    print(hybrid_results)
    print(fuzzy_hybrid_results)


if __name__ == "__main__": main()
