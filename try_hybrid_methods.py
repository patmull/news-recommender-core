import json
import time

import numpy as np
import pandas as pd

from recommender_core.data_handling.data_queries import RecommenderMethods
from recommender_core.recommender_algorithms.user_based_algorithms.collaboration_based_recommendation import SvdClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import \
    get_similarity_matrix_from_pairs_similarity, select_list_of_posts_for_user, \
    get_most_similar_by_hybrid


def main():
    test_user_id = 431

    searched_slug_1 = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    searched_slug_2 = "salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem"
    searched_slug_3 = "sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik"

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]

    start = time.time()
    print(get_most_similar_by_hybrid(test_user_id))
    end = time.time()
    print(end - start)


if __name__ == "__main__": main()
