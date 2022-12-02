import time

from src.recommender_core.recommender_algorithms.learn_to_rank.learn_to_rank_methods import LightGBMMethods, \
    get_posts_lightgbm


def main():
    start_time = time.time()

    light_gbm = LightGBMMethods()
    #light_gbm.train_lightgbm_document_based('tradicni-remeslo-a-rucni-prace-se-ceni-i-dnes-jejich-znacka-slavi-uspech')
    get_posts_lightgbm('zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy', True)
    print("--- %s seconds ---" % (time.time() - start_time))


# noinspection
if __name__ == "__main__": main()
