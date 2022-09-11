import time

from recommender_core.recommender_algorithms.learn_to_rank.learn_to_rank_methods import LightGBMMethods


def main():
    start_time = time.time()

    ligh_gbm = LightGBMMethods()
    # lighGBM.train_lightgbm_document_based('tradicni-remeslo-a-rucni-prace-se-ceni-i-dnes-jejich-znacka-slavi-uspech')
    ligh_gbm.get_posts_lightgbm('zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy', True)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__": main()
