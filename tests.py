import time

import numpy as np

from content_based_algorithms.tfidf import TfIdf


def tfidf():
    tf_idf = TfIdf()
    print(tf_idf.recommend_posts_by_all_features_preprocessed(
        "chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach"))


def main():
    results = []
    for i in range(0,30):
        start_time = time.time()
        tfidf()
        result = time.time() - start_time
        results.append(result)
        print("--- %s seconds ---" % result)

    print("Average time of execution:")
    print(np.average(results))


if __name__ == '__main__':
    main()
