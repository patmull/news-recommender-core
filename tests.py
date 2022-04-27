import time

import numpy as np

from content_based_algorithms.doc2vec import Doc2VecClass
from content_based_algorithms.helper import Helper
from content_based_algorithms.tfidf import TfIdf
from content_based_algorithms.word2vec import Word2VecClass


def tfidf():
    tf_idf = TfIdf()
    print(tf_idf.recommend_posts_by_all_features_preprocessed(
        "chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach"))


def main():
    """
    results = []
    for i in range(0,30):
        start_time = time.time()
        tfidf()
        result = time.time() - start_time
        results.append(result)
        print("--- %s seconds ---" % result)

    print("Average time of execution:")
    print(np.average(results))
    """

    helper = Helper()
    # helper.clear_blank_lines_from_txt("datasets/idnes_preprocessed.txt")
    # doc2vec = Doc2VecClass()
    # print(doc2vec.get_similar_doc2vec("chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach", train=True, limited=False))
    # tfIdf = TfIdf()
    # print(tfidf.recommend_posts_by_all_features_preprocessed("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"))
    word2vec = Word2VecClass()
    word2vec.get_similar_word2vec("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy")


if __name__ == '__main__':
    main()
