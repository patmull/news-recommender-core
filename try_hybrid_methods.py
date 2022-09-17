from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf


def main():
    test_user_id = 431
    tfidf = TfIdf()
    print(tfidf.get_most_similar_from_tfidf_matrix(test_user_id))


if __name__ == "__main__": main()
