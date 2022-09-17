from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.hybrid_methods import \
    get_most_similar_from_tfidf_matrix
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf


def main():
    test_user_id = 431
    searched_slug_1 = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    searched_slug_2 = "salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem"
    searched_slug_3 = "sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik"

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]
    print(get_most_similar_from_tfidf_matrix(user_id=test_user_id, posts_to_compare=test_slugs))


if __name__ == "__main__": main()
