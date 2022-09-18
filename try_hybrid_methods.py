import json
import time

import numpy as np
import pandas as pd

from recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import \
    get_similarity_matrix_from_pairs_similarity, select_list_of_posts_for_user, \
    get_most_similar_from_content_based_matrix_and_delivered_posts


def main():
    test_user_id = 431
    searched_slug_1 = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    searched_slug_2 = "salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem"
    searched_slug_3 = "sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik"

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]

    """
    list_of_slugs, list_of_slugs_from_history = select_list_of_posts_for_user(user_id=test_user_id,
                                                                              posts_to_compare=test_slugs)
    
    doc2vec = Doc2VecClass()
    get_similarity_matrix_from_pairs_similarity(doc2vec, list_of_slugs, test_slugs, list_of_slugs_from_history)
    """
    """
    word2vec = Word2VecClass()
    get_similarity_matrix_from_pairs_similarity(word2vec, list_of_slugs, test_slugs, list_of_slugs_from_history)
    
    start = time.time()
    print(get_most_similar_from_content_based_matrix_and_delivered_posts(test_user_id, test_slugs))
    end = time.time()
    print(end - start)
    """

    user_id = test_user_id

    results_df = pd.read_csv('research/hybrid/testing_hybrid_results.csv')

    print("results_df")
    print(results_df)

    cofficient_columns = ['coefficient_tfidf', 'coefficient_word2vec', 'coefficient_doc2vec']

    results_df[cofficient_columns] = (results_df[cofficient_columns] - results_df[cofficient_columns].mean()) \
                    / results_df[cofficient_columns].std()
    print("normalized_df:")
    print(results_df)
    results_df['coefficient'] = results_df.sum(axis=1)

    recommender_methods = RecommenderMethods()
    df_posts_categories = recommender_methods.get_posts_categories_dataframe()

    print("results_df:")
    print(results_df)

    results_df = results_df.merge(df_posts_categories, left_on='slug', right_on='slug')
    print("results_df after merge")
    print(results_df)

    recommend_methods = RecommenderMethods()
    user_categories = recommend_methods.get_user_categories(user_id)
    print("Categories for user " + str(user_id))
    print(user_categories)
    user_categories_list = user_categories['category_slug'].values.tolist()
    print("user_categories_list:")
    print(user_categories_list)

    results_df.coefficient = np.where(
        results_df["category_slug"].isin(user_categories_list),
                                      results_df.coefficient * 1.5,
                                      results_df.coefficient)

    results_df = results_df.set_index('slug')
    results_df = results_df.sort_values(by='coefficient', ascending=False)
    results_df = results_df['coefficient']
    results_df = results_df.rename_axis('slug').reset_index()

    hybrid_recommended_json = results_df.to_json(orient='records')
    parsed = json.loads(hybrid_recommended_json)
    hybrid_recommended_json = json.dumps(parsed)
    print(hybrid_recommended_json)


if __name__ == "__main__": main()
