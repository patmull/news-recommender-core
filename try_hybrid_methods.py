import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spacy_sentence_bert

from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.collaboration_based_recommendation import \
    SvdClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import \
    get_similarity_matrix_from_pairs_similarity, select_list_of_posts_for_user, \
    get_most_similar_by_hybrid
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    Classifier, predict_from_vectors


def main():
    """
    test_user_id = 431

    searched_slug_1 = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    searched_slug_2 = "salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem"
    searched_slug_3 = "sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik"

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]

    start = time.time()
    print(get_most_similar_by_hybrid(test_user_id))
    end = time.time()
    print(end - start)"""

    """
    test_user_id = 431
    user_methods = UserMethods()
    all_users_df = user_methods.get_users_dataframe()
    classifier = Classifier()
    print("Loading BERT multilingual model...")
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    classifier.predict_relevance_for_user(user_id=test_user_id, relevance_by='thumbs', force_retraining=False,
                                          bert_model=bert, use_only_sample_of=None, only_with_prefilled_bert_vectors=False,
                                          experiment_mode=False)
    """


    malformed_redis_key_name = "malformed_redis_key_name"
    columns_to_combine = ['category_title', 'all_features_preprocessed', 'full_text']

    recommender_methods = RecommenderMethods()
    user_id = 999999
    only_with_prefilled_bert_vectors = True

    # THUMBS VALUES
    """
    df_posts_users_categories_relevance = recommender_methods \
        .get_posts_users_categories_thumbs_df(user_id=user_id,
                                              only_with_bert_vectors=only_with_prefilled_bert_vectors)
    df_posts_users_categories_relevance = df_posts_users_categories_relevance.head(20)
    df_posts_users_categories_relevance \
        .to_csv(Path("tests/testing_datasets/testing_posts_categories_thumbs_data_for_df.csv"))

    target_variable_name = 'thumbs_values'

    classifier = Classifier()

    print("df_posts_users_categories_relevance")
    print(df_posts_users_categories_relevance)

    clf_svc, clf_random_forest, X_validation, y_validation, bert_model \
        = classifier.train_classifiers(df=df_posts_users_categories_relevance,
                                       columns_to_combine=columns_to_combine,
                                       target_variable_name=target_variable_name, user_id=user_id)

    predict_from_vectors(X_unseen_df=X_validation, clf=clf_svc, user_id=user_id,
                         predicted_var_for_redis_key_name=malformed_redis_key_name,
                         bert_model=bert_model, col_to_combine=columns_to_combine,
                         save_testing_csv=True)
    """

    # RATINGS VALUES
    df_posts_users_categories_relevance = recommender_methods \
        .get_posts_users_categories_ratings_df(user_id=user_id,
                                               only_with_bert_vectors=only_with_prefilled_bert_vectors)
    df_posts_users_categories_relevance = df_posts_users_categories_relevance.head(20)
    df_posts_users_categories_relevance \
        .to_csv(Path("tests/testing_datasets/testing_posts_categories_stars_data_for_df.csv"))

    target_variable_name = 'ratings_values'

    classifier = Classifier()

    clf_svc, clf_random_forest, X_validation, y_validation, bert_model \
        = classifier.train_classifiers(df=df_posts_users_categories_relevance,
                                       columns_to_combine=columns_to_combine,
                                       target_variable_name=target_variable_name, user_id=user_id)

    predict_from_vectors(X_unseen_df=X_validation, clf=clf_svc, user_id=user_id,
                         predicted_var_for_redis_key_name=malformed_redis_key_name,
                         bert_model=bert_model, col_to_combine=columns_to_combine,
                         save_testing_csv=True)


if __name__ == "__main__": main()
