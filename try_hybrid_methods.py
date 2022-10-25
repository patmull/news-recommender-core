from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import get_most_similar_by_hybrid, \
    precalculate_and_save_sim_matrix_for_all_posts, HybridConstants


def main():
    """
    user_id_for_test = 431

    searched_slug_1 = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    searched_slug_2 = "salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem"
    searched_slug_3 = "sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik"

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]

    start = time.time()
    print(get_most_similar_by_hybrid(user_id_for_test))
    end = time.time()
    print(end - start)"""

    """
    user_id_for_test = 431
    user_methods = UserMethods()
    all_users_df = user_methods.get_users_dataframe()
    classifier = Classifier()
    print("Loading BERT multilingual model...")
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    classifier.predict_relevance_for_user(user_id=user_id_for_test, relevance_by='thumbs', force_retraining=False,
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
    """
    # RATINGS VALUES
    df_posts_users_categories_relevance = recommender_methods \
        .get_posts_users_categories_ratings_df(user_id=user_id,
                                               only_with_bert_vectors=only_with_prefilled_bert_vectors)
    df_posts_users_categories_relevance = df_posts_users_categories_relevance.head(20)
    df_posts_users_categories_relevance \
        .to_csv(Path("tests/testing_datasets/testing_posts_categories_stars_data_for_df.csv"))

    target_variable_name = 'ratings_values'
    predicted_by_stars_redis_key_name = 'stars-ratings'

    classifier = Classifier()

    clf_svc, clf_random_forest, X_validation, y_validation, bert_model \
        = classifier.train_classifiers(df=df_posts_users_categories_relevance,
                                       columns_to_combine=columns_to_combine,
                                       target_variable_name=target_variable_name, user_id=user_id)

    predict_from_vectors(X_unseen_df=X_validation, clf=clf_svc, user_id=user_id,
                         predicted_var_for_redis_key_name=predicted_by_stars_redis_key_name,
                         bert_model=bert_model, col_to_combine=columns_to_combine,
                         save_testing_csv=True)
    """
    """
    classifier = Classifier()
    classifier.predict_relevance_for_user(use_only_sample_of=20, user_id=431, relevance_by='stars',
                                          force_retraining=True, save_df_posts_users_categories_relevance=True)
    """
    hybrid_constants = HybridConstants()
    hybrid_constants.set_constants_to_redis()
    """
    user_id_for_test = 431
    precalculate_and_save_sim_matrix_for_all_posts()
    print("=====================================")
    print("PRECALCULATION DONE")
    print("=====================================")
    print(get_most_similar_by_hybrid(user_id_for_test, load_from_precalc_sim_matrix=True))
    """

if __name__ == "__main__": main()
