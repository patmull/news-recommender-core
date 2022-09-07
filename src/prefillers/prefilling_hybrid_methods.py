from recommender_core.data_handling.data_queries import RecommenderMethods
from recommender_core.recommender_algorithms.hybrid.classifier import Classifier


def predict_ratings_for_all_users():
    recommender_methods = RecommenderMethods()
    all_users_df = recommender_methods.get_users_dataframe()
    classifier = Classifier()
    for user_row in zip(*all_users_df.to_dict("list").values()):
        print(user_row)
        classifier.predict_ratings(user_id=user_row)
