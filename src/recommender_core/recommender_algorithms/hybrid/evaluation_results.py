from src.recommender_core.data_handling.data_queries import RecommenderMethods


def get_admin_evaluation_results_dataframe():
    recommender_methods = RecommenderMethods()
    return recommender_methods.get_ranking_evaluation_results_dataframe()  # load_texts posts to dataframe

def get_user_evaluation_results_dataframe():
    recommender_methods = RecommenderMethods()
    return recommender_methods.get_item_evaluation_results_dataframe()  # load_texts posts to dataframe