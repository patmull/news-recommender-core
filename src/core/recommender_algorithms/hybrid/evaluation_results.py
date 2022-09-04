from src.core.data_handling.data_queries import RecommenderMethods


def get_results_dataframe():
    recommender_methods = RecommenderMethods()
    return recommender_methods.get_evaluation_results_dataframe()  # load_texts posts to dataframe
