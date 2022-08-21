from content_based_algorithms.data_queries import RecommenderMethods


def get_results_dataframe():
    recommenderMethods = RecommenderMethods()
    return recommenderMethods.get_results_dataframe()  # load_texts posts to dataframe

