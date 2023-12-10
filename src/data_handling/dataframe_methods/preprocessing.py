from src.prefillers.preprocessing.czech_preprocessing import preprocess
from src.recommender_core.data_handling.data_queries import RecommenderMethods


def preprocess_single_post_find_by_slug(slug, supplied_json=False):
    recommender_methods = RecommenderMethods()
    post_dataframe = recommender_methods.find_post_by_slug(slug)
    post_dataframe["title"] = post_dataframe["title"].map(lambda s: preprocess(s))
    post_dataframe["excerpt"] = post_dataframe["excerpt"].map(lambda s: preprocess(s))
    if supplied_json is False:
        return post_dataframe
    else:
        return post_dataframe.to_json()
