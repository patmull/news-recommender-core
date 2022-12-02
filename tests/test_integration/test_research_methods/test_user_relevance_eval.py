import pandas as pd

from src.recommender_core.data_handling.data_queries import RecommenderMethods


def test_get_user_evaluation_results_dataframe():
    recommender_methods = RecommenderMethods()
    results_df = recommender_methods.get_item_evaluation_results_dataframe()  # load_texts posts to dataframe
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df.index) > 0
