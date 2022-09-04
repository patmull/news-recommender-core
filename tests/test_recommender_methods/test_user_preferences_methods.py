import os.path
import random
import urllib.request
import certifi
import pytest


# Run with:
# python -m pytest .\tests\test_user_preferences_methods.py::test_user_keywords -rP
from src.core.data_handling.data_queries import RecommenderMethods
from src.core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.core.recommender_algorithms.user_based_algorithms.user_based_recommendation import UserBasedRecommendation


# py.test tests/test_recommender_methods/test_user_preferences_methods.py -k 'test_user_keyword_bad_input'
@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None
])
def test_user_keyword_bad_input(tested_input):

    with pytest.raises(ValueError):
        tfidf = TfIdf()
        tfidf.keyword_based_comparison(tested_input)


# TODO:
def test_user_categories():
    user_based_recommendation = UserBasedRecommendation()
    recommender_methods = RecommenderMethods()
    # TODO: Repair Error
    users = recommender_methods.get_users_dataframe()
    print("users:")
    print(users)
    list_of_user_ids = users['id'].to_list()
    random_position = random.randrange(len(list_of_user_ids))
    random_id = list_of_user_ids[random_position]
    num_of_recommended_posts = 5
    recommendations = user_based_recommendation.load_recommended_posts_for_user(random_id, num_of_recommended_posts)
    print("Recommendations:")
    print(recommendations)
    assert type(recommendations) is dict
    assert len(recommendations) > 0
    assert type(recommendations['columns']) is list
