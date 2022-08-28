import os.path
import random
import urllib.request
import certifi

from src.content_based_algorithms.tfidf import TfIdf
from src.data_handling.data_queries import RecommenderMethods

# Run with:
# python -m pytest .\tests\test_user_preferences_methods.py::test_user_keywords -rP
from src.user_based_recommendation import UserBasedRecommendation


def test_user_keywords():

    path_to_words_file = "tests/datasets/english_words.txt"
    if not os.path.exists(path_to_words_file):
        word_url = "https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
        response = urllib.request.urlopen(word_url, cafile=certifi.where())
        long_txt = response.read().decode()
        words = long_txt.splitlines()
    else:
        with open(path_to_words_file, "r") as word_list:
            words = word_list.read().split('\n')
    # words = long_txt.splitlines()s)
    random_phrase = words[random.randint(0, len(words))] + ' ' + words[random.randint(0, len(words))]

    # TODO: Check whether some currently working library exists
    # translate = Translation()
    # translated_random_phrase = translate.convert('en', 'cs', random_phrase)
    # print("translated_random_phrase:")
    # print(translated_random_phrase)

    tfidf = TfIdf()
    json_keywords = {"keywords": random_phrase}
    # input_json_keywords = json.dumps(json_keywords)
    similar_posts = tfidf.keyword_based_comparison(json_keywords["keywords"])
    assert type(similar_posts) is list
    assert len(similar_posts) > 0
    print(type(similar_posts[0]['slug']))
    assert type(similar_posts[0]['slug']) is str
    assert type(similar_posts[0]['coefficient']) is float
    assert len(similar_posts) > 0


# TODO:
def test_user_categories():
    user_based_recommendation = UserBasedRecommendation()
    recommender_methods = RecommenderMethods()
    # TODO: Repair Error
    recommender_methods.database.connect()
    users = recommender_methods.get_users_dataframe()
    recommender_methods.database.disconnect()
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
