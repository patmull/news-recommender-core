import json

from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import get_most_similar_by_hybrid

test_user_id = 999999

recommended = get_most_similar_by_hybrid(user_id=test_user_id, posts_to_compare=None,
                                         list_of_methods=None)
recommended_loaded = json.loads(recommended)
assert True
print(recommended_loaded)