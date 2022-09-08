
# fill_bert_vector_representation()
from prefillers.prefilling_hybrid_methods import predict_ratings_for_all_users_store_to_redis

# predict_ratings_for_all_users_store_to_redis()
from recommender_core.recommender_algorithms.hybrid.classifier import Classifier

classifier = Classifier()
classifier.predict_ratings_for_user(user_id=431)