import json

from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import get_most_similar_by_hybrid
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    Classifier

svm = Classifier()
print(svm.predict_relevance_for_user(user_id=431, relevance_by='thumbs'))
