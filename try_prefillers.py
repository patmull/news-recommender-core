# fill_bert_vector_representation()

# predict_ratings_for_all_users_store_to_redis()
import spacy_sentence_bert

from src.recommender_core.data_handling.data_manipulation import RedisMethods
from src.recommender_core.recommender_algorithms.hybrid.classifier import Classifier

classifier = Classifier()
try:
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    """
    classifier.predict_relevance_for_user(user_id=431, relevance_by='stars', only_with_bert_vectors=False,
                                          bert_model=bert, use_only_sample_of=20)
    """
    classifier.predict_relevance_for_user(user_id=431, relevance_by='stars', only_with_bert_vectors=False,
                                          bert_model=bert, use_only_sample_of=20, force_retraining=True)
    redis_methods = RedisMethods()
    r = redis_methods.get_redis_connection()
    print(r.smembers('posts_by_pred_ratings_user_431'))
except ValueError:
    print("Value error had occured when trying to get relevant thumbs for user. Skipping this user.")
# classifier.predict_relevance_for_user(user_id=431, relevance_by='ratings')
