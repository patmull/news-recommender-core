# fill_bert_vector_representation()

# predict_ratings_for_all_users_store_to_redis()
import spacy_sentence_bert

from src.recommender_core.recommender_algorithms.hybrid.classifier import Classifier

classifier = Classifier()
try:
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    classifier.predict_relevance_for_user(user_id=431, relevance_by='thumbs', only_with_bert_vectors=False,
                                          bert_model=bert)
except ValueError:
    print("Value error occured when trying to get relevant thumbs for user. Skipping"
          "this user.")
# classifier.predict_relevance_for_user(user_id=431, relevance_by='ratings')
