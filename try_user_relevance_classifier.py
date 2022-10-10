import spacy_sentence_bert

from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    Classifier

print("Loading BERT multilingual model...")
bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
svm = Classifier()
print(svm.predict_relevance_for_user(user_id=431, relevance_by='thumbs', only_with_prefilled_bert_vectors=False,
                                     bert_model=bert, force_retraining=True))
print(svm.predict_relevance_for_user(user_id=431, relevance_by='stars', only_with_prefilled_bert_vectors=False,
                                     bert_model=bert))
