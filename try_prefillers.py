from src.prefillers.prefilling_all import run_prefilling


"""
classifier = Classifier()
try:
    bert = spacy_sentence_bert.load_doc2vec_model('xx_stsb_xlm_r_multilingual')
    classifier.predict_relevance_for_user(user_id=431, relevance_by='stars', only_with_prefilled_bert_vectors=False,
                                          bert_model=bert, use_only_sample_of=20)

    classifier.predict_relevance_for_user(user_id=431, relevance_by='stars', only_with_prefilled_bert_vectors=False,
                                          bert_model=bert, use_only_sample_of=20, force_retraining=True)
    r = get_redis_connection()
    print(r.smembers('posts_by_pred_ratings_user_431'))
except ValueError:
    print("Value error had occurred when trying to get relevant thumbs for user. Skipping this user.")
    # classifier.predict_relevance_for_user(user_id=431, relevance_by='ratings')

"""
"""
# noinspection PyPep8Naming
DB_USER = os.environ.get('DB_RECOMMENDER_USER')
# noinspection PyPep8Naming
DB_PASSWORD = os.environ.get('DB_RECOMMENDER_PASSWORD')
# noinspection PyPep8Naming
DB_HOST = os.environ.get('DB_RECOMMENDER_HOST')
# noinspection PyPep8Naming
DB_NAME = os.environ.get('DB_RECOMMENDER_NAME')

assert type(DB_USER) is str
assert type(DB_PASSWORD) is str
assert type(DB_HOST) is str
assert type(DB_NAME) is str

assert bool(DB_USER) is True  # not empty
assert bool(DB_PASSWORD) is True
assert bool(DB_HOST) is True
assert bool(DB_NAME) is True
databse_methods = DatabaseMethods()
databse_methods.connect()
mockconnect.assert_called()
assert 1 == mockconnect.call_count
assert mockconnect.call_args_list[0] == call(user=DB_USER, password=DB_PASSWORD,
                                             host=DB_HOST, dbname=DB_NAME)
"""

# run_prefilling_collaborative(test_run=True)
run_prefilling(skip_cache_refresh=True)
