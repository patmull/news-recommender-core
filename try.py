from datetime import datetime

from src.recommender_core.data_handling.data_manipulation import get_redis_connection

"""
database = DatabaseMethods()
database.connect()
user_categories_thumbs_df = database.get_posts_users_categories_thumbs()
assert isinstance(user_categories_thumbs_df, pd.DataFrame)
THUMBS_COLUMNS_NEEDED = ['thumbs_values', 'thumbs_created_at', 'all_features_preprocessed', 'full_text']
assert THUMBS_COLUMNS_NEEDED in user_categories_thumbs_df.columns
assert len(user_categories_thumbs_df.index) > 0  # assert there are rows in dataframe

database.disconnect()
"""

# TODO: Add test user who will have some thumbs from posts that are already prefilled...
# (Currently there are no thumb rated posts that are prefilled)
"""
svm = Classifier()
svm.predict_relevance_for_user(use_only_sample_of=20, user_id=431, relevance_by='thumbs')
"""
# TODO: Unit test bad input handling
"""
svm = Classifier()
svm.predict_relevance_for_user(use_only_sample_of=20, user_id=431, relevance_by='stars')
svm.predict_relevance_for_user(use_only_sample_of=20, user_id=431, relevance_by='thumbs')
"""

"""
r = get_redis_connection()
now = datetime.now()
test_value = 'test_' + str(now.strftime("%m/%d/%Y %H:%M:%S"))
r.set('test_pair', test_value)
assert r.get('test_pair').decode() == test_value
r.delete("posts_by_pred_ratings_user_371")
test_value = "vytvorili-prvni-rib-eye-steak-ze-zkumavky-chutna-jako-prave-maso"
test_user = "posts_by_pred_ratings_user_371"
res = r.sadd(test_value, test_user)
print(res)
print(r.smembers(test_value))
"""
