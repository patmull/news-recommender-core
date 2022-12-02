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

# TODO: Add test user who will have some thumbs from posts that have already prefilled BERT vectors...
# (Currently there are no thumb rated posts that are prefilled)
"""
svm = Classifier()
svm.predict_relevance_for_user(use_only_sample_of=20, user_id=431, relevance_by='thumbs')
"""
# TODO: Unit test bad input handling
