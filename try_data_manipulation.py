from app import check_if_cache_exists_and_fresh
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods

"""
database_methods = DatabaseMethods()
df = database_methods.get_posts_dataframe_from_cache()
print(df.head(10))
print(df.columns)
print(df)
"""

"""
if not check_if_cache_exists_and_fresh():
    print("Posts cache file does not exists or older than 1 day.")
    print("Creating posts cache file...")
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)
"""
"""
user_methods = UserMethods()

print(user_methods.get_user_dataframe(user_id=371))
"""
database_methods = DatabaseMethods()
database_methods.connect()
database_methods.null_test_user_prefilled_records(241, ['recommended_by_best_rated_by_others_in_user_categories'])
database_methods.disconnect()
