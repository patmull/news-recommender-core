from app import check_if_cache_exists_and_fresh
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.recommender_core.data_handling.data_queries import RecommenderMethods

"""
database_methods = DatabaseMethods()
df = database_methods.get_posts_dataframe_from_cache()
print(df.head(10))
print(df.columns)
print(df)
"""

if not check_if_cache_exists_and_fresh():
    print("Posts cache file does not exists or older than 1 day.")
    print("Creating posts cache file...")
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)