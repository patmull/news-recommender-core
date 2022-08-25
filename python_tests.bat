python -m pytest .\tests\test_data_handling\test_data_queries.py
python -m pytest .\tests\test_preprocessing\test_preprocessing.py
python -m pytest .\tests\test_recommender_methods\test_content_based_methods.py
python -m pytest .\tests\test_recommender_methods\test_hybrid_methods.py
python -m pytest .\tests\test_needed_columns.py
python -m pytest .\tests\test_prefilled_recommendations.py
python -m pytest .\tests\test_user_preferences_methods.py
# TODO: Remove the cached_posts_dataframe_test.pkl programmatically (ideally)