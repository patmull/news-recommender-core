

"""
recommender_methods = RecommenderMethods()
posts_df = recommender_methods.database.get_posts_dataframe_from_sql()
posts_df = posts_df.sort_values(by="created_at", ascending=False)
tested_slug = posts_df['slug'].iloc[0]
doc2vec = Doc2VecClass()
print("tested_slug:")
print(tested_slug)
print(doc2vec.get_similar_doc2vec(tested_slug))
print(doc2vec.get_similar_doc2vec(tested_slug, full_text=True))
"""

from src.recommender_core.data_handling.data_manipulation import RedisMethods
from src.recommender_core.recommender_algorithms.hybrid.classifier import SVM
from datetime import datetime
"""
database = Database()
database.connect()
print(database.get_posts_users_categories_thumbs().to_string())
database.disconnect()
"""


svm = SVM()
svm.predict_ratings(show_only_sample_of=20, user_id=371)


"""
redis_methods = RedisMethods()
r = redis_methods.get_redis_connection()
now = datetime.now()
test_value = 'test_' + str(now.strftime("%m/%d/%Y %H:%M:%S"))
r.set('test_pair', test_value)
assert r.get('test_pair').decode() == test_value
"""