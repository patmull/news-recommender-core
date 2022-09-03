
from src.content_based_algorithms.doc2vec import Doc2VecClass
from src.data_handling.data_queries import RecommenderMethods

recommender_methods = RecommenderMethods()
posts_df = recommender_methods.database.get_posts_dataframe_from_sql()
posts_df = posts_df.sort_values(by="created_at", ascending=False)
tested_slug = posts_df['slug'].iloc[0]
doc2vec = Doc2VecClass()
print("tested_slug:")
print(tested_slug)
print(doc2vec.get_similar_doc2vec(tested_slug))
print(doc2vec.get_similar_doc2vec(tested_slug, full_text=True))
