from src.data_handling.data_queries import RecommenderMethods
from src.methods.content_based.tfidf import TfIdf

if __name__ == '__main__':
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)
    tfidf = TfIdf()
    tfidf.save_sparse_matrix(for_hybrid=False)
    tfidf.recommend_posts_by_all_features_preprocessed('zaostala-zeme-vubec-kde-hledat-nejlepsi-dovolenou-v-bulharsku')
