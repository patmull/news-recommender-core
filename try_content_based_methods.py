from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf

if __name__ == '__main__':
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)
    tfidf = TfIdf()
    tfidf.save_sparse_matrix(for_hybrid=False)
    print(tfidf.recommend_posts_by_all_features_preprocessed('zaostala-zeme-vubec-kde-hledat-nejlepsi-dovolenou-v-bulharsku'))

    # Sample faulty Word2Vec Idnes 3:
    # 'kviz-znate-starou-prahu-otestujte-sve-znalosti-o-nejhezcim-meste-ceska'