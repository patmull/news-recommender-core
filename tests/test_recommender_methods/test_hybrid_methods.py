import numpy as np

from content_based_algorithms.doc2vec import Doc2VecClass
from data_connection import Database

# RUN WITH:
# python -m pytest .tests\test_recommender_methods\test_content_based_methods.py::TestClass::test_method


# python -m pytest .\tests\test_recommender_methods\test_hybrid_methods.py::test_doc2vec_vector_representation
def test_doc2vec_vector_representation():
    database = Database()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)

    doc2vec = Doc2VecClass()
    doc2vec.load_model()
    vector_representation = doc2vec.get_vector_representation(random_post_slug)

    assert type(vector_representation) is np.ndarray
    assert len(vector_representation) > 0