import numpy as np
import pytest

from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods

# RUN WITH:
# python -m pytest .tests\test_recommender_methods\test_content_based_methods.py::TestClass::test_method


# py.test tests/test_recommender_methods/test_user_preferences_methods.py -k 'test_user_keyword_bad_input'
@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None
])
def test_user_keyword_bad_input(tested_input):

    with pytest.raises(ValueError):
        doc2vec = Doc2VecClass()
        doc2vec.get_vector_representation(tested_input)


def test_doc2vec_vector_representation():
    database = DatabaseMethods()
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