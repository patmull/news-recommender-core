from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


def simple_example():
    text = ["London Paris London", "Paris Paris London"]
    cv = CountVectorizer(analyzer='word', min_df=10, stop_words='czech', lowercase=True,
                         token_pattern='[a-zA-Z0-9]{3,}')
    print(text)
    print(cv)


def cosine_similarity_n_space(m1, m2=None, batch_size=100):
    assert m1.shape[1] == m2.shape[1] and isinstance(batch_size, int) == True

    ret: Any = np.ndarray((m1.shape[0], m2.shape[0]))  # Added Any due to MyPy typing warning

    batches = m1.shape[0] // batch_size

    if m1.shape[0] % batch_size != 0:
        batches = batches + 1

    for row_i in range(0, batches):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2)
        ret[start: end] = sim

    return ret


class CosineTransformer:

    def __init__(self):
        self.cosine_sim_df = pd.DataFrame()
        self.cosine_sim = None
        self.count_matrix = None
        self.similar_articles = None
        self.sorted_similar_articles = None
        self.database = DatabaseMethods()
        self.posts_df = pd.DataFrame()

    def get_cosine_sim_use_own_matrix(self, own_tfidf_matrix, df):
        own_tfidf_matrix_csr = sparse.csr_matrix(own_tfidf_matrix.astype(dtype=np.float16)).astype(dtype=np.float16)
        cosine_sim = cosine_similarity_n_space(own_tfidf_matrix_csr, own_tfidf_matrix_csr)
        cosine_sim_df = pd.DataFrame(cosine_sim, index=df['slug'],
                                     columns=df['slug'])  # finding original record of post belonging to slug
        del cosine_sim
        self.cosine_sim_df = cosine_sim_df
        return self.cosine_sim_df
