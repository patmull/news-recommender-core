import os
import sys

import numpy as np
import pandas as pd

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def random_row_remover(remove_n=234):
    doc2vec_file = "\\doc2vec\\evaluation\\idnes\\doc2vec_tuning_results_random_search_full.csv"
    path_to_df = __location__ + doc2vec_file
    df = pd.read_csv(path_to_df)
    drop_indices = np.random.choice(df.index, remove_n, replace=False)
    df_subset = df.drop(drop_indices)
    print("df_subset:")
    print(df_subset)
    path_to_csv = "\\doc2vec\\evaluation\\idnes\\doc2vec_tuning_results_random_search.csv"
    path_to_df = __location__ + path_to_csv
    df_subset.to_csv(path_to_df)


random_row_remover()