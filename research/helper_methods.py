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


def extract_and_sort_columns_from_results(target_model_variant, sort_by='category_title', target_stats=None):
    df_queries = pd.read_csv('save_relevance_results_by_queries.csv')
    df_queries = df_queries.loc[df_queries['model_variant'].isin(target_model_variant)]
    df_queries = df_queries.sort_values(by=sort_by)
    # df_queries = df_queries.drop('post_slug')
    target_csv_path = 'filtered_results.csv'
    df_queries.to_csv(target_csv_path, index=False) # Set to false to get rid of "Unnamed: 0" column
    if target_stats is not None:
        target_csv_path = 'transposed_and_filtered_results.csv'
        res = df_queries.pivot_table(index=['slug'], columns='model_variant',
                             values=target_stats, aggfunc='first').reset_index()
        print(res.to_string())
        res.to_csv(target_csv_path)


extract_and_sort_columns_from_results(target_model_variant=['word2vec-eval-2', 'tfidf-full-text'], target_stats=['AP','DCG'])
# random_row_remover()