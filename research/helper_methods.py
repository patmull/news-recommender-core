import os

import numpy as np
import pandas as pd

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def random_row_remover(file_full_results=None, path_to_csv=None, remove_n=234):
    """
    Removing random rows when results of Doc2Vec needs to be compared with other results of the same size.

    :param file_full_results: string path to CSV of results. If None, then default path to Doc2Vec results CSV is used.
    :param path_to_csv: string path to the target CSV of results. If None,
    then default path to target Doc2Vec results CSV is used.
    :param remove_n: number of rows to be deleted
    :return:
    """
    if file_full_results is None:
        file_full_results = "doc2vec/evaluation/idnes/doc2vec_tuning_results_random_search_full.csv"
    path_to_df = __location__ + file_full_results
    df = pd.read_csv(path_to_df)
    drop_indices = np.random.choice(df.index, remove_n, replace=False)
    df_subset = df.drop(drop_indices)
    print("df_subset:")
    print(df_subset)
    if path_to_csv is None:
        path_to_csv = "doc2vec/evaluation/idnes/doc2vec_tuning_results_random_search.csv"
    path_to_df = __location__ + path_to_csv
    df_subset.to_csv(path_to_df)


def extract_and_sort_columns_from_results(target_model_variant, sort_by='category_title', target_stats=None,
                                          path_to_results=None):
    """
    Method for filtering columns of target model variants and sorting it by the selected column.
    Saves the exzracted results to filtered_results.csv file.

    :param target_model_variant: list of strings of column names
    :param sort_by: string of column according to which we should sort
    :param target_stats: string of which statistics column from the CSV to use, i.e. AP, DCG...
    :return:
    """
    if path_to_results is None:
        path_to_results = 'save_relevance_results_by_queries.csv'
    df_queries = pd.read_csv(path_to_results)
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