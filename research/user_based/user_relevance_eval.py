import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, balanced_accuracy_score, confusion_matrix, \
    dcg_score, f1_score, jaccard_score, ndcg_score, precision_recall_curve, top_k_accuracy_score

from src.recommender_core.data_handling.model_methods.user_methods import UserMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.user_evaluation_results import \
    get_user_evaluation_results_dataframe


def user_relevance_asessment():
    user_methods = UserMethods()
    user_id_list = user_methods.get_all_users()['id'].tolist()
    precision_score_weighted_list = []
    dcg_score_list = []
    dcg_score_at_k_list = []
    f1_score_list = []
    jaccard_score_list = []
    ndcg_score_list = []
    ndcg_score_at_k_list = []
    balanced_accuracy_score_list = []
    precision_score_list = []
    N_list = []

    sections = get_user_evaluation_results_dataframe()['method_section'].unique().tolist()

    sections_list = []

    for section in sections:
        user_eval_df = get_user_evaluation_results_dataframe()
        user_eval_df = user_eval_df.loc[user_eval_df['method_section'] == section]
        true_relevance = user_eval_df['value'].tolist()

        if len(true_relevance) > 0:
            pred = np.full((1, len(true_relevance)), 1)[0]

            method_sections_list = user_eval_df['method_section'].tolist()

            print("PRECISION SCORE:")
            precision_score_weighted_list.append(precision_score(true_relevance, pred, average='weighted'))
            print("DCG SCORE:")
            dcg_score_list.append(dcg_score([true_relevance], [pred]))
            print("DCG AT K=5:")
            dcg_score_at_k_list.append(dcg_score([true_relevance], [pred], k=5))
            print("F1-SCORE:")
            f1_score_list.append(f1_score(true_relevance, pred, average='weighted'))
            print("JACCARD SCORE:")
            jaccard_score_list.append(jaccard_score(true_relevance, pred))
            print("NDCG:")
            ndcg_score_list.append(ndcg_score([true_relevance], [pred]))
            print("NDCG AT K:")
            ndcg_score_at_k_list.append(ndcg_score([true_relevance], [pred], k=5))
            print("BALANCED_ACCURACY:")
            balanced_accuracy_score_list.append(balanced_accuracy_score(true_relevance, pred))
            print("PRECISION SCORE:")
            precision_score_list.append(precision_score(true_relevance, pred, average=None))

            print(user_id_list)
            print(len(user_id_list))
            print(method_sections_list)
            print(len(method_sections_list))
            print(precision_score_list)
            print(len(precision_score_list))

            sections_list.append(section)

            print("true_relevance:")
            print(true_relevance)
            N_list.append(len(true_relevance))

    df = pd.DataFrame({
        'precision_score': precision_score_list,
        'balanced_accuracy_score': balanced_accuracy_score_list,
        'dcg_score': dcg_score_list,
        'dcg_score_at_k': dcg_score_at_k_list,
        'f1_score': f1_score_list,
        'jaccard_score': jaccard_score_list,
        'ndcg_score': ndcg_score_list,
        'ndcg_at_k_score': ndcg_score_at_k_list,
        'precision_score_weighted': precision_score_weighted_list,
        'sections_list': sections_list,
        'N': N_list
    })

    output_file = Path("research/user_based/user_eval_results.csv")
    df.to_csv(output_file.as_posix(), header=True)
