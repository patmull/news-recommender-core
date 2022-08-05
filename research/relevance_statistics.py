import ast
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, balanced_accuracy_score, confusion_matrix, \
    dcg_score, f1_score, jaccard_score, ndcg_score, precision_recall_curve, top_k_accuracy_score

import evaluation_results


def try_statistics():
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1])
    y_scores = np.array([0.50562590360641, 0.49579480290413, 0.494285851717, 0.48409512639046, 0.48319244384766, 0.47853890061378, 0.47748681902885, 0.47595044970512, 0.47588595747948, 0.47272825241089, 0.46808642148972, 0.46751618385315, 0.46486243605614, 0.46478125452995, 0.46305647492409, 0.46148332953453, 0.45954248309135, 0.45872023701668, 0.45820289850235, 0.45669832825661])
    print("AVERAGE PRECISION:")
    y_pred = np.full((1, 20), 1)[0]
    print(average_precision_score(y_true, y_scores))
    print("PRECISION SCORE:")
    print(precision_score(y_true, y_pred, average='weighted'))
    print("BALANCED_ACCURACY:")
    print(balanced_accuracy_score(y_true, y_pred))
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_true, y_pred))

    true_relevance = y_true
    scores = y_scores
    print("DCG SCORE:")
    print(dcg_score([true_relevance], [y_pred]))
    print("DCG AT K=5:")
    print(dcg_score([true_relevance], [y_pred], k=5))

    print("F1-SCORE:")
    print(f1_score(y_true, y_pred, average='weighted'))

    print("JACCARD SCORE:")
    print(jaccard_score(y_true, y_pred))

    print("NDCG:")
    print(ndcg_score([true_relevance], [y_pred]))
    print("NDCG AT K:")
    print(ndcg_score([true_relevance], [y_pred], k=5))

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    print("PRECISION:")
    print(precision)
    print("RECALL:")
    print(recall)
    print("THRESHOLD:")
    print(thresholds)

    print("PRECISION SCORE:")
    print(precision_score(y_true, y_pred, average=None))

    print("TOP k=5 ACCURACY:")
    print(top_k_accuracy_score(y_true, scores, k=2))


def model_ap():
    evaluation_results_df = evaluation_results.get_results_dataframe()
    print(evaluation_results_df.head(10).to_string())
    dict_of_model_stats = {}
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df['model_name']])
    
    list_of_aps = []
    list_of_aps.append([average_precision_score(x['relevance'], x['coefficient'])
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])
    print(list_of_models)

    print(list_of_aps)

    dict_of_model_stats = {'ap': list_of_aps[0], 'model_name': list_of_models[0]}
    print(dict_of_model_stats)

    model_ap_results = pd.DataFrame.from_dict(dict_of_model_stats)
    """
    list_of_aps.append([print(x['relevance'])
                        for x in evaluation_results_df['results_part_2']])
    """
    return model_ap_results[['model_name', 'ap']].groupby(['model_name']).mean()


def models_complete_statistics():
    evaluation_results_df = evaluation_results.get_results_dataframe()
    print(evaluation_results_df.head(10).to_string())
    dict_of_model_stats = {}
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df['model_name']])
    print(list_of_models)

    print("AVERAGE PRECISION:")
    list_of_aps = []
    list_of_aps.append([average_precision_score(x['relevance'], x['coefficient'])
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])

    print(list_of_aps)

    y_pred = np.full((1, 20), 1)[0]
    print("WEIGHTED PRECISION SCORE:")
    list_of_ps = []
    list_of_ps.append([precision_score(x['relevance'], y_pred, average='weighted')
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])
    print(list_of_ps)

    print("BALANCED_ACCURACY:")
    list_of_balanced_accuracies = []
    list_of_balanced_accuracies.append([balanced_accuracy_score(x['relevance'], y_pred)
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])
    print(list_of_balanced_accuracies)
    """
    true_relevance = y_true
    scores = y_scores
    """
    print("DCG AT K:")
    list_of_dcgs = []
    list_of_dcgs.append([dcg_score([x['relevance']], [y_pred])
                                        if len(x['relevance']) == len(x['coefficient'])
                                        else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                                        for x in evaluation_results_df['results_part_2']])
    print(list_of_dcgs)

    print("DCG AT K=5:")
    list_of_dcg_at_k = []
    list_of_dcg_at_k.append([dcg_score([x['relevance']], [y_pred], k=5)
                                        if len(x['relevance']) == len(x['coefficient'])
                                        else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                                        for x in evaluation_results_df['results_part_2']])
    print(list_of_dcg_at_k)

    print("F1-SCORE (WEIGHTED AVERAGE):")
    list_of_f1_score = []
    list_of_f1_score.append([f1_score(x['relevance'], y_pred, average='weighted')
                                        if len(x['relevance']) == len(x['coefficient'])
                                        else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                                        for x in evaluation_results_df['results_part_2']])
    print(list_of_f1_score)

    print("NDCG:")
    list_of_ndcgs = []
    list_of_ndcgs.append([ndcg_score([x['relevance']], [y_pred])
                             if len(x['relevance']) == len(x['coefficient'])
                             else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                             for x in evaluation_results_df['results_part_2']])
    print(list_of_ndcgs)

    print("NDCG AT 5:")
    list_of_ndcgs_at_k = []
    list_of_ndcgs_at_k.append([ndcg_score([x['relevance']], [y_pred], k=5)
                          if len(x['relevance']) == len(x['coefficient'])
                          else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                          for x in evaluation_results_df['results_part_2']])
    print(list_of_ndcgs_at_k)

    print("PRECISION, RECALL, THRESHOLDS:")
    list_of_precisions = []
    list_of_recalls = []
    list_of_thresholds = []
    list_of_precisions, list_of_recalls, list_of_thresholds = zip(*(precision_recall_curve(x['relevance'], x['coefficient'])
                               if len(x['relevance']) == len(x['coefficient'])
                               else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                               for x in evaluation_results_df['results_part_2']))
    print(list_of_precisions)
    print(list_of_recalls)
    print(list_of_thresholds)

    list_of_precisions = []
    print("PRECISION SCORE:")
    list_of_precisions.append([precision_score(x['relevance'], y_pred, average=None)
                               if len(x['relevance']) == len(x['coefficient'])
                               else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                               for x in evaluation_results_df['results_part_2']])
    print(list_of_precisions)
    """
    list_of_top_k_accuracy = []
    print("TOP k=5 ACCURACY:")
    list_of_top_k_accuracy.append([top_k_accuracy_score(x['relevance'], x['coefficient'], k=2)
                               if len(x['relevance']) == len(x['coefficient'])
                               else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                               for x in evaluation_results_df['results_part_2']])
    print(list_of_top_k_accuracy)
    """

    # completation of results
    # TODO: Add other columns
    dict_of_model_stats = {'ap': list_of_aps[0], 'model_name': list_of_models[0]}
    print(dict_of_model_stats)

    model_results = pd.DataFrame.from_dict(dict_of_model_stats, orient='index').transpose()
    
    return model_results


def print_confusion_matrix():
    evaluation_results_df = evaluation_results.get_results_dataframe()
    print(evaluation_results_df.head(10).to_string())
    dict_of_model_stats = {}
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df['model_name']])
    print(list_of_models)

    print("AVERAGE PRECISION:")
    list_of_confusion_matrices = []
    list_of_confusion_matrices.append([confusion_matrix(x['relevance'], x['coefficient'])
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])

    print("CONFUSION MATRIX:")


warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
stats = models_complete_statistics()
print(stats)