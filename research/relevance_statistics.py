import ast
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, balanced_accuracy_score, confusion_matrix, \
    dcg_score, f1_score, jaccard_score, ndcg_score, precision_recall_curve, top_k_accuracy_score
import seaborn as sns
import evaluation_results
from content_based_algorithms.data_queries import RecommenderMethods

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

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


def model_ap(investigate_by='model_name'):
    evaluation_results_df = evaluation_results.get_results_dataframe()
    print(evaluation_results_df.head(10).to_string())
    dict_of_model_stats = {}
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df[investigate_by]])
    
    list_of_aps = []
    list_of_aps.append([average_precision_score(x['relevance'], x['coefficient'])
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])
    print(list_of_models)
    print(list_of_aps)

    dict_of_model_stats = {'ap': list_of_aps[0], investigate_by: list_of_models[0]}
    print(dict_of_model_stats)

    model_ap_results = pd.DataFrame.from_dict(dict_of_model_stats)
    """
    list_of_aps.append([print(x['relevance'])
                        for x in evaluation_results_df['results_part_2']])
    """
    return model_ap_results[[investigate_by, 'ap']].groupby([investigate_by]).mean()


def model_variant_ap(variant=None):
    evaluation_results_df = evaluation_results.get_results_dataframe()
    print(evaluation_results_df.head(10).to_string())

    if variant is not None:
        evaluation_results_df = evaluation_results_df.loc[evaluation_results_df['model_variant'] == variant]

    dict_of_model_stats = {}
    list_of_aps = []
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df['model_variant']])

    print(list_of_aps)
    print(list_of_models)

    dict_of_model_stats = {'ap': list_of_aps[0], 'model_variant': list_of_models[0]}
    print(dict_of_model_stats)

    model_ap_results = pd.DataFrame.from_dict(dict_of_model_stats)
    """
    list_of_aps.append([print(x['relevance'])
                        for x in evaluation_results_df['results_part_2']])
    """
    return model_ap_results[['model_variant', 'ap']].groupby(['model_variant']).mean()


def models_complete_statistics(investigate_by='model_name', k=5, save_results_for_every_item=False):
    evaluation_results_df = evaluation_results.get_results_dataframe()
    print(evaluation_results_df.head(10).to_string())
    dict_of_model_stats = {}
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df[investigate_by]])
    if save_results_for_every_item:
        list_of_slugs = []
        list_of_slugs.append([x for x in evaluation_results_df['query_slug']])
    print(list_of_models)
    print("evaluation_results_df['results_part_2']")
    print(evaluation_results_df['results_part_2'].to_string())

    print("AVERAGE PRECISION:")
    list_of_aps = []
    list_of_aps.append([average_precision_score(x['relevance'], x['coefficient'], average='weighted')
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])

    print(list_of_aps)

    y_pred = np.full((1, 20), 1)[0]
    print("WEIGHTED PRECISION SCORE:")
    list_of_ps = []
    list_of_ps.append([precision_score(x['relevance'], y_pred, average='macro')
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
    print("DCG:")
    list_of_dcgs = []
    list_of_dcgs.append([dcg_score([x['relevance']], [y_pred])
                                        if len(x['relevance']) == len(x['coefficient'])
                                        else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                                        for x in evaluation_results_df['results_part_2']])
    print(list_of_dcgs)

    print("DCG AT K=5:")
    list_of_dcg_at_k = []
    list_of_dcg_at_k.append([dcg_score([x['relevance']], [y_pred], k=k)
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
    list_of_ndcgs_at_k.append([ndcg_score([x['relevance']], [y_pred], k=k)
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
    list_of_precisions.append([precision_score(x['relevance'], y_pred, average='macro')
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
    dict_of_model_stats = {'AP': list_of_aps[0], 'precision_score': list_of_ps[0],
                           'balanced_accuracies': list_of_balanced_accuracies[0],
                           'DCG': list_of_dcgs[0], 'DCG_AT_' + str(k): list_of_dcg_at_k,
                           'F1-SCORE': list_of_f1_score[0], 'NDCG': list_of_ndcgs[0],
                           'NDCG_AT_' + str(k): list_of_ndcgs_at_k[0],
                           investigate_by: list_of_models[0], 'slug': list_of_slugs[0]}
    print(dict_of_model_stats)

    model_results = pd.DataFrame.from_dict(dict_of_model_stats, orient='index').transpose()

    model_results = model_results.fillna(0)
    print("Model results:")
    print(model_results.to_string())
    if save_results_for_every_item:
        print("Saving also results for every item.")
        path_to_resuts = 'save_relevance_results_by_queries.csv'
        recommender_methods = RecommenderMethods()
        posts_categories__ratings_df = recommender_methods.join_posts_ratings_categories()
        categories_df = posts_categories__ratings_df[['category_title', 'post_slug']]
        model_results_joined_with_category = pd.merge(model_results, categories_df, left_on='slug', right_on='post_slug', how='')
        model_results_joined_with_category.to_csv(path_to_resuts, index=False)

    else:
        model_results = model_results.drop(['slug'])

    return model_results.groupby([investigate_by]).mean()


def plot_confusion_matrix(cm, title):
    plt.tight_layout()

    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title(title)

    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.tight_layout()

    ## Display the visualization of the Confusion Matrix.
    plt.show()


def show_confusion_matrix():
    print("Please be awware that confusion matrix is only ")
    evaluation_results_df = evaluation_results.get_results_dataframe()
    print(evaluation_results_df.head(10).to_string())
    dict_of_model_stats = {}
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df['model_variant']])
    print(list_of_models)

    y_pred = np.full((1, 20), 1)[0]

    list_of_models.append([x for x in evaluation_results_df['model_name']])
    print(list_of_models)

    print("AVERAGE PRECISION:")
    list_of_confusion_matrices = []
    list_of_confusion_matrices.append([confusion_matrix(x['relevance'], y_pred)
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])

    print("CONFUSION MATRIX:")

    print(list_of_confusion_matrices)
    # for cm in list_of_confusion_matrices:
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    cm = list_of_confusion_matrices[0][0]
    list_of_confusion_matrices_selected = []
    for row in list_of_confusion_matrices:
        for item in row:
            print("item")
            print(type(item))
            if item.shape == (2,2):
                item = np.asmatrix(item)
                print("item after convrsion to matrix")
                print(item)
                list_of_confusion_matrices_selected.append(item)

    print("list_of_confusion_matrices_selected:")
    print(list_of_confusion_matrices_selected)
    cm_mean = np.mean(list_of_confusion_matrices_selected, axis=0)
    print(cm_mean)
    plt.figure()
    plot_confusion_matrix(cm_mean, "Confusion matrix")


stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=True)
print("Means of model's metrics:")
print(stats.to_string())
# print(show_confusion_matrix())