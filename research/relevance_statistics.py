import random
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, balanced_accuracy_score, confusion_matrix, \
    dcg_score, f1_score, jaccard_score, ndcg_score, precision_recall_curve, top_k_accuracy_score
import seaborn as sns

from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.user_evaluation_results import \
    get_admin_evaluation_results_dataframe

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

def try_statistics():
    """
    Experimenting with statistics. This is only playground for trying how statistics methods from Numpy works.

    :return:
    """
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
    """
    Computing average precision of the given content-based model.

    :param investigate_by:
    :return:
    """
    evaluation_results_df = get_admin_evaluation_results_dataframe()
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
    """
    Calculating the model variant's average precision.

    :param variant:
    :return:
    """

    evaluation_results_df = get_admin_evaluation_results_dataframe()
    print(evaluation_results_df.head(10).to_string())

    if variant is not None:
        evaluation_results_df = evaluation_results_df.loc[evaluation_results_df['model_variant'] == variant]

    dict_of_model_stats = {}
    list_of_aps = []
    list_of_aps = []
    list_of_aps.append([average_precision_score(x['relevance'], x['coefficient'])
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])
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


def models_complete_statistics(investigate_by, k=5, save_results_for_every_item=False, crop_by_date=False, last_n_by_date=None):
    global list_of_slugs, list_of_created_at
    evaluation_results_df = get_admin_evaluation_results_dataframe()

    if crop_by_date:
        if last_n_by_date is not None:
            evaluation_results_df = evaluation_results_df.sort_values(by=['created_at'], ascending=False)
            evaluation_results_df = evaluation_results_df.head(last_n_by_date)
        else:
            ValueError("When cropping by date, the 'date' argument needs to be set, otherwise it will show the date but not"
                       "crop the date")
    print("evaluation_results_df")
    print(evaluation_results_df.head(10).to_string())
    dict_of_model_stats = {}
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df[investigate_by]])
    if save_results_for_every_item:
        list_of_slugs = []
        list_of_slugs.append([x for x in evaluation_results_df['query_slug']])

    if crop_by_date:
        list_of_created_at = []
        list_of_created_at.append([x for x in evaluation_results_df['created_at']])
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
    if save_results_for_every_item is True:
        dict_of_model_stats = {'AP': list_of_aps[0], 'precision_score': list_of_ps[0],
                               'balanced_accuracies': list_of_balanced_accuracies[0],
                               'DCG': list_of_dcgs[0], 'DCG_AT_' + str(k): list_of_dcg_at_k[0],
                               'F1-SCORE': list_of_f1_score[0], 'NDCG': list_of_ndcgs[0],
                               'NDCG_AT_' + str(k): list_of_ndcgs_at_k[0],
                               investigate_by: list_of_models[0], 'slug': list_of_slugs[0],
                               'created_at': list_of_created_at[0]}
    else:
        dict_of_model_stats = {'AP': list_of_aps[0], 'precision_score': list_of_ps[0],
                               'balanced_accuracies': list_of_balanced_accuracies[0],
                               'DCG': list_of_dcgs[0], 'DCG_AT_' + str(k): list_of_dcg_at_k[0],
                               'F1-SCORE': list_of_f1_score[0], 'NDCG': list_of_ndcgs[0],
                               'NDCG_AT_' + str(k): list_of_ndcgs_at_k[0],
                               investigate_by: list_of_models[0]}

    print(dict_of_model_stats)

    model_results = pd.DataFrame.from_dict(dict_of_model_stats, orient='index').transpose()

    model_results = model_results.fillna(0)

    print("Model results:")
    print(model_results.to_string())
    # joining with rvaluated posts slugs
    if save_results_for_every_item is True:
        print("Saving also results for every item.")
        path_to_resuts = 'save_relevance_results_by_queries.csv'
        recommender_methods = RecommenderMethods()
        posts_categories__ratings_df = recommender_methods.join_posts_ratings_categories()
        print("model_results.columns")
        print(posts_categories__ratings_df.columns)
        categories_df = posts_categories__ratings_df[['category_title', 'post_slug']]
        model_results = pd.merge(model_results, categories_df, left_on='slug', right_on='post_slug', how='left')
       #  model_results_joined_with_category.to_csv(path_to_resuts, index=False)
        print("model_results.columns")
        print(model_results.columns)
        print(model_results.to_string())
        grouped_results = model_results.groupby(['AP', 'model_variant', 'slug', 'created_at',
       'category_title']).mean().reset_index()
       # full_results = pd.merge(grouped_results, categories_df, left_on='slug', right_on='post_slug', how='left')

        print("full_results.columns")
        print(grouped_results.columns)
        grouped_results = grouped_results.sort_values(by=['created_at', 'category_title'])
        transposed_results = grouped_results.pivot_table('AP', ['slug', 'category_title'], 'model_variant')
        return transposed_results
    else:
        return model_results.groupby([investigate_by]).mean().reset_index()


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
    """
    Printing confusion matrix of evaluation results

    :return:
    """
    evaluation_results_df = get_admin_evaluation_results_dataframe()
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


def print_model_variant_relevances():
    stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=False)
    print("Means of model's metrics:")
    print(stats.to_string())


def save_model_variant_relevances(crop_by_date=False, last_n_by_date=None):
    """
    Saves the model variant to unique file with hash added on the suffix.

    :param crop_by_date: Crop the results by date.
    :param last_n_by_date: Select only the last N rows.

    :return:
    """
    stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=False,
                                       crop_by_date=crop_by_date, last_n_by_date=last_n_by_date)
    print("Means of model's metrics:")
    print(stats.to_string())
    print("Saving CSV with user evaluation results...")
    stats = stats.round(2)
    hash = random.getrandbits(128)
    path_for_saving = "research/word2vec/evaluation/word2vec_tuning_relevance_results" + str(hash) + ".csv"
    stats.to_csv(path_for_saving)
    print("Results saved.")


def print_model_variant_relevances_for_each_article(save_to_csv=False, crop_by_date=False):
    """
    Printing model variants for each article to console.

    :param save_to_csv:
    :param crop_by_date:
    :return:
    """
    stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=True, crop_by_date=crop_by_date)
    print("Means of model's metrics:")
    print(stats.to_string())
    if save_to_csv is True:
        stats = stats.round(2)
        path_for_saving = "research/word2vec/evaluation/cswiki/word2vec_tuning_relevance_results_by_each_article.csv"
        stats.to_csv(path_for_saving)
        print("Results saved.")


def print_overall_model_relevances():
    """
    Method for running the model statistics.

    :return:
    """
    stats = models_complete_statistics(investigate_by='model_name', save_results_for_every_item=True)
    print("Means of model's metrics:")
    print(stats.to_string())


def print_confusion_matrix():
    """
    Printing the confusion matrix to console.

    :return:
    """
    print(show_confusion_matrix())
