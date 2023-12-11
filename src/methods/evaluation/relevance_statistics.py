import random
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, balanced_accuracy_score, confusion_matrix, \
    dcg_score, f1_score, ndcg_score

from src.data_handling.data_queries import RecommenderMethods
from src.methods.user_based.user_relevance_classifier.user_evaluation_results import \
    get_admin_evaluation_results_dataframe

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


# ***HERE WAS A TRY_STATISTICS, CHECK TO SEE BASIC WORK WITH THE STATISTICS.***
def model_ap(investigate_by='model_name'):
    evaluation_results_df = get_admin_evaluation_results_dataframe()

    list_of_models = [[x for x in evaluation_results_df[investigate_by]]]

    list_of_aps = [[average_precision_score(x['relevance'], x['coefficient'])
                    if len(x['relevance']) == len(x['coefficient'])
                    else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                    for x in evaluation_results_df['results_part_2']]]

    dict_of_model_stats = {'ap': list_of_aps[0], investigate_by: list_of_models[0]}

    model_ap_results = pd.DataFrame.from_dict(dict_of_model_stats)

    return model_ap_results[[investigate_by, 'ap']].groupby([investigate_by]).mean()


def model_variant_ap(variant=None):
    evaluation_results_df = get_admin_evaluation_results_dataframe()

    if variant is not None:
        evaluation_results_df = evaluation_results_df.loc[evaluation_results_df['model_variant'] == variant]

    list_of_aps = [[average_precision_score(x['relevance'], x['coefficient'])
                    if len(x['relevance']) == len(x['coefficient'])
                    else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                    for x in evaluation_results_df['results_part_2']]]
    list_of_models = [[x for x in evaluation_results_df['model_variant']]]
    dict_of_model_stats = {'ap': list_of_aps[0], 'model_variant': list_of_models[0]}
    model_ap_results = pd.DataFrame.from_dict(dict_of_model_stats)

    return model_ap_results[['model_variant', 'ap']].groupby(['model_variant']).mean()


def models_complete_statistics(investigate_by, k=5, save_results_for_every_item=False, crop_by_date=False,
                               last_n_by_date=None):
    evaluation_results_df = get_admin_evaluation_results_dataframe()

    if crop_by_date:
        if last_n_by_date is not None:
            evaluation_results_df = evaluation_results_df.sort_values(by=['created_at'], ascending=False)
            evaluation_results_df = evaluation_results_df.head(last_n_by_date)
        else:
            raise ValueError("When cropping by date, the 'date' argument needs to be set, "
                             "otherwise it will show the date but not crop the date")

    list_of_models = [[x for x in evaluation_results_df[investigate_by]]]
    list_of_slugs = []
    if save_results_for_every_item:
        evaluation_results_df['query_slug'].dropna(inplace=True)
        list_of_slugs.append([x for x in evaluation_results_df['query_slug']])

    list_of_created_at = []
    if crop_by_date:
        list_of_created_at.append([x for x in evaluation_results_df['created_at']])

    list_of_aps = []
    evaluation_results_df['results_part_2'].dropna(inplace=True)

    # AVERAGE PRECISION
    for index, row in evaluation_results_df.iterrows():
        # type(row['results_part_2']:
        results_df = row['results_part_2']
        relevance_dict = dict((k, results_df[k]) for k in ['relevance', 'coefficient']
                              if k in results_df)
        f = len(relevance_dict[next(iter(relevance_dict))])
        if all(len(x) == f for x in relevance_dict.values()):
            try:
                list_of_aps.append(average_precision_score(relevance_dict['relevance'],
                                                           relevance_dict['coefficient'], average='weighted'))
            except TypeError as e:
                # TypeError:
                print(e)
                # Skipping record. Does not have the same number of column
                continue

    # WEIGHTED PRECISION SCORE:
    list_of_ps = [[precision_score(x['relevance'], np.full((1, len(x['relevance'])), 1)[0], average='macro')
                   for x in evaluation_results_df['results_part_2']
                   if None not in x['relevance']]]
    # BALANCED_ACCURACY:
    list_of_balanced_accuracies = [[balanced_accuracy_score(x['relevance'],
                                                            np.full((1, len(x['relevance'])), 1)[0])
                                    for x in evaluation_results_df['results_part_2']
                                    if None not in x['relevance']
                                    ]]
    # DCG:
    list_of_dcgs = [[dcg_score([x['relevance']], [np.full((1, len(x['relevance'])), 1)[0]])
                     for x in evaluation_results_df['results_part_2']
                     if None not in x['relevance']
                     ]]
    # DCG AT K=5:
    list_of_dcg_at_k = [[dcg_score([x['relevance']], [np.full((1, len(x['relevance'])), 1)[0]], k=k)
                         for x in evaluation_results_df['results_part_2']
                         if None not in x['relevance']
                         ]]
    # F1-SCORE (WEIGHTED AVERAGE):
    list_of_f1_score = [[f1_score(x['relevance'], np.full((1, len(x['relevance'])), 1)[0], average='weighted')
                         for x in evaluation_results_df['results_part_2']
                         if None not in x['relevance']
                         ]]

    # NDCG:
    list_of_ndcgs = [[ndcg_score([x['relevance']], [np.full((1, len(x['relevance'])), 1)[0]])
                      for x in evaluation_results_df['results_part_2']
                      if None not in x['relevance']
                      ]]

    # NDCG AT 5:
    list_of_ndcgs_at_k = [[ndcg_score([x['relevance']], [np.full((1, len(x['relevance'])), 1)[0]], k=k)
                           for x in evaluation_results_df['results_part_2']
                           if None not in x['relevance']
                           ]]

    # PRECISION SCORE:
    # completion of results
    # TODO: Add other columns
    if save_results_for_every_item is True:
        dict_of_model_stats = {'AP': list_of_aps, 'precision_score': list_of_ps[0],
                               'balanced_accuracies': list_of_balanced_accuracies[0],
                               'DCG': list_of_dcgs[0], 'DCG_AT_' + str(k): list_of_dcg_at_k[0],
                               'F1-SCORE': list_of_f1_score[0], 'NDCG': list_of_ndcgs[0],
                               'NDCG_AT_' + str(k): list_of_ndcgs_at_k[0],
                               investigate_by: list_of_models[0], 'slug': list_of_slugs[0],
                               'created_at': list_of_created_at[0]}
    else:
        dict_of_model_stats = {'AP': list_of_aps, 'precision_score': list_of_ps[0],
                               'balanced_accuracies': list_of_balanced_accuracies[0],
                               'DCG': list_of_dcgs[0], 'DCG_AT_' + str(k): list_of_dcg_at_k[0],
                               'F1-SCORE': list_of_f1_score[0], 'NDCG': list_of_ndcgs[0],
                               'NDCG_AT_' + str(k): list_of_ndcgs_at_k[0],
                               investigate_by: list_of_models[0]}

    model_results = pd.DataFrame.from_dict(dict_of_model_stats, orient='index').transpose()

    model_results = model_results.fillna(0)

    # joining with evaluated posts slugs
    if save_results_for_every_item is True:
        # Saving also results for every item.
        recommender_methods = RecommenderMethods()
        posts_categories__ratings_df = recommender_methods.get_posts_categories_dataframe()
        categories_df = posts_categories__ratings_df[['category_title', 'post_slug']]
        model_results = pd.merge(model_results, categories_df, left_on='slug', right_on='post_slug', how='left')
        grouped_results = model_results.groupby(['AP', 'model_variant', 'slug', 'created_at',
                                                 'category_title']).mean().reset_index()
        # full_results = pd.merge(grouped_results, categories_df, left_on='slug', right_on='post_slug', how='left')

        # full_results.columns
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

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.tight_layout()

    # Display the visualization of the Confusion Matrix.
    plt.show()


def show_confusion_matrix():
    # Please be aware that confusion matrix is only
    evaluation_results_df = get_admin_evaluation_results_dataframe()
    list_of_models = [[x for x in evaluation_results_df['model_variant']]]

    y_pred = np.full((1, 20), 1)[0]

    list_of_models.append([x for x in evaluation_results_df['model_name']])

    # AVERAGE PRECISION:
    list_of_confusion_matrices = [[confusion_matrix(x['relevance'], y_pred)
                                   if len(x['relevance']) == len(x['coefficient'])
                                   else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                                   for x in evaluation_results_df['results_part_2']]]

    # CONFUSION MATRIX:

    # for cm in list_of_confusion_matrices:
    np.set_printoptions(precision=2)
    # Confusion matrix, without normalization
    list_of_confusion_matrices_selected = []
    for row in list_of_confusion_matrices:
        for item in row:
            # item
            if item.shape == (2, 2):
                item = np.asmatrix(item)
                # item after conversion to matrix
                list_of_confusion_matrices_selected.append(item)

    # list_of_confusion_matrices_selected:
    cm_mean = np.mean(list_of_confusion_matrices_selected, axis=0)
    plt.figure()
    plot_confusion_matrix(cm_mean, "Confusion matrix")


def print_model_variant_relevances():
    stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=False)
    # Means of model's metrics:
    print(stats)


def save_model_variant_relevances(crop_by_date=False, last_n_by_date=None):
    stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=False,
                                       crop_by_date=crop_by_date, last_n_by_date=last_n_by_date)
    # Means of model's metrics.
    # Saving CSV with evalutation tuning results...
    stats = stats.round(2)
    _hash = random.getrandbits(128)
    path_for_saving = "stats/word2vec/tuning/word2vec_tuning_relevance_results" + str(_hash) + ".csv"
    stats.to_csv(path_for_saving)
    # Results saved.


def print_model_variant_relevances_for_each_article(save_to_csv=False, crop_by_date=False):
    stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=True,
                                       crop_by_date=crop_by_date)
    # Means of model's metrics:
    if save_to_csv is True:
        stats = stats.round(2)
        path_for_saving = "stats/word2vec/tuning/cswiki/word2vec_tuning_relevance_results_by_each_article.csv"
        stats.to_csv(path_for_saving)
        # Results saved.


def print_overall_model_relevances():
    stats = models_complete_statistics(investigate_by='model_name', save_results_for_every_item=True)
    # Means of model's metrics:
    print(stats.to_string())


def print_confusion_matrix():
    print(show_confusion_matrix())
