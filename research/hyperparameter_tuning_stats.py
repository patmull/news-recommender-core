import matplotlib
import pandas as pd
import seaborn
from scipy.stats import kendalltau, pearsonr, spearmanr

tuning_results_df = pd.read_csv('word2vec/evaluation/idnes/word2vec_tuning_results_random_search.csv', sep=",")


def kendall_pval(x, y):
    return kendalltau(x, y)[1]


def pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


def spearmanr_pval(x, y):
    return spearmanr(x, y)[1]


# print(tuning_results_df.to_string())

tuning_results_df_filtered = tuning_results_df.drop(columns=['Model_Variant'])

print("Kendall corr:")
kendall_corr = tuning_results_df_filtered.corr(method=kendall_pval)
print(kendall_corr.to_string())
print("Spearman corr:")
spearman_corr = tuning_results_df_filtered.corr(method=spearmanr_pval)
print(spearman_corr.to_string())
spearman_corr = tuning_results_df_filtered.corr(method='spearman')

spearman_corr = spearman_corr.rename(columns = {'Word_pairs_test_Pearson_coeff': 'Pair_t_Pearson', 'Word_pairs_test_Pearson_p-val': 'Pair_p', 'Word_pairs_test_Spearman_coeff': 'Pair_Spearman', 'Word_pairs_test_Spearman_p-val': 'Pair_t_Spearman', 'Word_pairs_test_Out-of-vocab_ratio': 'Out_of_vocab'})

# Correlated variables:
# Min_count: Word_pairs_test_Out-of-vocab_ratio (this makes sense since it controls the dropout of words)
# Epochs:  correlation with word_pairs test (with Spearman stats)
# Sample:  correlation with word_pairs test (with Spearman stats)
# Softmax: correlation with analogies test and word_pairs test (with Spearman stats)
# Correlation of analogies test and word pair test as expected

matplotlib.pyplot.figure(figsize=(6, 6), dpi=1200)
seaborn.set(font_scale=0.7)
heatmap = seaborn.heatmap(spearman_corr, annot=True, annot_kws={"size": 6})
heatmap.figure.tight_layout()
matplotlib.pyplot.show()