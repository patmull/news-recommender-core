# load the dataset and split it into training and testing sets
import pandas as pd
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


# ***HERE was a RandomForrestRegression. ABANDONED DUE TO: no longer being used
class Anova:

    def __init__(self, dataset):
        if dataset == "brute-force-word2vec":
            self.filename = '../../../../../stats/evaluations/word2vec/tuning/idnes/word2vec_modely_srovnani_filtered.csv'
            self.semicolon = True
        elif dataset == "random_order-search-word2vec":
            self.filename = '../../../../../stats/evaluations/word2vec/tuning/idnes/word2vec_tuning_results_random_search.csv'
            self.semicolon = False
        elif dataset == "random_order-search-doc2vec":
            self.filename = '../../../../../stats/evaluations/doc2vec/doc2vec_tuning_results_random_search.csv'
            self.semicolon = False
        else:
            raise ValueError("Selected dataset does not match with any option.")

    # load the dataset
    def load_dataset(self, filename, y_feature, semicolon=False):
        # load the dataset as a pandas DataFrame
        if semicolon is True:
            sep = ";"
        else:
            sep = ","
        dataset = pd.read_csv(filename, sep=sep)
        # removing rows with zeros
        # retrieve numpy array
        X = dataset[['Negative', 'Vector_size', 'Window', 'Min_count', 'Epochs', 'Sample', 'Softmax']]
        y = dataset[[y_feature]]
        return X, y

    # feature selection
    def select_features(self, X_train, y_train, X_test):
        # configure to select all features
        fs = SelectKBest(score_func=f_classif, k='all')
        # learn relationship from training data
        fs.fit(X_train, y_train)
        # transform train_enabled input data
        X_train_fs = fs.transform(X_train)
        # transform tests input data
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    def run(self, run_on, _col):
        # load the dataset
        X, y = self.load_dataset(self.filename, run_on, self.semicolon)
        # split into train_enabled and tests sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        # feature selection
        X_train_fs, X_test_fs, fs = self.select_features(X_train, y_train, X_test)
        # what are scores for the features
        X_column_names = X.columns.values
        for j in range(len(fs.scores_)):
            print('Feature %s: %f' % (X_column_names[j], fs.scores_[j]))
        # plot the scores
        _col.set_title(run_on)
        _col.bar(X_column_names, fs.scores_)
        _col.tick_params(labelrotation=45)

list_of_y_features = ['Analogies_test',
                      'Word_pairs_test_Out-of-vocab_ratio',
                      'Word_pairs_test_Spearman_coeff',
                      'Word_pairs_test_Pearson_coeff']

fig, ax = pyplot.subplots(nrows=2, ncols=2)
anova = Anova(dataset="random_order-search-doc2vec")
i = 0
for row in ax:
    for col in row:
        anova.run(list_of_y_features[i], col=col)
        i = i + 1

pyplot.tight_layout()
pyplot.show()
