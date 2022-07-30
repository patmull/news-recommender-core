# load the dataset and split it into training and testing sets
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

# TODO: Not finished yet. Not clear how can be useful!
class RandomForestRegression():

    def run(self):

        dataset = pd.read_csv('word2vec/evaluation/idnes/word2vec_modely_srovnani_filtered.csv', sep=";")
        print(dataset.head(10).to_string())
        dataset_x = dataset[['Negative', 'Vector_size', 'Window', 'Min_count', 'Epochs', 'Sample', 'Softmax']]
        dataset_y = dataset[['Analogies_test']]
        # converting float to int so it can be labeled as ordinal variable
        dataset_x = dataset_x[dataset_x['Sample'].notnull()].copy()
        dataset_x['Sample'] = dataset_x['Sample'].astype(int).astype(str)
        X = dataset_x
        Y = dataset_y
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.30, random_state=101)
        # train the model on train set without using GridSearchCV
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # print prediction results
        dataset_x_unique = dataset_x.rename(columns={'Negative': 'model__Negative', 'Vector_size': 'model__Vector_size',
                                                            'Window': 'model__Window', 'Min_count': 'model__Min_count',
                                                            'Epochs': 'model__Epochs', 'Sample': 'model__Sample', 'Softmax':
                                                            'model__Softmax'})
        dataset_x_unique = dataset_x_unique.apply(lambda x: x.unique())
        print("dataset_x_unique:")
        print(dataset_x_unique.head(10).to_string())

        print("Available params:")
        print(model.get_params().keys())


        dataset_x_unique_dict = dataset_x_unique.to_dict()

        # SETTING PARAMS

        # Number of trees in random forest
        n_estimators = [200, 500, 1000, 1500, 2000]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [10, 50, 110]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        CV_rfc = GridSearchCV(estimator=model, param_grid=random_grid, refit=True, verbose=3, n_jobs=-1)
        CV_rfc.fit(X_train, y_train)


        print("Best parameters according for Random Forrest Classifier:")
        print(CV_rfc.best_params_)


class Anova():

    def __init__(self, dataset):
        if dataset == "brute-force":
            self.filename = 'word2vec/evaluation/idnes/word2vec_modely_srovnani_filtered.csv'
            self.semicolon = True
        elif dataset == "random-search":
            self.filename = 'word2vec/evaluation/idnes/word2vec_tuning_results_random_search.csv'
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
        X = dataset[['Negative', 'Vector_size', 'Window', 'Epochs', 'Sample', 'Softmax']]
        y = dataset[[y_feature]]
        return X, y

    # feature selection
    def select_features(self, X_train, y_train, X_test):
        # configure to select all features
        fs = SelectKBest(score_func=f_classif, k='all')
        # learn relationship from training data
        fs.fit(X_train, y_train)
        # transform train input data
        X_train_fs = fs.transform(X_train)
        # transform test input data
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    def run(self, run_on, col):
        # load the dataset
        X, y = self.load_dataset(self.filename, run_on, self.semicolon)
        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        # feature selection
        X_train_fs, X_test_fs, fs = self.select_features(X_train, y_train, X_test)
        # what are scores for the features
        X_column_names = X.columns.values
        for i in range(len(fs.scores_)):
            print('Feature %s: %f' % (X_column_names[i], fs.scores_[i]))
        # plot the scores
        col.set_title(run_on)
        col.bar(X_column_names, fs.scores_)
        col.tick_params(labelrotation=45)
"""
random_fores_regressor = RandomForestRegression()
random_fores_regressor.run()
"""

list_of_y_features = ['Analogies_test', 'Word_pairs_test_Out-of-vocab_ratio',
                      'Word_pairs_test_Spearman_coeff', 'Word_pairs_test_Pearson_coeff']

fig, ax = pyplot.subplots(nrows=2, ncols=2)
anova = Anova(dataset="random-search")
i = 0
for row in ax:
    for col in row:
        anova.run(list_of_y_features[i], col=col)
        i = i + 1

pyplot.tight_layout()
pyplot.show()

"""
anova.run('Word_pairs_test_Spearman_coeff')
anova.run('Word_pairs_test_Pearson_coeff')
"""

"""
grid_searcher = GridSearcher()
grid_searcher.run()
"""