# load the dataset and split it into training and testing sets
import pandas as pd
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from src.methods.content_based.word2vec.tuning.hyperparameter_tuning import Anova


# noinspection PyPep8Naming
def run():
    dataset = pd.read_csv(
        '../../../../../stats/evaluations/word2vec/tuning/idnes/word2vec_modely_srovnani_filtered.csv', sep=";")
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
    # train_enabled the model on train_enabled set without using GridSearchCV
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

    print("Available stats:")
    print(model.get_params().keys())

    # SETTING PARAMS
    # Number of trees in random_order forest
    n_estimators = [200, 500, 1000, 1500, 2000]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [10, 50, 110, None]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random_order grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    CV_rfc = GridSearchCV(estimator=model, param_grid=random_grid, refit=True, verbose=3, n_jobs=-1)
    CV_rfc.fit(X_train, y_train)

    print("Best hyperparameters according for Random Forrest Classifier:")
    print(CV_rfc.best_params_)


def plot_anova():
    list_of_y_features = ['Analogies_test', 'Word_pairs_test_Out-of-vocab_ratio',
                          'Word_pairs_test_Spearman_coeff', 'Word_pairs_test_Pearson_coeff']

    fig, ax = pyplot.subplots(nrows=2, ncols=2)
    anova = Anova(dataset="random_order-search-doc2vec")
    i = 0
    for row in ax:
        for col in row:
            anova.run(list_of_y_features[i], _col=col)
            i = i + 1

    pyplot.tight_layout()
    pyplot.show()


if __name__ == '__main__':
    plot_anova()
