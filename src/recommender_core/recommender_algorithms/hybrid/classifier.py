import os.path
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy_sentence_bert
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.recommender_core.data_handling.data_queries import RecommenderMethods


class SVM:
    """
    Global models = models for all users

    """

    def __init__(self):
        self.path_to_models_folder = "full_models/hybrid/classifiers/global_models"
        self.model_save_location = Path()

    def train_classifiers(self, df, columns_to_combine, target_variable_name):
        # https://metatext.io/models/distilbert-base-multilingual-cased
        bert_model = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
        df_predicted = pd.DataFrame()
        df_predicted[target_variable_name] = df[target_variable_name]

        # TODO: Somehow combine multiple columns_to_combine
        df = df.fillna('')

        df['combined'] = df[columns_to_combine].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        print("df['combined']")
        print(df['combined'].iloc[0])

        # leaving out 20% for validation set
        print("Splitting dataset to train / validation...")
        X_train, X_validation, y_train, y_validation = train_test_split(df['combined'].tolist(),
                                                                        df_predicted[target_variable_name]
                                                                        .tolist(), test_size=0.2)
        # TODO: Preprocessing
        print("Converting text to vectors...")
        df['vector'] = df['combined'].apply(lambda x: bert_model(x).vector)
        print("Splitting dataset to train / test...")
        X_train, X_test, y_train, y_test = train_test_split(df['vector'].tolist(),
                                                            df_predicted[target_variable_name]
                                                            .tolist(), test_size=0.2)

        print("X_train:")
        print(X_train[:5])

        print("y_train")
        print(y_train[:5])

        print("Training using SVC method...")
        clf_svc = SVC(gamma='auto')
        clf_svc.fit(X_train, y_train)
        y_pred = clf_svc.predict(X_test)
        print("SVC results accuracy score:")
        print(accuracy_score(y_test, y_pred))

        print("Saving the SVC model...")
        print("Folder: " + self.path_to_models_folder)
        Path(self.path_to_models_folder).mkdir(parents=True, exist_ok=True)
        model_file_name = 'svc_classifier_' + target_variable_name + '.pkl'
        print(self.path_to_models_folder)
        print(model_file_name)
        path_to_models_pathlib = Path(self.path_to_models_folder)
        path_to_save = Path.joinpath(path_to_models_pathlib, model_file_name)
        joblib.dump(clf_svc, path_to_save)

        print("Training using RandomForest method...")
        clf_random_forest = RandomForestClassifier(max_depth=9, random_state=0)
        clf_random_forest.fit(X_train, y_train)
        y_pred = clf_random_forest.predict(X_test)
        print("Random Forest Classifier accuracy score:")
        print(accuracy_score(y_test, y_pred))

        print("Saving the random forest model...")
        print("Folder: " + self.path_to_models_folder)
        Path(self.path_to_models_folder).mkdir(parents=True, exist_ok=True)
        model_file_name = 'ranom_forest_classifier_' + target_variable_name + '.pkl'
        print(self.path_to_models_folder)
        print(model_file_name)
        path_to_models_pathlib = Path(self.path_to_models_folder)
        path_to_save = Path.joinpath(path_to_models_pathlib, model_file_name)
        joblib.dump(clf_random_forest, path_to_save)

        return clf_svc, clf_random_forest, X_validation, y_validation, bert_model

    def predict_thumbs(self):
        recommender_methods = RecommenderMethods()
        df = recommender_methods.get_posts_users_categories_thumbs_df()
        columns_to_combine = ['category_title', 'all_features_preprocessed']
        clf_svc, clf_random_forest, X_validation, y_validation, bert_model = self.train_classifiers(df=df,
                                                                                                    columns_to_combine=columns_to_combine,
                                                                                                    target_variable_name='thumbs_value')

        features_list = X_validation
        contexts_list = y_validation

        print("features_list")
        print(features_list)

        print("contexts_list")
        print(contexts_list)
        print("=========================")
        print("Results of SVC:")
        print("=========================")
        self.show_true_predicted(features_list, contexts_list, clf_svc, bert_model)
        print("=========================")
        print("Results of Random Forest Classifier:")
        print("=========================")
        self.show_true_predicted(features_list, contexts_list, clf_random_forest, bert_model)

    def predict_ratings(self):
        recommender_methods = RecommenderMethods()
        df = recommender_methods.get_posts_users_categories_ratings_df()

        columns_to_combine = ['category_title', 'all_features_preprocessed']
        clf_svc, clf_random_forest, X_validation, y_validation, bert_model \
            = self.train_classifiers(df=df, columns_to_combine=columns_to_combine, target_variable_name='rating_value')

        features_list = X_validation
        contexts_list = y_validation

        print("features_list")
        print(features_list)

        print("contexts_list")
        print(contexts_list)

        print("=========================")
        print("Results of SVC:")
        print("=========================")
        self.show_true_predicted(features_list, contexts_list, clf_svc, bert_model)
        print("=========================")
        print("Results of Random Forest Classifier:")
        print("=========================")
        self.show_true_predicted(features_list, contexts_list, clf_random_forest, bert_model)

    def show_true_predicted(self, features_list, contexts_list, clf, bert_model):
        for features_combined, context in zip(features_list, contexts_list):
            print(
                f"True Label: {context}, Predicted Label: {clf.predict(bert_model(features_combined).vector.reshape(1, -1))[0]} \n")
            print("CONTENT:")
            print(features_combined)
