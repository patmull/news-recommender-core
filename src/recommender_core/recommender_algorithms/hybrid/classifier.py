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

    # TODO: Model loading instead of re-training
    # TODO: Splitting model by user, e.g.: user_models/user_XY/random_forest_classifier_rating_value_user_XY.pkl
    # TODO: Hyperparameter tuning
    # TODO: Prepare for integration to API
    # TODO: Finish Python <--> PHP communication

    def __init__(self):
        self.path_to_models_global_folder = "full_models/hybrid/classifiers/global_models"
        self.path_to_models_user_folder = "full_models/hybrid/classifiers/users_models"
        self.model_save_location = Path()

    def train_classifiers(self, df, columns_to_combine, target_variable_name, user_id=None):
        # https://metatext.io/models/distilbert-base-multilingual-cased
        bert_model = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
        df_predicted = pd.DataFrame()
        df_predicted[target_variable_name] = df[target_variable_name]

        df = df.fillna('')

        df['combined'] = df[columns_to_combine].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        print("df['combined']")
        print(df['combined'].iloc[0])

        # leaving out 20% for validation set
        print("Splitting dataset to train / validation...")
        X_train, X_validation, y_train, y_validation = train_test_split(df['combined'].tolist(),
                                                                        df_predicted[target_variable_name]
                                                                        .tolist(), test_size=0.2)
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

        print("Training using RandomForest method...")
        clf_random_forest = RandomForestClassifier(max_depth=9, random_state=0)
        clf_random_forest.fit(X_train, y_train)
        y_pred = clf_random_forest.predict(X_test)
        print("Random Forest Classifier accuracy score:")
        print(accuracy_score(y_test, y_pred))

        print("Saving the SVC model...")
        if user_id is not None:
            print("Folder: " + self.path_to_models_user_folder)
            Path(self.path_to_models_user_folder).mkdir(parents=True, exist_ok=True)

            model_file_name_svc = 'svc_classifier_' + target_variable_name + '_user_' + str(user_id) + '.pkl'
            model_file_name_random_forest = 'random_forest_classifier_' + target_variable_name + '_user_' \
                                            + str(user_id) + '.pkl'
            print(self.path_to_models_global_folder)
            print(model_file_name_svc)
            print(model_file_name_random_forest)
            path_to_models_pathlib = Path(self.path_to_models_user_folder)
            path_to_save_svc = Path.joinpath(path_to_models_pathlib, model_file_name_svc)
            joblib.dump(clf_random_forest, path_to_save_svc)
            path_to_save_forest = Path.joinpath(path_to_models_pathlib, model_file_name_random_forest)
            joblib.dump(clf_random_forest, path_to_save_forest)

        else:
            print("Folder: " + self.path_to_models_global_folder)
            Path(self.path_to_models_global_folder).mkdir(parents=True, exist_ok=True)
            model_file_name = 'svc_classifier_' + target_variable_name + '.pkl'
            print(self.path_to_models_global_folder)
            print(model_file_name)
            path_to_models_pathlib = Path(self.path_to_models_global_folder)
            path_to_save_svc = Path.joinpath(path_to_models_pathlib, model_file_name)
            joblib.dump(clf_svc, path_to_save_svc)
            print("Saving the random forest model...")
            print("Folder: " + self.path_to_models_global_folder)
            Path(self.path_to_models_global_folder).mkdir(parents=True, exist_ok=True)
            model_file_name = 'random_forest_classifier_' + target_variable_name + '.pkl'
            print(self.path_to_models_global_folder)
            print(model_file_name)
            path_to_models_pathlib = Path(self.path_to_models_global_folder)
            path_to_save_forest = Path.joinpath(path_to_models_pathlib, model_file_name)
            joblib.dump(clf_random_forest, path_to_save_forest)

        return clf_svc, clf_random_forest, X_validation, y_validation, bert_model

    def load_classifiers(self, df, input_variables, predicted_variable, user_id=None):
        # https://metatext.io/models/distilbert-base-multilingual-cased
        print("Loading sentence bert multilingual model...")
        bert_model = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
        if predicted_variable == 'thumbs_values' or predicted_variable == 'ratings_values':
            if user_id is None:
                model_file_name_svc = 'svc_classifier_' + predicted_variable + '.pkl'
                model_file_name_random_forest = 'random_forest_classifier_' + predicted_variable + '.pkl'
                path_to_models_pathlib = Path(self.path_to_models_global_folder)
            else:
                print("Loading user's personalized classifier model for user " + str(user_id))
                model_file_name_svc = 'svc_classifier_' + predicted_variable + '_user_' + str(user_id) + '.pkl'
                model_file_name_random_forest = 'random_forest_classifier_' + predicted_variable + '_user_' \
                                                + str(user_id) + '.pkl'
                path_to_models_pathlib = Path(self.path_to_models_user_folder)
            path_to_load_svc = Path.joinpath(path_to_models_pathlib, model_file_name_svc)
            path_to_load_random_forest = Path.joinpath(path_to_models_pathlib, model_file_name_random_forest)
        else:
            raise ValueError("Loading of model with inserted name of predicted variable is not supported. Are you sure"
                             "about the value of the 'predicted_variable'?")

        try:
            clf_svc = joblib.load(path_to_load_svc)
        except FileNotFoundError as file_not_found_error:
            print(file_not_found_error)
            print("Model file was not found in the location, training from the start...")
            self.train_classifiers(df=df, columns_to_combine=input_variables,
                                   target_variable_name=predicted_variable, user_id=user_id)
            clf_svc = joblib.load(path_to_load_svc)

        try:
            clf_random_forest = joblib.load(path_to_load_random_forest)
        except FileNotFoundError as file_not_found_error:
            print(file_not_found_error)
            print("Model file was not found in the location, training from the start...")
            self.train_classifiers(df=df, columns_to_combine=input_variables,
                                   target_variable_name=predicted_variable, user_id=user_id)
            clf_random_forest = joblib.load(path_to_load_random_forest)

        X_validation = df[input_variables]
        # y_validation = df[predicted_variable]

        return clf_svc, clf_random_forest, X_validation, bert_model

    def predict_thumbs(self, force_retraining=False):
        recommender_methods = RecommenderMethods()
        df = recommender_methods.get_posts_users_categories_thumbs_df()
        columns_to_use = ['category_title', 'all_features_preprocessed']

        if force_retraining is True:
            clf_svc, clf_random_forest, X_validation, y_validation, bert_model \
                = self.train_classifiers(df=df, columns_to_combine=columns_to_use,
                                         target_variable_name='thumbs_value')
        else:
            clf_svc, clf_random_forest, X_validation, bert_model \
                = self.load_classifiers(df=df, input_variables=columns_to_use, predicted_variable='thumbs_value')

        print("=========================")
        print("Results of SVC:")
        print("=========================")
        self.show_predicted(X_validation, columns_to_use, clf_svc, bert_model)
        print("=========================")
        print("Results of Random Forest Classifier:")
        print("=========================")
        self.show_predicted(X_validation, columns_to_use, clf_random_forest, bert_model)

    def predict_ratings(self, force_retraining=False, show_only_sample_of=None, user_id=None):
        recommender_methods = RecommenderMethods()
        if user_id is None:
            df = recommender_methods.get_posts_categories_dataframe()
        else:
            df = recommender_methods.get_posts_users_categories_ratings_df()
        df = df.rename(columns={'title': 'category_title'})
        columns_to_use = ['category_title', 'all_features_preprocessed']

        if force_retraining is True:
            clf_svc, clf_random_forest, X_validation, y_validation, bert_model \
                = self.train_classifiers(df=df, columns_to_combine=columns_to_use,
                                         target_variable_name='ratings_values',  user_id=user_id)
        else:
            clf_svc, clf_random_forest, X_validation, bert_model \
                = self.load_classifiers(df=df, input_variables=columns_to_use,
                                        predicted_variable='ratings_values', user_id=user_id)
        if not type(show_only_sample_of) is None:
            if type(show_only_sample_of) is int:
                X_validation = X_validation.sample(show_only_sample_of)

        print("=========================")
        print("Results of SVC:")
        print("=========================")
        self.show_predicted(X_unseen_df=X_validation, input_variables=columns_to_use, clf=clf_svc,
                            bert_model=bert_model)
        print("=========================")
        print("Results of Random Forest Classifier:")
        print("=========================")
        self.show_predicted(X_unseen_df=X_validation, input_variables=columns_to_use, clf=clf_random_forest,
                            bert_model=bert_model)

    def show_true_vs_predicted(self, features_list, contexts_list, clf, bert_model):
        """
        Method for evaluation on validation dataset, not actual unseen dataset.
        """
        for features_combined, context in zip(features_list, contexts_list):
            print(
                f"True Label: {context}, "
                f"Predicted Label: {clf.predict(bert_model(features_combined).vector.reshape(1, -1))[0]} \n")
            print("CONTENT:")
            print(features_combined)

    def show_predicted(self, X_unseen_df, input_variables, clf, bert_model):
        """
        Method for evaluation on validation dataset, not actual unseen dataset.
        """
        print("Combining the selected columns")
        X_unseen_df['combined'] = X_unseen_df[input_variables].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                     axis=1)
        print("Vectorizing the selected columns...")
        # TODO: Takes a lot of time... Probably pre-calculate.
        y_pred_unseen = X_unseen_df['combined'].apply(lambda x: clf.predict(bert_model(x).vector.reshape(1, -1))[0])
        y_pred_unseen = y_pred_unseen.rename('prediction')
        # y_pred_df = pd.DataFrame({'article_index': y_pred_unseen.index, 'predictions': y_pred_unseen.values})
        print(y_pred_unseen.head(20))
        df_results = pd.merge(X_unseen_df,  pd.DataFrame(y_pred_unseen), how='left', left_index=True, right_index=True)
        print(df_results.head(20))
        df_results.head(20).to_csv('research/hybrid/testing_hybrid_classifier_df_results.csv')

        """
        for features_combined in zip(features_list):
            print(
                f"Predicted Label: {clf.predict(bert_model(features_combined).vector.reshape(1, -1))[0]} \n")
            print("CONTENT:")
            print(features_combined)
        """