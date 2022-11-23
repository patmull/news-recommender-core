import os
import pickle
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import spacy_sentence_bert
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score, confusion_matrix

from src.constants.naming import Naming
from src.presentation.data_logging import log_dataframe_info
from src.recommender_core.data_handling.data_manipulation import get_redis_connection
from src.recommender_core.data_handling.data_queries import RecommenderMethods

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from classifier.")


def load_bert_model():
    bert_model = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    return bert_model


def get_df_predicted(df, target_variable_name):
    df_predicted = pd.DataFrame()
    df_predicted[target_variable_name] = df[target_variable_name]

    # leaving out 20% for validation set
    print("Splitting dataset to train_enabled / validation...")
    return df_predicted


def show_true_vs_predicted(features_list, contexts_list, clf, bert_model):
    """
    Method for evaluation on validation dataset, not actual unseen dataset.
    """
    for features_combined, context in zip(features_list, contexts_list):
        print(
            f"True Label: {context}, "
            f"Predicted Label: {clf.predict(bert_model(features_combined).vector.reshape(1, -1))[0]} \n")
        print("CONTENT:")


def predict_from_vectors(X_unseen_df, clf, predicted_var_for_redis_key_name, user_id,
                         save_testing_csv=False, bert_model=None, col_to_combine=None, testing_mode=False,
                         store_to_redis=True):
    """

    @param store_to_redis:
    @param X_unseen_df:
    @param clf:
    @param predicted_var_for_redis_key_name:
    @param user_id:
    @param save_testing_csv:
    @param bert_model:
    @param col_to_combine:
    @param testing_mode: Allows threshold = 0 to make sure some value is added.
    @return:

    Method for actual live, deployed use. This uses the already filled vectors from PostgreSQL but if doesn't
    exists, calculate new ones from passed BERT model.

    If this method takes a lot of time, prefill BERT vectors with prefilling function fill_bert_vector_representation().
    """
    if predicted_var_for_redis_key_name == Naming.PREDICTED_BY_THUMBS_REDIS_KEY_NAME:
        if testing_mode is False:
            threshold = 1  # binary relevance rating
        else:
            threshold = 0
    elif predicted_var_for_redis_key_name == Naming.PREDICTED_BY_STARS_REDIS_KEY_NAME:
        if testing_mode is False:
            threshold = 3  # the Likert scale
        else:
            threshold = 0
    else:
        raise ValueError("No from passed predicted rating key names matches the available options!")

    if bert_model is not None:
        if col_to_combine is None:
            raise ValueError("If BERT model is supplied, then column list needs "
                             "to be supplied to col_to_combine_parameter!")

    print("X_unseen_df size:")
    print(X_unseen_df)
    print(len(X_unseen_df.index))

    print("Vectoring the selected columns...")
    # TODO: Takes a lot of time... Probably pre-calculate.
    print("X_unseen_df:")
    print(X_unseen_df)

    print("Loading vectors or creating new if does not exists...")
    # noinspection  PyPep8
    y_pred_unseen = X_unseen_df \
        .apply(lambda x: clf
               .predict(pickle
                        .loads(x['bert_vector_representation']))[0] if pd.notnull(x['bert_vector_representation']) else
    clf.predict(bert_model(' '.join(str(x[col_to_combine]))).vector.reshape(1, -1))[0], axis=1)

    y_pred_unseen = y_pred_unseen.rename('prediction')

    logging.debug("X_unseen_df:")
    logging.debug(X_unseen_df.columns)
    logging.debug("y_pred_unseen:")
    logging.debug(pd.DataFrame(y_pred_unseen).columns)

    df_results = pd.merge(X_unseen_df, pd.DataFrame(y_pred_unseen), how='left', left_index=True, right_index=True)

    logging.debug("df_results:")
    logging.debug(df_results.columns)

    # NOTICE: Freshness of articles is already handled in predict_relevance_for_user() method

    if save_testing_csv is True:
        # noinspection PyTypeChecker
        df_results.head(20).to_csv('research/user_based/testing_hybrid_classifier_df_results.csv')

    if store_to_redis:

        if user_id is not None:
            r = get_redis_connection()
            user_redis_key = 'user' + Naming.REDIS_DELIMITER + str(user_id) + Naming.REDIS_DELIMITER \
                             + 'post-classifier-by-' + predicted_var_for_redis_key_name
            # remove old records
            r.delete(user_redis_key)
            logging.debug("iteration through records:")
            i = 0
            # fetch Redis set with a new set of recommended posts
            for row in zip(*df_results.to_dict("list").values()):
                slug = "" + row[3] + ""
                logging.info("-------------------")
                logging.info("Predicted rating for slug | " + slug + ":")

                logging.debug("row[5]:")
                logging.info(row[5])
                if row[5] is not None:
                    # If predicted rating is == 1 (= relevant)
                    if int(row[5]) >= threshold:
                        # Saving individually to set
                        logging.info("Adding REDIS KEY")
                        r.sadd(user_redis_key, slug)
                        logging.info("Inserted record num. " + str(i))
                        i = i + 1
                else:
                    logging.warning("No predicted values found. Skipping this record.")
                    pass


# noinspection PyPep8Naming
def show_predicted(X_unseen_df, input_variables, clf, bert_model, save_testing_csv=False):
    """
    Method for evaluation on validation dataset, not actual unseen dataset.
    Use for experimentation with features.
    """
    print("Combining the selected columns")
    X_unseen_df['combined'] = X_unseen_df[input_variables].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                 axis=1)
    print("Vectorizing the selected columns...")
    y_pred_unseen = X_unseen_df['combined'].apply(lambda x: clf.predict(bert_model(x).vector.reshape(1, -1))[0])
    y_pred_unseen = y_pred_unseen.rename('prediction')
    df_results = pd.merge(X_unseen_df, pd.DataFrame(y_pred_unseen), how='left', left_index=True, right_index=True)
    if save_testing_csv is True:
        # noinspection PyTypeChecker
        df_results.head(20).to_csv('research/hybrid/testing_hybrid_classifier_df_results.csv')


class Classifier:
    """
    Global models = models for all users
    """

    # TODO: Prepare for test_integration to API
    # TODO: Finish Python <--> PHP communication
    # TODO: Hyper parameter tuning

    def __init__(self):
        self.path_to_models_global_folder = "full_models/hybrid/classifiers/global_models"
        self.path_to_models_user_folder = "full_models/hybrid/classifiers/users_models"
        self.model_save_location = Path()
        self.bert_model = None

    def train_classifiers(self, df, columns_to_combine, target_variable_name, user_id=None, test_run=None):
        if test_run is None:
            test_size = 0.2
        else:
            test_size = 0.5

        logging.debug("Loading Bert model...")
        self.bert_model = load_bert_model()
        # https://metatext.io/models/distilbert-base-multilingual-cased
        df_predicted = get_df_predicted(df, target_variable_name)

        # TODO: Replace NaN columns with title, then replace with emppty string as a last resort. PRIORITY: MEDIUM-LOW
        df = df.fillna('')

        try:
            df['combined'] = df[columns_to_combine].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        except IndexError as ie:
            logging.debug(df.columns)
            logging.warning("Index error had occurred.")
            logging.warning("In this stage of project deployment, exception will be raised. "
                            "Please try to fix this issue.")
            log_dataframe_info(df)

            raise ie
        # Preventing IndexError error
        logging.debug('df.columns:')
        logging.debug(df.columns)

        try:
            logging.debug("df['combined']")
            logging.debug(df['combined'].iloc[0])
        except IndexError as ie:
            logging.warning("Index error had occurred (even after replacing empty columns with post title).")
            logging.warning("This is probably caused by empty dataframe.")
            logging.warning("In this stage of project deployment, exception will be raised. "
                            "Please try to fix this issue.")
            log_dataframe_info(df)

            raise ie
        # noinspection PyPep8Naming
        X_train, X_validation, y_train, y_validation = train_test_split(df['combined'].tolist(),
                                                                        df_predicted[target_variable_name]
                                                                        .tolist(), test_size=test_size)
        logging.debug("Converting text to vectors...")
        df['vector'] = df['combined'].apply(lambda x: self.bert_model(x).vector)
        logging.debug("Splitting dataset to train_enabled / test...")
        # noinspection PyPep8Naming
        X_train, X_test, y_train, y_test = train_test_split(df['vector'].tolist(),
                                                            df_predicted[target_variable_name]
                                                            .tolist(), test_size=test_size)

        logging.info("Training using SVC method...")

        # defining parameter range
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }

        grid = GridSearchCV(SVC(gamma='auto'), param_grid, refit=True, verbose=3)

        try:
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
        except Exception as e:
            logging.warning(e)
            raise e

        logging.info("================================")
        logging.info("Hyper parameter tuning results:")
        logging.info("================================")
        logging.info("best parameters after tuning:")
        logging.info(grid.best_params_)
        logging.info("model after hyper-parameter tuning:")
        logging.info(grid.best_estimator_)

        clf_svc = grid

        logging.info("SVC results accuracy score:")

        logging.info(accuracy_score(y_test, y_pred))
        logging.info("PRECISION SCORE:")
        logging.info(precision_score(y_test, y_pred, average='weighted'))
        logging.info("BALANCED_ACCURACY:")
        logging.info(balanced_accuracy_score(y_test, y_pred))
        logging.info("CONFUSION MATRIX:")
        logging.info(confusion_matrix(y_test, y_pred))
        logging.info("PRECISION SCORE:")
        logging.info(precision_score(y_test, y_pred, average=None))

        path_to_eval_results_file = Path("research/hybrid/classifier_eval_results.txt")

        self.save_eval_results(path_to_eval_results_file, user_id=user_id, method="SVC", y_test=y_test,
                               y_pred=y_pred)

        param_grid = {
            "max_depth": [3, None],
            "max_features": [1, 3, 10],
            "min_samples_split": [1, 3, 10],
            "min_samples_leaf": [1, 3, 10],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        }

        logging.info("Training using RandomForest method...")
        grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3)

        grid.fit(X_train, y_train)
        logging.info("best parameters after tuning:")
        print(grid.best_params_)
        logging.info("model after hyper-parameter tuning:")
        print(grid.best_estimator_)

        clf_random_forest = grid

        y_pred = clf_random_forest.predict(X_test)

        logging.info("================================")
        logging.info("Hyper parameter tuning results:")
        logging.info("================================")
        logging.info("best parameters after tuning:")
        logging.info(grid.best_params_)
        logging.info("model after hyper-parameter tuning:")
        logging.info(grid.best_estimator_)
        logging.info("Random Forest classifier accuracy score:")
        logging.info(accuracy_score(y_test, y_pred))

        self.save_eval_results(path_to_eval_results_file, user_id=user_id, method="Random Forrest", y_test=y_test,
                               y_pred=y_pred)

        logging.info("Saving the SVC model...")
        if user_id is not None:
            logging.info("Folder: " + self.path_to_models_user_folder)
            Path(self.path_to_models_user_folder).mkdir(parents=True, exist_ok=True)

            model_file_name_svc = 'svc_classifier_' + target_variable_name + '_user_' + str(user_id) + '.pkl'
            model_file_name_random_forest = 'random_forest_classifier_' + target_variable_name + '_user_' \
                                            + str(user_id) + '.pkl'
            path_to_models_pathlib = Path(self.path_to_models_user_folder)
            path_to_save_svc = Path.joinpath(path_to_models_pathlib, model_file_name_svc)
            joblib.dump(clf_random_forest, path_to_save_svc)
            path_to_save_forest = Path.joinpath(path_to_models_pathlib, model_file_name_random_forest)
            joblib.dump(clf_random_forest, path_to_save_forest)

        else:
            logging.info("Folder: " + self.path_to_models_global_folder)
            Path(self.path_to_models_global_folder).mkdir(parents=True, exist_ok=True)
            model_file_name = 'svc_classifier_' + target_variable_name + '.pkl'
            logging.debug(self.path_to_models_global_folder)
            logging.debug(model_file_name)
            path_to_models_pathlib = Path(self.path_to_models_global_folder)
            path_to_save_svc = Path.joinpath(path_to_models_pathlib, model_file_name)
            joblib.dump(clf_svc, path_to_save_svc)
            logging.info("Saving the random forest model...")
            logging.info("Folder: " + self.path_to_models_global_folder)
            Path(self.path_to_models_global_folder).mkdir(parents=True, exist_ok=True)
            model_file_name = 'random_forest_classifier_' + target_variable_name + '.pkl'
            logging.debug(self.path_to_models_global_folder)
            logging.debug(model_file_name)
            path_to_models_pathlib = Path(self.path_to_models_global_folder)
            path_to_save_forest = Path.joinpath(path_to_models_pathlib, model_file_name)
            joblib.dump(clf_random_forest, path_to_save_forest)

        return clf_svc, clf_random_forest, X_validation, y_validation, self.bert_model

    def load_classifiers(self, df, input_variables, predicted_variable, user_id=None):
        # https://metatext.io/models/distilbert-base-multilingual-cased

        if predicted_variable == 'thumbs_values' or predicted_variable == 'ratings_values':
            if user_id is None:
                model_file_name_svc = 'svc_classifier_' + predicted_variable + '.pkl'
                model_file_name_random_forest = 'random_forest_classifier_' + predicted_variable + '.pkl'
                path_to_models_pathlib = Path(self.path_to_models_global_folder)
            else:
                logging.info("Loading user's personalized classifiers models for user " + str(user_id))
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
            logging.debug("Loading SVC...")
            clf_svc = joblib.load(path_to_load_svc)
        except FileNotFoundError as file_not_found_error:
            logging.warning(file_not_found_error)
            logging.warning("Model file was not found in the location, training from the start...")
            try:
                self.train_classifiers(df=df, columns_to_combine=input_variables,
                                       target_variable_name=predicted_variable, user_id=user_id)
            except ValueError as ve:
                logging.warning(ve)
                raise ve
            clf_svc = joblib.load(path_to_load_svc)
        except KeyError as ke:
            logging.warning(ke)
            logging.warning("Model file was not found in the location, training from the start...")
            try:
                self.train_classifiers(df=df, columns_to_combine=input_variables,
                                       target_variable_name=predicted_variable, user_id=user_id)
            except ValueError as ve:
                logging.warning(ve)
                raise ve
            clf_svc = joblib.load(path_to_load_svc)
        except Exception as e:
            raise e

        try:
            logging.warning("Loading Random Forest...")
            clf_random_forest = joblib.load(path_to_load_random_forest)
        except FileNotFoundError as file_not_found_error:
            logging.warning(file_not_found_error)
            logging.warning("Model file was not found in the location, training from the start...")
            self.train_classifiers(df=df, columns_to_combine=input_variables,
                                   target_variable_name=predicted_variable, user_id=user_id)
            clf_random_forest = joblib.load(path_to_load_random_forest)
        except KeyError as ke:
            logging.warning(ke)
            logging.warning("Model file was not found in the location, training from the start...")
            try:
                self.train_classifiers(df=df, columns_to_combine=input_variables,
                                       target_variable_name=predicted_variable, user_id=user_id)
            except ValueError as ve:
                logging.warning(ve)
                raise ve
            clf_svc = joblib.load(path_to_load_svc)
        except Exception as e:
            raise e

        return clf_svc, clf_random_forest

    def predict_relevance_for_user(self, relevance_by, force_retraining=False, use_only_sample_of=None, user_id=None,
                                   experiment_mode=False, only_with_prefilled_bert_vectors=False, bert_model=None,
                                   latest_posts=True, save_df_posts_users_categories_relevance=False,
                                   store_to_redis=True):
        if only_with_prefilled_bert_vectors is False:
            if bert_model is None:
                raise ValueError("Loaded BERT model needs to be supplied if only_with_prefilled_bert_vectors parameter"
                                 "is set to False")

        columns_to_combine = ['category_title', 'all_features_preprocessed', 'full_text']

        recommender_methods = RecommenderMethods()
        all_user_df = recommender_methods.get_all_users()

        logging.debug("all_user_df.columns")
        logging.debug(all_user_df.columns)

        if type(user_id) == int:
            if user_id not in all_user_df["id"].values:
                raise ValueError("User with id %d not found in DB." % (user_id,))
        else:
            raise ValueError("Bad data type for argument user_id")

        if not type(relevance_by) == str:
            raise ValueError("Bad data type for argument relevance_by")

        if use_only_sample_of is not None:
            if not type(use_only_sample_of) == int:
                raise ValueError("Bad data type for argument use_only_sample_of")

        if relevance_by == 'thumbs':
            df_posts_users_categories_relevance = recommender_methods \
                .get_posts_users_categories_thumbs_df(user_id=user_id,
                                                      only_with_bert_vectors=only_with_prefilled_bert_vectors)

            if len(df_posts_users_categories_relevance.index) == 0:
                logging.warning("Length of df_posts_users_categories_relevance is 0.")
                logging.warning("This is probably cause by user not participating in any thumbs rating yet.")
                logging.warning("Skipping this user.")
                raise ValueError("df_posts_users_categories_relevance length is 0. User probbaly has not "
                                 "participated in thumbs rating.")
            elif len(df_posts_users_categories_relevance.index) < 0:
                logging.warning("Non sufficient number of examples for calculating the classifier.")
                logging.warning("Skipping this user.")
                raise ValueError("User needs to provide at least 10 ratings.")

            logging.debug("df_posts_users_categories_relevance:")
            logging.debug(df_posts_users_categories_relevance)

            if save_df_posts_users_categories_relevance:
                df_posts_users_categories_relevance.to_csv(
                    Path('tests/testing_datasets/true_posts_categories_thumbs_data_for_df.csv'))

            target_variable_name = 'thumbs_values'
            predicted_var_for_redis_key_name = Naming.PREDICTED_BY_THUMBS_REDIS_KEY_NAME
        elif relevance_by == 'stars':
            df_posts_users_categories_relevance = recommender_methods \
                .get_posts_users_categories_ratings_df(user_id=user_id,
                                                       only_with_bert_vectors=only_with_prefilled_bert_vectors)

            if len(df_posts_users_categories_relevance.index) == 0:
                logging.warning("Length of df_posts_users_categories_relevance is 0.")
                logging.warning("This is probably cause by user not participating in any stars rating yet.")
                logging.warning("Skipping this user.")
                raise ValueError("df_posts_users_categories_relevance length is 0. User probably has not "
                                 "participated in stars rating.")
            logging.debug("df_posts_users_categories_relevance:")
            logging.debug(df_posts_users_categories_relevance)

            if save_df_posts_users_categories_relevance:
                df_posts_users_categories_relevance.to_csv(
                    Path('tests/testing_datasets/true_posts_categories_stars_data_for_df.csv'))

            target_variable_name = 'ratings_values'
            predicted_var_for_redis_key_name = Naming.PREDICTED_BY_STARS_REDIS_KEY_NAME
        else:
            raise ValueError("No options from allowed relevance options selected.")

        df_posts_categories = recommender_methods \
            .get_posts_categories_dataframe(only_with_bert_vectors=only_with_prefilled_bert_vectors,
                                            from_cache=False)

        df_posts_categories = df_posts_categories.rename(columns={'title': 'category_title'})
        df_posts_categories = df_posts_categories.rename(columns={'created_at_x': 'post_created_at'})

        if latest_posts:
            logging.debug("df_posts_categories")
            logging.debug(df_posts_categories)
            logging.debug(df_posts_categories.columns)

            logging.debug("df_posts_categories, created_at column")
            logging.debug(df_posts_categories['post_created_at'].head(10))
            df_posts_categories["post_created_at"] = pd.to_datetime(df_posts_categories["post_created_at"])

            # Getting 100 latest (newest) posts by created date to filter only new articles for user
            df_posts_categories = df_posts_categories.sort_values(by="post_created_at", ascending=False)
            df_posts_categories = df_posts_categories.head(100)
            logging.debug("df_posts_categories, created_at column")
            logging.debug(df_posts_categories['post_created_at'].head(10))

        if force_retraining is True:
            logging.info("Retraining the classifier")
            # noinspection PyPep8Naming
            clf_svc, clf_random_forest, X_validation, y_validation, bert_model \
                = self.train_classifiers(df=df_posts_users_categories_relevance, columns_to_combine=columns_to_combine,
                                         target_variable_name=target_variable_name, user_id=user_id)
        else:
            clf_svc, clf_random_forest \
                = self.load_classifiers(df=df_posts_users_categories_relevance, input_variables=columns_to_combine,
                                        predicted_variable=target_variable_name, user_id=user_id)

        if experiment_mode is True:
            # noinspection PyPep8Naming
            X_unseen = df_posts_categories[columns_to_combine]
            if not type(use_only_sample_of) is None:
                if type(use_only_sample_of) is int:
                    # noinspection PyPep8Naming
                    X_unseen = X_unseen.sample(use_only_sample_of)
            logging.debug("Loading sentence bert multilingual model...")
            logging.debug("=========================")
            logging.debug("Results of SVC:")
            logging.debug("=========================")
            show_predicted(X_unseen_df=X_unseen, input_variables=columns_to_combine, clf=clf_svc,
                           bert_model=bert_model)
            logging.debug("=========================")
            logging.debug("Results of Random Forest:")
            logging.debug("=========================")
            show_predicted(X_unseen_df=X_unseen, input_variables=columns_to_combine, clf=clf_random_forest,
                           bert_model=bert_model)
        else:
            columns_to_select = columns_to_combine + ['slug', 'bert_vector_representation']
            # noinspection PyPep8Naming
            X_unseen = df_posts_categories[columns_to_select]
            if not type(use_only_sample_of) is None:
                if type(use_only_sample_of) is int:
                    # noinspection PyPep8Naming
                    X_unseen = X_unseen.sample(use_only_sample_of)
            logging.debug("=========================")
            logging.debug("Inserting by SVC:")
            logging.debug("=========================")

            logging.debug("X_unseen:")
            logging.debug(X_unseen)
            logging.debug("clf_svc:")
            logging.debug(clf_svc)

            predict_from_vectors(X_unseen_df=X_unseen, clf=clf_svc, user_id=user_id,
                                 predicted_var_for_redis_key_name=predicted_var_for_redis_key_name,
                                 bert_model=bert_model, col_to_combine=columns_to_combine,
                                 save_testing_csv=True, store_to_redis=store_to_redis)

            logging.debug("=========================")
            logging.debug("Inserting by Random Forest:")
            logging.debug("=========================")
            predict_from_vectors(X_unseen_df=X_unseen, clf=clf_random_forest, user_id=user_id,
                                 predicted_var_for_redis_key_name=predicted_var_for_redis_key_name,
                                 bert_model=bert_model, col_to_combine=columns_to_combine,
                                 save_testing_csv=True, store_to_redis=store_to_redis)


    def save_eval_results(self, file, user_id, method, y_test, y_pred, to_txt=True, to_csv=True):

        if to_txt:
            with open(file, 'a') as f:
                f.write("==========================\n")
                f.write("MODEL %s.\n" % method)
                f.write("===========================\n")
                f.write("===========================\n")
                f.write("==============\n")
                f.write("USER ID:\n")
                f.write(str(user_id))
                f.write("==================\n")
                f.write("ACCURACY SCORE:\n")
                f.write(str(accuracy_score(y_test, y_pred)))
                f.write("PRECISION SCORE:\n")
                f.write(str(precision_score(y_test, y_pred, average='weighted')))
                f.write("BALANCED_ACCURACY:\n")
                f.write(str(balanced_accuracy_score(y_test, y_pred)))
                f.write("CONFUSION MATRIX:\n")
                f.write(str(confusion_matrix(y_test, y_pred)))
                f.write("PRECISION SCORE:\n")
                f.write(str(precision_score(y_test, y_pred, average=None)))

        if to_csv:
            user_id_list = []
            accuracy_score_list = []
            precision_score_weighted = []
            balanced_accuracy_score_list = []
            precision_score_list = []
            method_list = []
            y_test_list = []
            y_pred_list = []

            user_id_list.append(user_id)
            accuracy_score_list.append(accuracy_score(y_test, y_pred))
            precision_score_weighted.append(precision_score(y_test, y_pred, average='weighted'))
            balanced_accuracy_score_list.append(balanced_accuracy_score(y_test, y_pred))
            precision_score_list.append(precision_score(y_test, y_pred, average=None))
            y_test_list.append(len(y_test))
            y_pred_list.append(len(y_pred))

            method_list.append(method)

            df = pd.DataFrame({
                'user_id': user_id_list,
                'accuracy_score': accuracy_score_list,
                'precision_score_weighted': precision_score_weighted,
                'balanced_accuracy_score': balanced_accuracy_score_list,
                'precision_score': precision_score_list,
                'y_test': len(y_test_list),
                'y_pred': len(y_pred_list),
                'method': method_list
            })

            output_file = Path("research/hybrid/classifier_eval_results.csv")
            df.to_csv(output_file.as_posix(), mode='a', header=not os.path.exists(output_file))
