import pickle
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy_sentence_bert
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.constants.naming import Naming
from src.recommender_core.data_handling.data_manipulation import get_redis_connection
from src.recommender_core.data_handling.data_queries import RecommenderMethods

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)


def load_bert_model():
    bert_model = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    return bert_model


def get_df_predicted(df, target_variable_name):
    df_predicted = pd.DataFrame()
    df_predicted[target_variable_name] = df[target_variable_name]

    # leaving out 20% for validation set
    logging.debug("Splitting dataset to train_enabled / validation...")
    return df_predicted


def show_true_vs_predicted(features_list, contexts_list, clf, bert_model):
    """
    Method for evaluation on validation dataset, not actual unseen dataset.
    """
    for features_combined, context in zip(features_list, contexts_list):
        logging.debug(
            f"True Label: {context}, "
            f"Predicted Label: {clf.predict(bert_model(features_combined).vector.reshape(1, -1))[0]} \n")
        logging.debug("CONTENT:")
        logging.debug(features_combined)


def predict_from_vectors(X_unseen_df, clf, predicted_var_for_redis_key_name, user_id=None,
                         save_testing_csv=False, bert_model=None, col_to_combine=None):
    """
    Method for actual live, deployed use. This uses the already filled vectors from PostgreSQL but if doesn't
    exists, calculate new ones from passed BERT model.

    If this method takes a lot of time, use BERT vector prefilling function fill_bert_vector_representation().
    """

    if predicted_var_for_redis_key_name == Naming.PREDICTED_BY_THUMBS_REDIS_KEY_NAME:
        threshold = 1  # binary relevance rating
    elif predicted_var_for_redis_key_name == Naming.PREDICTED_BY_STARS_REDIS_KEY_NAME:
        threshold = 3  # the Likert scale
    else:
        raise ValueError("No from passed predicted rating key names matches the available options!")

    if bert_model is not None:
        if col_to_combine is None:
            raise ValueError("If BERT model is supplied, then column list needs "
                             "to be supplied to col_to_combine_parameter!")

    logging.debug("Vectoring the selected columns...")
    # TODO: Takes a lot of time... Probably pre-calculate.
    logging.debug("X_unseen_df:")
    logging.debug(X_unseen_df)

    logging.debug("Loading vectors or creating new if does not exists...")
    # noinspection  PyPep8
    y_pred_unseen = X_unseen_df \
        .apply(lambda x: clf
               .predict(pickle
                        .loads(x['bert_vector_representation']))[0]
    if pd.notnull(x['bert_vector_representation'])
    else clf.predict(bert_model(' '.join(str(x[col_to_combine]))).vector.reshape(1, -1))[0], axis=1)

    y_pred_unseen = y_pred_unseen.rename('prediction')
    df_results = pd.merge(X_unseen_df, pd.DataFrame(y_pred_unseen), how='left', left_index=True, right_index=True)
    if save_testing_csv is True:
        # noinspection PyTypeChecker
        df_results.head(20).to_csv('research/hybrid/testing_hybrid_classifier_df_results.csv')

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
            logging.debug("len(row):")
            logging.debug(len(row))
            logging.debug(user_redis_key)
            slug = "" + row[3] + ""
            logging.debug("-------------------")
            logging.debug("Predicted rating for slug | " + slug + ":")
            logging.debug(row[5])
            logging.debug(int(row[5]))

            if row[5] is not None:
                # If predicted rating is == 1 (= relevant)
                if int(row[5]) >= threshold:
                    # Saving individually to set
                    res = r.sadd(user_redis_key, slug)
                    logging.debug(res)
                    logging.debug(r.smembers(user_redis_key))
                    logging.debug("Inserted record num. " + str(i))
                    i = i + 1
            else:
                logging.debug("No predicted values found. Skipping this record.")
                pass

        logging.debug("Items saved for Redis:")
        logging.debug(r.smembers(user_redis_key))


def show_predicted(X_unseen_df, input_variables, clf, bert_model, save_testing_csv=False):
    """
    Method for evaluation on validation dataset, not actual unseen dataset.
    Use for experimentation with features.
    """
    logging.debug("Combining the selected columns")
    X_unseen_df['combined'] = X_unseen_df[input_variables].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                 axis=1)
    logging.debug("Vectorizing the selected columns...")
    y_pred_unseen = X_unseen_df['combined'].apply(lambda x: clf.predict(bert_model(x).vector.reshape(1, -1))[0])
    y_pred_unseen = y_pred_unseen.rename('prediction')
    logging.debug(y_pred_unseen.head(20))
    df_results = pd.merge(X_unseen_df, pd.DataFrame(y_pred_unseen), how='left', left_index=True, right_index=True)
    logging.debug(df_results.head(20))
    if save_testing_csv is True:
        # noinspection PyTypeChecker
        df_results.head(20).to_csv('research/hybrid/testing_hybrid_classifier_df_results.csv')


class Classifier:
    """
    Global models = models for all users
    """

    # TODO: Prepare for test_integration to API
    # TODO: Finish Python <--> PHP communication
    # TODO: Hyperparameter tuning

    def __init__(self):
        self.path_to_models_global_folder = "full_models/hybrid/classifiers/global_models"
        self.path_to_models_user_folder = "full_models/hybrid/classifiers/users_models"
        self.model_save_location = Path()
        self.bert_model = None

    def train_classifiers(self, df, columns_to_combine, target_variable_name, user_id=None):
        logging.debug("Loading Bert model...")
        self.bert_model = load_bert_model()
        # https://metatext.io/models/distilbert-base-multilingual-cased
        df_predicted = get_df_predicted(df, target_variable_name)

        df = df.fillna('')

        print("df.columns:")
        print(df.columns)

        df['combined'] = df[columns_to_combine].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        print("df['combined']")
        print(df['combined'].iloc[0])
        # noinspection PyPep8Naming
        X_train, X_validation, y_train, y_validation = train_test_split(df['combined'].tolist(),
                                                                        df_predicted[target_variable_name]
                                                                        .tolist(), test_size=0.2)
        logging.debug("Converting text to vectors...")
        df['vector'] = df['combined'].apply(lambda x: self.bert_model(x).vector)
        logging.debug("Splitting dataset to train_enabled / test...")
        # noinspection PyPep8Naming
        X_train, X_test, y_train, y_test = train_test_split(df['vector'].tolist(),
                                                            df_predicted[target_variable_name]
                                                            .tolist(), test_size=0.2)

        logging.debug("X_train:")
        logging.debug(X_train[:5])

        logging.debug("y_train")
        logging.debug(y_train[:5])

        logging.debug("Training using SVC method...")
        clf_svc = SVC(gamma='auto')
        clf_svc.fit(X_train, y_train)
        y_pred = clf_svc.predict(X_test)
        logging.debug("SVC results accuracy score:")
        logging.debug(accuracy_score(y_test, y_pred))

        logging.debug("Training using RandomForest method...")
        clf_random_forest = RandomForestClassifier(max_depth=9, random_state=0)
        clf_random_forest.fit(X_train, y_train)
        y_pred = clf_random_forest.predict(X_test)
        logging.debug("Random Forest Classifier accuracy score:")
        logging.debug(accuracy_score(y_test, y_pred))

        logging.debug("Saving the SVC model...")
        if user_id is not None:
            logging.debug("Folder: " + self.path_to_models_user_folder)
            Path(self.path_to_models_user_folder).mkdir(parents=True, exist_ok=True)

            model_file_name_svc = 'svc_classifier_' + target_variable_name + '_user_' + str(user_id) + '.pkl'
            model_file_name_random_forest = 'random_forest_classifier_' + target_variable_name + '_user_' \
                                            + str(user_id) + '.pkl'
            logging.debug(self.path_to_models_global_folder)
            logging.debug(model_file_name_svc)
            logging.debug(model_file_name_random_forest)
            path_to_models_pathlib = Path(self.path_to_models_user_folder)
            path_to_save_svc = Path.joinpath(path_to_models_pathlib, model_file_name_svc)
            joblib.dump(clf_random_forest, path_to_save_svc)
            path_to_save_forest = Path.joinpath(path_to_models_pathlib, model_file_name_random_forest)
            joblib.dump(clf_random_forest, path_to_save_forest)

        else:
            logging.debug("Folder: " + self.path_to_models_global_folder)
            Path(self.path_to_models_global_folder).mkdir(parents=True, exist_ok=True)
            model_file_name = 'svc_classifier_' + target_variable_name + '.pkl'
            logging.debug(self.path_to_models_global_folder)
            logging.debug(model_file_name)
            path_to_models_pathlib = Path(self.path_to_models_global_folder)
            path_to_save_svc = Path.joinpath(path_to_models_pathlib, model_file_name)
            joblib.dump(clf_svc, path_to_save_svc)
            logging.debug("Saving the random forest model...")
            logging.debug("Folder: " + self.path_to_models_global_folder)
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
                logging.debug("Loading user's personalized classifiers models for user " + str(user_id))
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
            logging.debug(file_not_found_error)
            logging.debug("Model file was not found in the location, training from the start...")
            self.train_classifiers(df=df, columns_to_combine=input_variables,
                                   target_variable_name=predicted_variable, user_id=user_id)
            clf_svc = joblib.load(path_to_load_svc)

        try:
            logging.debug("Loading Random Forest...")
            clf_random_forest = joblib.load(path_to_load_random_forest)
        except FileNotFoundError as file_not_found_error:
            logging.debug(file_not_found_error)
            logging.debug("Model file was not found in the location, training from the start...")
            self.train_classifiers(df=df, columns_to_combine=input_variables,
                                   target_variable_name=predicted_variable, user_id=user_id)
            clf_random_forest = joblib.load(path_to_load_random_forest)

        return clf_svc, clf_random_forest

    def predict_relevance_for_user(self, relevance_by, force_retraining=False, use_only_sample_of=None, user_id=None,
                                   experiment_mode=False, only_with_prefilled_bert_vectors=True, bert_model=None):
        if only_with_prefilled_bert_vectors is False:
            if bert_model is None:
                raise ValueError("Loaded BERT model needs to be supplied if only_with_prefilled_bert_vectors parameter"
                                 "is set to False")

        columns_to_combine = ['category_title', 'all_features_preprocessed', 'full_text']

        recommender_methods = RecommenderMethods()
        all_user_df = recommender_methods.get_all_users()

        print("all_user_df.columns")
        print(all_user_df.columns)

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
            target_variable_name = 'thumbs_values'
            predicted_var_for_redis_key_name = Naming.PREDICTED_BY_THUMBS_REDIS_KEY_NAME
        elif relevance_by == 'stars':
            df_posts_users_categories_relevance = recommender_methods \
                .get_posts_users_categories_ratings_df(user_id=user_id,
                                                       only_with_bert_vectors=only_with_prefilled_bert_vectors)
            target_variable_name = 'ratings_values'
            predicted_var_for_redis_key_name = Naming.PREDICTED_BY_STARS_REDIS_KEY_NAME
        else:
            raise ValueError("No options from allowed relevance options selected.")

        df_posts_categories = recommender_methods \
            .get_posts_categories_dataframe(only_with_bert_vectors=only_with_prefilled_bert_vectors,
                                            from_cache=False)

        df_posts_categories = df_posts_categories.rename(columns={'title': 'category_title'})

        if force_retraining is True:
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
            X_validation = df_posts_categories[columns_to_combine]
            if not type(use_only_sample_of) is None:
                if type(use_only_sample_of) is int:
                    # noinspection PyPep8Naming
                    X_validation = X_validation.sample(use_only_sample_of)
            logging.debug("Loading sentence bert multilingual model...")
            logging.debug("=========================")
            logging.debug("Results of SVC:")
            logging.debug("=========================")
            show_predicted(X_unseen_df=X_validation, input_variables=columns_to_combine, clf=clf_svc,
                           bert_model=bert_model)
            logging.debug("=========================")
            logging.debug("Results of Random Forest:")
            logging.debug("=========================")
            show_predicted(X_unseen_df=X_validation, input_variables=columns_to_combine, clf=clf_random_forest,
                           bert_model=bert_model)
        else:
            columns_to_select = columns_to_combine + ['slug', 'bert_vector_representation']
            # noinspection PyPep8Naming
            X_validation = df_posts_categories[columns_to_select]
            if not type(use_only_sample_of) is None:
                if type(use_only_sample_of) is int:
                    # noinspection PyPep8Naming
                    X_validation = X_validation.sample(use_only_sample_of)
            logging.debug("=========================")
            logging.debug("Inserting by SVC:")
            logging.debug("=========================")
            predict_from_vectors(X_unseen_df=X_validation, clf=clf_svc, user_id=user_id,
                                 predicted_var_for_redis_key_name=predicted_var_for_redis_key_name,
                                 bert_model=bert_model, col_to_combine=columns_to_combine,
                                 save_testing_csv=True)

            logging.debug("=========================")
            logging.debug("Inserting by Random Forest:")
            logging.debug("=========================")
            predict_from_vectors(X_unseen_df=X_validation, clf=clf_random_forest, user_id=user_id,
                                 predicted_var_for_redis_key_name=predicted_var_for_redis_key_name,
                                 bert_model=bert_model, col_to_combine=columns_to_combine,
                                 save_testing_csv=True)
