# RUN WITH: python -m pytest tests/test_integration/test_recommender_methods/test_hybrid_methods.py::TestClassifier
import os
from pathlib import Path
from unittest import TestCase

import pytest

from src.constants.naming import Naming
from src.recommender_core.data_handling.data_manipulation import get_redis_connection
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    Classifier, predict_from_vectors


def train_and_predict(classifier, df_posts_users_categories_relevance, columns_to_combine,
                      target_variable_name, redis_key_name, user_id, only_with_prefilled_bert_vectors):
    recommender_methods = RecommenderMethods()
    clf_svc, clf_random_forest, X_validation, y_validation, bert_model \
        = classifier.train_classifiers(df=df_posts_users_categories_relevance,
                                       columns_to_combine=columns_to_combine,
                                       target_variable_name=target_variable_name, user_id=user_id)

    try:
        predict_from_vectors(X_unseen_df=X_validation, clf=clf_svc, user_id=user_id,
                             predicted_var_for_redis_key_name=redis_key_name,
                             bert_model=bert_model, col_to_combine=columns_to_combine,
                             save_testing_csv=True, testing_mode=True)
    except AttributeError as e:
        print("AttributeError exception:")
        print(e)
        df_posts_categories = recommender_methods \
            .get_posts_categories_dataframe(only_with_bert_vectors=only_with_prefilled_bert_vectors,
                                            from_cache=False)

        df_posts_categories = df_posts_categories.rename(columns={'title': 'category_title'})
        use_only_sample_of = 20
        columns_to_select = columns_to_combine + ['slug', 'bert_vector_representation']
        # noinspection PyPep8Naming
        X_validation = df_posts_categories[columns_to_select]
        if not type(use_only_sample_of) is None:
            if type(use_only_sample_of) is int:
                # noinspection PyPep8Naming
                X_validation = X_validation.sample(use_only_sample_of)
        predict_from_vectors(X_unseen_df=X_validation, clf=clf_svc, user_id=user_id,
                             predicted_var_for_redis_key_name=redis_key_name,
                             bert_model=bert_model, col_to_combine=columns_to_combine,
                             save_testing_csv=True, testing_mode=True)
    except Exception as e:
        raise e




# RUN WITH:
# python -m pytest tests/test_integration/test_recommender_methods/test_user_relevance_classifier.py::TestClassifier
class TestClassifier(TestCase):

    # TODO: This is really nasty test with duplicated code. Try to simplify logic
    def predict_from_vectors_testing(self, redis_key_name, target_variable_name, user_id):
        columns_to_combine = ['category_title', 'all_features_preprocessed', 'full_text']

        if target_variable_name == 'ratings_values':
            path_to_csv_test_file = "tests/testing_datasets/testing_posts_categories_stars_data_for_df.csv"
        elif target_variable_name == 'thumbs_values':
            path_to_csv_test_file = "tests/testing_datasets/testing_posts_categories_thumbs_data_for_df.csv"
        else:
            raise NotImplementedError("This target variable is not implemented")

        recommender_methods = RecommenderMethods()
        only_with_prefilled_bert_vectors = True

        if target_variable_name == 'thumbs_values':

            # STARS VALUES
            df_posts_users_categories_relevance = recommender_methods \
                .get_posts_users_categories_thumbs_df(user_id=user_id,
                                                      only_with_bert_vectors=only_with_prefilled_bert_vectors)
            df_posts_users_categories_relevance = df_posts_users_categories_relevance.head(20)
            df_posts_users_categories_relevance \
                .to_csv(Path(path_to_csv_test_file))

            classifier = Classifier()

            print("df_posts_users_categories_relevance")
            print(df_posts_users_categories_relevance)
            try:
                train_and_predict(classifier, df_posts_users_categories_relevance, columns_to_combine,
                                  target_variable_name, redis_key_name, user_id, only_with_prefilled_bert_vectors)
            except Exception as e:
                raise e

        elif target_variable_name == 'ratings_values':
            # THUMBS VALUES
            df_posts_users_categories_relevance = recommender_methods \
                .get_posts_users_categories_ratings_df(user_id=user_id,
                                                       only_with_bert_vectors=only_with_prefilled_bert_vectors)
            df_posts_users_categories_relevance = df_posts_users_categories_relevance.head(20)
            df_posts_users_categories_relevance \
                .to_csv(Path(path_to_csv_test_file))

            target_variable_name = 'ratings_values'

            classifier = Classifier()

            try:
                train_and_predict(classifier, df_posts_users_categories_relevance, columns_to_combine,
                                  target_variable_name, redis_key_name, user_id, only_with_prefilled_bert_vectors)
            except Exception as e:
                raise e

        else:
            raise NotImplementedError("Check the target_variable_name argument. "
                                      "Passed target_variable_name is not implemented!")

    def prepare_redis(self, user_id, predicted_by_redis_key_name):
        user_redis_key = 'user' + Naming.REDIS_DELIMITER + str(user_id) + Naming.REDIS_DELIMITER \
                         + 'post-classifier-by-' + predicted_by_redis_key_name
        r = get_redis_connection()
        try:
            r.delete(user_redis_key)
        except Exception as e:
            print("Redis delete exception:")
            print(e)

        return r, user_redis_key

    # Bad Day
    def test_predict_from_vectors(self):

        user_id = 999999
        target_variable_name = 'thumbs_values'
        malformed_redis_key_name = "malformed_redis_key_name"
        with pytest.raises(ValueError):
            self.predict_from_vectors_testing(malformed_redis_key_name, target_variable_name, user_id)

        target_variable_name = 'ratings_values'
        malformed_redis_key_name = "malformed_redis_key_name"
        with pytest.raises(ValueError):
            self.predict_from_vectors_testing(malformed_redis_key_name, target_variable_name, user_id)

        model_file_name_classifier_random_forrest = 'random_forest_classifier_' + target_variable_name + '_user_' \
                                                    + str(user_id) + '.pkl'

        path_to_classifier_folder = 'full_models/hybrid/classifiers/users_models/'
        path_to_classifier_model = path_to_classifier_folder + model_file_name_classifier_random_forrest
        assert os.path.exists(Path(path_to_classifier_model).as_posix())

        model_file_name_classifier_random_forrest = 'svc_classifier_' + target_variable_name + '_user_' \
                                                    + str(user_id) + '.pkl'

        path_to_classifier_folder = 'full_models/hybrid/classifiers/users_models/'
        path_to_classifier_model = path_to_classifier_folder + model_file_name_classifier_random_forrest
        assert os.path.exists(Path(path_to_classifier_model).as_posix())

        target_variable_name = 'thumbs_values'
        predicted_by_thumbs_redis_key_name = 'thumbs-ratings'

        r, user_redis_key = self.prepare_redis(user_id, predicted_by_redis_key_name=predicted_by_thumbs_redis_key_name)

        assert (r.exists(user_redis_key) == 0)
        self.predict_from_vectors_testing(predicted_by_thumbs_redis_key_name, target_variable_name, user_id)
        assert (r.exists(user_redis_key) > 0)

        # self.teardown(user_redis_key, r)

        target_variable_name = 'ratings_values'
        predicted_by_stars_redis_key_name = 'stars-ratings'

        r, user_redis_key = self.prepare_redis(user_id, predicted_by_redis_key_name=predicted_by_stars_redis_key_name)

        assert (r.exists(user_redis_key) == 0)
        self.predict_from_vectors_testing(predicted_by_stars_redis_key_name, target_variable_name, user_id)
        assert (r.exists(user_redis_key) > 0)

        def teardown(redis_key, redis):
            redis.delete(redis_key)
            print("Teardown")

        teardown(user_redis_key, r)
