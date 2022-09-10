import ast
import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from research.relevance_statistics import models_complete_statistics
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.lda import Lda
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.recommender_algorithms.learn_to_rank.learn_to_rank_methods import LightGBM


# noinspection DuplicatedCode
class HybridRecommendation:
    def __init__(self):
        self.NUM_OF_RESULTS = 20.0

    def get_hybrid_recommendation(self, slug, use_lightgbm=True, NUM_OF_CONSIDERED_MODEL_VARIANTS=4):

        stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=True)

        print(stats.columns)
        print(stats.index)
        stats_tfidf = stats.loc[stats['model_name'] == 'tfidf']
        stats_word2vec = stats.loc[stats['model_name'] == 'word2vec']
        stats_doc2vec = stats.loc[stats['model_name'] == 'doc2vec']
        stats_lda = stats.loc[stats['model_name'] == 'lda']

        print(stats_tfidf.columns)
        sorted_by_statistic = 'AP'
        stats_tfidf = stats_tfidf.sort_values(by=sorted_by_statistic, ascending=False)
        stats_word2vec = stats_word2vec.sort_values(by=sorted_by_statistic, ascending=False)
        stats_doc2vec = stats_doc2vec.sort_values(by=sorted_by_statistic, ascending=False)
        stats_lda = stats_lda.sort_values(by=sorted_by_statistic, ascending=False)

        print("stats_tfidf:")
        print(stats_tfidf)

        top_tfidf_variant = stats_tfidf.iloc[0]
        top_word2vec_variant = stats_word2vec.iloc[0]
        top_doc2vec_variant = stats_doc2vec.iloc[0]
        top_lda_variant = stats_lda.iloc[0]

        print("top_tfidf_variant:")
        print(top_tfidf_variant)
        print(top_tfidf_variant['model_name'])

        global variant
        if top_tfidf_variant['model_variant'] == "tfidf":
            variant = "short-text"
        elif top_tfidf_variant['model_variant'] == "tfidf-full-text":
            variant = "full-text"
        else:
            raise ValueError("No from selected variants matches available options.")

        tfidf = TfIdf()
        recommender_methods = RecommenderMethods()
        # TODO: Fix this
        tf_idf_results = tfidf.get_prefilled_full_text(slug=slug, variant=variant)

        tf_idf_results = ast.literal_eval(tf_idf_results)
        json_data = json.loads(json.dumps(tf_idf_results))
        tf_idf_results = pd.json_normalize(json_data)

        if top_word2vec_variant['model_variant'] == "word2vec":
            variant = 'idnes_short_text'
        elif top_word2vec_variant['model_variant'] == "word2vec-full-text":
            variant = 'idnes_full_text'
        elif top_word2vec_variant['model_variant'] == "word2vec-eval-1":
            variant = 'idnes_eval_1'
        elif top_word2vec_variant['model_variant'] == "word2vec-eval-2":
            variant = 'idnes_eval_2'
        elif top_word2vec_variant['model_variant'] == "word2vec-eval-3":
            variant = 'idnes_eval_3'
        elif top_word2vec_variant['model_variant'] == "word2vec-eval-4":
            variant = 'idnes_eval_4'
        elif top_word2vec_variant['model_variant'] == "word2vec-fasttext":
            variant = 'fasttext_limited'
        elif top_word2vec_variant['model_variant'] == "word2vec-fasttext-full-text":
            variant = 'fasttext_limited_full_text'
        elif top_word2vec_variant['model_variant'] == "word2vec-cswiki-eval-1":
            variant = 'wiki_eval_1'
        else:
            ValueError("No variant selected matches available options.")

        word2vec = Word2VecClass()
        word2vec_results = word2vec.get_prefilled_full_text(slug, variant)
        word2vec_results = ast.literal_eval(word2vec_results)
        json_data = json.loads(json.dumps(word2vec_results))
        word2vec_results = pd.json_normalize(json_data)

        if top_doc2vec_variant['model_variant'] == "doc2vec":
            variant = 'idnes_short_text'
        elif top_doc2vec_variant['model_variant'] == "doc2vec-full-text":
            variant = 'idnes_full_text'
        elif top_doc2vec_variant['model_variant'] == "doc2vec-cswiki-eval-1":
            variant = 'wiki_eval_1'
        else:
            ValueError("No variant selected matches available options.")

        doc2vec = Doc2VecClass()
        doc2vec_results = doc2vec.get_prefilled_full_text(slug, variant)
        doc2vec_results = ast.literal_eval(doc2vec_results)
        json_data = json.loads(json.dumps(doc2vec_results))
        doc2vec_results = pd.json_normalize(json_data)

        if top_lda_variant['model_variant'] == "lda":
            variant = 'idnes_short_text'
        elif top_lda_variant['model_variant'] == "lda-full-text":
            variant = 'idnes_full_text'
        elif top_lda_variant['model_variant'] == "lda-cswiki-eval-1":
            variant = 'wiki_eval_1'
        else:
            ValueError("No variant selected matches available options.")

        lda = Lda()
        lda_results = lda.get_prefilled_full_text(slug, variant)
        lda_results = ast.literal_eval(lda_results)
        json_data = json.loads(json.dumps(lda_results))
        lda_results = pd.json_normalize(json_data)

        print("tf_idf_results:")
        print(tf_idf_results)
        print("word2vec_results:")
        print(word2vec_results)
        print("doc2vec_results:")
        print(doc2vec_results)
        print("lda_results:")
        print(lda_results)

        recommender_methods = RecommenderMethods()
        relevance_results = recommender_methods.get_relevance_results_dataframe()
        print("Relevance results:")
        print(relevance_results)

        # simplified so far -- full text model = short text model

        relevance_results = relevance_results.replace("doc2vec-full-text", "doc2vec")
        relevance_results = relevance_results.replace("tfidf-full-text", "tfidf")
        relevance_results = relevance_results.replace("word2vec-full-text", "word2vec")
        relevance_results = relevance_results.replace("lda-full-text", "lda")

        print("Relevance results after rename:")
        print(relevance_results.columns)

        print(relevance_results["results_part_3"])

        # Check if model exists in results
        tfidf_found = False
        word2vec_found = False
        doc2vec_found = False
        lda_found = False
        print("checking counts:")
        if "tfidf" in relevance_results.model_name.values:
            print("TfIdf appears in results")
            tfidf_found = True
        if "word2vec" in relevance_results.model_name.values:
            print("word2vec appears in results")
            word2vec_found = True
        if "doc2vec" in relevance_results.model_name.values:
            print("doc2vec appears in results")
            doc2vec_found = True
        if "lda" in relevance_results.model_name.values:
            print("lda appears in results")
            lda_found = True

        json_data = pd.json_normalize(relevance_results.results_part_3)
        print("json_data")
        print(json_data)

        relevance_results_merged = pd.merge(relevance_results, json_data, left_index=True, right_index=True)

        relevance_results_mean = relevance_results_merged.groupby('model_name', as_index=False)[
            'mean_average_precision'].mean()

        print("relevance_results_mean")
        print(relevance_results_mean)

        map_tfidf = relevance_results_mean.loc[relevance_results_mean['model_name'] == "tfidf"]
        print("map_tfidf")
        print(map_tfidf)
        if tfidf_found is True:
            map_tfidf = map_tfidf["mean_average_precision"]
            number_of_selected_rows_from_tfidf = self.NUM_OF_RESULTS * map_tfidf
            number_of_selected_rows_from_tfidf.reset_index()
            number_of_selected_rows_from_tfidf = number_of_selected_rows_from_tfidf.values[0]
        else:
            number_of_selected_rows_from_tfidf = 0

        # It's series! .values needs to be used!
        print(number_of_selected_rows_from_tfidf)

        # derive MAP from JSON
        map_word2vec = relevance_results_mean.loc[relevance_results_mean['model_name'] == "word2vec"]
        print("map_word2vec")
        print(map_word2vec)
        if word2vec_found is True:
            map_word2vec = map_word2vec["mean_average_precision"]
            number_of_selected_rows_from_word2vec = self.NUM_OF_RESULTS * map_word2vec
            number_of_selected_rows_from_word2vec.reset_index()
            print("number_of_selected_rows_from_tfidf:")
            # It's series! .values needs to be used!
            number_of_selected_rows_from_word2vec = number_of_selected_rows_from_word2vec.values[0]
        else:
            number_of_selected_rows_from_word2vec = 0

        print("number_of_selected_rows_from_word2vec")
        print(number_of_selected_rows_from_word2vec)

        map_doc2vec = relevance_results_mean.loc[relevance_results_mean['model_name'] == "doc2vec"]
        print("map_doc2vec")
        print(map_doc2vec)
        print("doc2vec_found")
        print(doc2vec_found)
        if doc2vec_found is True:
            map_doc2vec = map_doc2vec["mean_average_precision"]
            number_of_selected_rows_from_doc2vec = 20.0 * map_doc2vec
            number_of_selected_rows_from_doc2vec.reset_index()
            print("number_of_selected_rows_from_docvec:")
            # It's series! .values needs to be used!
            number_of_selected_rows_from_doc2vec = number_of_selected_rows_from_doc2vec.values[0]
        else:
            number_of_selected_rows_from_doc2vec = 0
        print("map_doc2vec[]")
        print(number_of_selected_rows_from_doc2vec)

        map_lda = relevance_results_mean.loc[relevance_results_mean['model_name'] == "lda"]
        print("map_lda")
        print(map_lda)
        if lda_found is True:
            map_lda = map_lda["mean_average_precision"]
            number_of_selected_rows_from_lda = self.NUM_OF_RESULTS * map_lda
            number_of_selected_rows_from_lda.reset_index()
            print("number_of_selected_rows_from_lda:")
            # It's series! .values needs to be used!
            number_of_selected_rows_from_lda = number_of_selected_rows_from_lda.values[0]
        else:
            number_of_selected_rows_from_lda = 0.0

        print("number_of_selected_rows_from_lda")
        print(number_of_selected_rows_from_lda)

        total = number_of_selected_rows_from_tfidf + number_of_selected_rows_from_word2vec + number_of_selected_rows_from_doc2vec + number_of_selected_rows_from_lda

        print("total")
        print(total)

        tfidf_rate = (float(number_of_selected_rows_from_tfidf) / float(total)) * 100
        word2vec_rate = (float(number_of_selected_rows_from_word2vec) / float(total)) * 100
        doc2vec_rate = (float(number_of_selected_rows_from_doc2vec) / float(total)) * 100
        lda_rate = (float(number_of_selected_rows_from_lda) / float(total)) * 100

        print("tfidf_rate:")
        print(tfidf_rate)
        print("word2vec_rate:")
        print(word2vec_rate)
        print("doc2vec_rate:")
        print(doc2vec_rate)
        print("lda_rate:")
        print(lda_rate)

        percentage_total = tfidf_rate + word2vec_rate + doc2vec_rate + lda_rate
        print("percentage_total")
        print(percentage_total)

        number_of_recommended = 20

        tf_idf_additional_selection, word2vec_additional_selection, doc2vec_additional_selection, \
        lda_additional_selection, number_of_recommended_tfidf, number_of_recommended_word2vec, \
        number_of_recommended_doc2vec, number_of_recommended_lda = self.model_selection(tfidf_rate, word2vec_rate,
                                                                                        doc2vec_rate, lda_rate,
                                                                                        number_of_recommended,
                                                                                        tf_idf_results,
                                                                                        word2vec_results,
                                                                                        doc2vec_results, lda_results)

        final_df = pd.concat([tf_idf_additional_selection, word2vec_additional_selection, doc2vec_additional_selection,
                              lda_additional_selection])

        tf_idf_selection = tf_idf_results.head(number_of_recommended_tfidf)
        word2vec_selection = word2vec_results.head(number_of_recommended_word2vec)
        doc2vec_selection = doc2vec_results.head(number_of_recommended_doc2vec)
        lda_selection = lda_results.head(number_of_recommended_lda)

        final_df = pd.concat([tf_idf_selection, word2vec_selection, doc2vec_selection, lda_selection])

        print("final_df")
        print(final_df)

        # normalisation of coefficient
        final_df["coefficient"] = (final_df["coefficient"] - final_df["coefficient"].mean()) / final_df[
            "coefficient"].std()

        print("final_df_normalized")
        print(final_df)

        # Removing duplicate posts
        final_df = final_df.drop_duplicates(subset=['slug'])

        # Fillling free places (if any occurs)
        if len(final_df.index) == 20:
            print("final_df:")
            print(final_df)
        else:
            print("Dataframe not full")
            num_of_free_places = 20 - len(final_df.index)
            # recalculating
            print("tfidf_rate")
            print(tfidf_rate)
            print("num_of_free_places")
            print(num_of_free_places)
            num_of_free_places = 20 - final_df.index
            # recalculating

            tf_idf_additional_selection, word2vec_additional_selection, doc2vec_additional_selection, \
            lda_additional_selection, number_of_recommended_tfidf, number_of_recommended_word2vec, \
            number_of_recommended_doc2vec, number_of_recommended_lda = self.model_selection(tfidf_rate, word2vec_rate,
                                                                                            doc2vec_rate, lda_rate,
                                                                                            number_of_recommended,
                                                                                            tf_idf_results,
                                                                                            word2vec_results,
                                                                                            doc2vec_results,
                                                                                            lda_results)

            final_df = pd.concat(
                [final_df, tf_idf_additional_selection, word2vec_additional_selection, doc2vec_additional_selection,
                 lda_additional_selection])

        final_df = final_df.sort_values(by=['coefficient'], ascending=False)
        print("final_df:")
        print(final_df)

        if use_lightgbm is True:
            return self.get_posts_lightgbm(final_df)
        else:
            result = final_df.to_json(orient="records")
            parsed = json.loads(result)

            # TODO: Try if this variant works better:
            # return final_df.head(num_of_recommended).to_json(orient='records')
            return json.dumps(parsed, indent=4)

    # noinspection DuplicatedCode
    def get_posts_lightgbm(self, results, use_categorical_columns=True):

        global one_hot_encoder, categorical_columns_after_encoding
        lightgbm = LightGBM()

        consider_only_top_limit = 20
        if use_categorical_columns is True:
            one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int32)

        features = ["user_id", "coefficient", "relevance", "relevance_val", "views", "model_name"]
        categorical_columns = [
            'category_title', 'model_name'
        ]

        results['coefficient'] = results['coefficient'].astype(np.float16)

        print("tf_idf_results.dtypes:")
        print(results.dtypes)

        post_category_df = self.prepare_categories()

        print("results.columns:")
        print(results.columns)
        print("post_category_df.columns:")
        print(post_category_df.columns)

        results = results.merge(post_category_df, left_on='slug', right_on='slug')
        results = results.rename({"doc2vec_representation": "doc2vec"}, axis=1)
        df2 = pd.DataFrame(results)
        doc2vec_column_name_base = "doc2vec_col_"

        df2.dropna(subset=['doc2vec'], inplace=True)

        df2['doc2vec'] = df2['doc2vec'].apply(lambda x: json.loads(x))
        df2 = pd.DataFrame(df2['doc2vec'].to_list(), index=df2.index).add_prefix(doc2vec_column_name_base)
        for column in df2.columns:
            df2[column] = df2[column].astype(np.float16)
        results = pd.concat([results, df2], axis=1)
        print("df2.dtypes")
        print(df2.dtypes)
        del df2
        gc.collect()

        #####
        # TODO: Find and fill missing Doc2Vec values (like in the training phase)

        tf_idf_results_old = results
        if use_categorical_columns is True:
            numerical_columns = [
                "coefficient", "views", 'doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3',
                'doc2vec_col_4', 'doc2vec_col_5',
                'doc2vec_col_6', 'doc2vec_col_7'
            ]
            one_hot_encoder.fit(post_category_df[categorical_columns])
            del post_category_df
            gc.collect()
            results = lightgbm.preprocess_one_hot(results, one_hot_encoder, numerical_columns,
                                                  categorical_columns)
            results['slug'] = tf_idf_results_old['slug']
        else:
            del post_category_df
            gc.collect()

        features_X = ['coefficient', 'views']

        all_columns = ['user_id', 'query_id', 'slug', 'query_slug', 'coefficient', 'relevance', 'id_x', 'title_x',
                       'excerpt', 'body', 'views', 'keywords', 'category', 'description', 'all_features_preprocessed',
                       'body_preprocessed']
        if use_categorical_columns is True:
            categorical_columns_after_encoding = [x for x in all_columns if x.startswith("category_")]
            features.extend(categorical_columns_after_encoding)
        if use_categorical_columns is True:
            features_X.extend(categorical_columns_after_encoding)
            features_X.extend(
                ['doc2vec_col_0', 'doc2vec_col_1', 'doc2vec_col_2', 'doc2vec_col_3', 'doc2vec_col_4', 'doc2vec_col_5',
                 'doc2vec_col_6', 'doc2vec_col_7'])

        pred_df = lightgbm.make_post_feature(results)
        lightgbm_model_file = Path("models/lightgbm.pkl")
        if lightgbm_model_file.exists():
            model = pickle.load(open('models/lightgbm.pkl', 'rb'))
        else:
            print("LightGBM model not found. Training from available relevance testing results datasets...")
            lightgbm.train_lightgbm_user_based()
            model = pickle.load(open('models/lightgbm.pkl', 'rb'))
        predictions = model.predict(results[features_X])  # .values.reshape(-1,1) when single feature is used
        del results
        gc.collect()
        topk_idx = np.argsort(predictions)[::-1][:consider_only_top_limit]
        recommend_df = pred_df.loc[topk_idx].reset_index(drop=True)
        recommend_df['predictions'] = predictions

        recommend_df.sort_values(by=['predictions'], inplace=True, ascending=False)
        recommend_df = recommend_df[['slug', 'predictions']]
        recommend_df.to_json()
        result = recommend_df.to_json(orient="records")
        parsed = json.loads(result)
        return json.dumps(parsed, indent=4)

    def test_hybrid(self):
        self.get_hybrid_recommendation("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy")

    def model_selection(self, tfidf_rate, word2vec_rate, doc2vec_rate, lda_rate, number_of_recommended,
                        tf_idf_results, word2vec_results, doc2vec_results, lda_results):
        number_of_recommended_tfidf = round((tfidf_rate / 100) * number_of_recommended)
        number_of_recommended_word2vec = round((word2vec_rate / 100) * number_of_recommended)
        number_of_recommended_doc2vec = round((doc2vec_rate / 100) * number_of_recommended)
        number_of_recommended_lda = round((lda_rate / 100) * number_of_recommended)

        print("number_of_recommended_tfidf")
        print(number_of_recommended_tfidf)
        print("number_of_recommended_word2vec")
        print(number_of_recommended_word2vec)
        print("number_of_recommended_doc2vec")
        print(number_of_recommended_doc2vec)
        print("number_of_recommended_lda")
        print(number_of_recommended_lda)

        tf_idf_additional_selection = tf_idf_results.head(number_of_recommended_tfidf)
        word2vec_additional_selection = word2vec_results.head(number_of_recommended_word2vec)
        doc2vec_additional_selection = doc2vec_results.head(number_of_recommended_doc2vec)
        lda_additional_selection = lda_results.head(number_of_recommended_lda)

        return tf_idf_additional_selection, word2vec_additional_selection, doc2vec_additional_selection, \
               lda_additional_selection, number_of_recommended_tfidf, number_of_recommended_word2vec, \
               number_of_recommended_doc2vec, number_of_recommended_lda

    def prepare_categories(self):
        recommender_methods = RecommenderMethods()
        gc.collect()
        post_category_df = recommender_methods.get_posts_categories_dataframe()

        gc.collect()

        post_category_df = post_category_df.rename(columns={'slug_x': 'slug'})
        post_category_df = post_category_df.rename(columns={'title_y': 'category'})
        post_category_df['model_name'] = 'tfidf'

        return post_category_df