import csv
import json
import os
import random

import gensim
import numpy as np
import psycopg2
import tqdm
from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import common_texts

from content_based_algorithms import data_queries
from content_based_algorithms.data_queries import RecommenderMethods
from content_based_algorithms.doc_sim import DocSim
from content_based_algorithms.helper import NumpyEncoder, Helper
import pandas as pd
import time as t

from data_conenction import Database
from preprocessing.cz_preprocessing import CzPreprocess


class Word2VecClass:
    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/w2v_embedding_all_in_one"

    def __init__(self):
        self.documents = None
        self.df = None
        self.database = Database()

    def get_posts_dataframe(self):
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    def get_categories_dataframe(self):
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe(pd)
        self.database.disconnect()
        return self.categories_df

    def join_posts_ratings_categories(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
             'all_features_preprocessed']]

    def join_posts_ratings_categories_full_text(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
             'all_features_preprocessed', 'body_preprocessed', 'full_text']]

    def eval_word2vec_idnes_model(self):
        recommenderMethods = RecommenderMethods()

        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        del self.posts_df
        del self.categories_df

        documents_df = pd.DataFrame()
        documents_training_df = pd.DataFrame()

        documents_df["features_to_use"] = self.df["title_y"] + " " + self.df["keywords"] + ' ' + self.df[
            "all_features_preprocessed"]
        documents_df["slug"] = self.df["slug_x"]

        documents_training_df["features_to_use"] = self.df["title_y"] + " " + self.df["keywords"] + " " + self.df[
            "all_features_preprocessed"]
        documents_training_df["features_to_use"] = documents_training_df["features_to_use"].replace(",", "")
        documents_training_df["features_to_use"] = documents_training_df["features_to_use"].str.split(" ")

        texts = documents_training_df["features_to_use"].tolist()
        texts = data_queries.remove_stopwords(texts)

        del self.df

        # documents_df['features_combined'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        # documents = list(map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        # Uncomment for change of model
        # self.refresh_model()

        # word2vec_embedding = KeyedVectors.load(self.amazon_bucket_url)
        # self.amazon_bucket_url#

        print("Loading Word2Vec FastText model...")
        word2vec_embedding = KeyedVectors.load("models/w2v_model_limited")
        # word2vec_embedding = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
        # word2vec_embedding = KeyedVectors.load(self.amazon_bucket_url)
        # global word2vec_embedding
        self.eval_word2vec(texts)

    # @profile
    def get_similar_word2vec(self, searched_slug):
        recommenderMethods = RecommenderMethods()

        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        found_post_dataframe = recommenderMethods.find_post_by_slug(searched_slug)
        found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                  found_post_dataframe.iloc[0]['title_y'] + " " + \
                                                  found_post_dataframe.iloc[0]['all_features_preprocessed']

        del self.posts_df
        del self.categories_df

        documents_df = pd.DataFrame()
        documents_training_df = pd.DataFrame()

        documents_df["features_to_use"] = self.df["title_y"] + " " + self.df["keywords"] + ' ' + self.df[
            "all_features_preprocessed"]
        documents_df["slug"] = self.df["slug_x"]
        found_post = found_post_dataframe['features_to_use'].iloc[0]

        documents_training_df["features_to_use"] = self.df["title_y"] + " " + self.df["keywords"] + " " + self.df["all_features_preprocessed"]
        documents_training_df["features_to_use"] = documents_training_df["features_to_use"].replace(",", "")
        documents_training_df["features_to_use"] = documents_training_df["features_to_use"].str.split(" ")

        texts = documents_training_df["features_to_use"].tolist()
        texts = data_queries.remove_stopwords(texts)

        del self.df
        del found_post_dataframe

        # documents_df['features_combined'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        # documents = list(map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        # Uncomment for change of model
        # self.refresh_model()

        # word2vec_embedding = KeyedVectors.load(self.amazon_bucket_url)
        # self.amazon_bucket_url

        print("Loading Word2Vec FastText model...")
        word2vec_embedding = KeyedVectors.load("models/w2v_model_limited")
        # word2vec_embedding = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
        # word2vec_embedding = KeyedVectors.load(self.amazon_bucket_url)
        # global word2vec_embedding
        model_idnes = self.eval_word2vec(texts, word2vec_embedding)

        ds = DocSim(word2vec_embedding)

        # del word2vec_embedding
        # documents_df['features_to_use'] = documents_df.replace(',','', regex=True)

        documents_df['features_to_use'] = documents_df['features_to_use'] + "; " + documents_df['slug']
        list_of_document_features = documents_df["features_to_use"].tolist()
        del documents_df
        # https://github.com/v1shwa/document-similarity with my edits
        print("Similarities on Wikipedia.cz model:")
        most_similar_articles_with_scores = ds.calculate_similarity_wiki_model_gensim(found_post,
                                                                    list_of_document_features)[:21]

        print("Similarities on iDNES.cz model:")
        ds = DocSim(model_idnes)
        most_similar_articles_with_scores = ds.calculate_similarity_idnes_model_gensim(found_post,
                                                                    list_of_document_features)[:21]
        # removing post itself
        del most_similar_articles_with_scores[0]  # removing post itself

        # workaround due to float32 error in while converting to JSON
        return json.loads(json.dumps(most_similar_articles_with_scores, cls=NumpyEncoder))

    # @profile
    def get_similar_word2vec_full_text(self, searched_slug):
        recommenderMethods = RecommenderMethods()

        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories_full_text()
        # post_found = self.(search_slug)

        # search_terms = 'Domácí. Zemřel poslední krkonošský nosič Helmut Hofer, ikona Velké Úpy. Ve věku 88 let zemřel potomek slavného rodu vysokohorských nosičů Helmut Hofer z Velké Úpy. Byl posledním žijícím nosičem v Krkonoších, starodávným řemeslem se po staletí živili generace jeho předků. Jako nosič pracoval pro Českou boudu na Sněžce mezi lety 1948 až 1953.'
        found_post_dataframe = recommenderMethods.find_post_by_slug(searched_slug)
        found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                  found_post_dataframe.iloc[0]['title_y'] + " " + \
                                                  found_post_dataframe.iloc[0]['all_features_preprocessed'] + " " + \
                                                  found_post_dataframe.iloc[0]['body_preprocessed']

        del self.posts_df
        del self.categories_df

        # cols = ["title_y", "title_x", "excerpt", "keywords", "slug_x", "all_features_preprocessed"]
        documents_df = pd.DataFrame()

        documents_df["features_to_use"] = self.df["keywords"] + '||' + self.df["title_y"] + ' ' + self.df[
            "all_features_preprocessed"] + ' ' + self.df["body_preprocessed"]
        documents_df["slug"] = self.df["slug_x"]
        found_post = found_post_dataframe['features_to_use'].iloc[0]

        del self.df
        del found_post_dataframe

        print("Loading word2vec model...")

        try:
            word2vec_embedding = KeyedVectors.load("models/w2v_model_limited")
        except FileNotFoundError:
            print("Downloading from Dropbox...")
            dropbox_access_token = "njfHaiDhqfIAAAAAAAAAAX_9zCacCLdpxxXNThA69dVhAsqAa_EwzDUyH1ZHt5tY"
            recommenderMethods = RecommenderMethods()
            recommenderMethods.dropbox_file_download(dropbox_access_token, "models/w2v_model_limited.vectors.npy",
                                  "/w2v_model.vectors.npy")
            word2vec_embedding = KeyedVectors.load("models/w2v_model_limited")

        ds = DocSim(word2vec_embedding)

        documents_df['features_to_use'] = documents_df['features_to_use'].str.replace(';', ' ')
        documents_df['features_to_use'] = documents_df['features_to_use'].str.replace(r'\r\n', '', regex=True)
        documents_df['features_to_use'] = documents_df['features_to_use'] + "; " + documents_df['slug']
        list_of_document_features = documents_df["features_to_use"].tolist()
        del documents_df
        # https://github.com/v1shwa/document-similarity with my edits

        most_similar_articles_with_scores = ds.calculate_similarity(found_post,
                                                                    list_of_document_features)[:21]
        # removing post itself
        del most_similar_articles_with_scores[0]  # removing post itself

        # workaround due to float32 error in while converting to JSON
        return json.loads(json.dumps(most_similar_articles_with_scores, cls=NumpyEncoder))

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def save_full_model_to_smaller(self):
        print("Saving full model to limited model...")
        word2vec_embedding = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full", limit=87000)  #
        word2vec_embedding.save("models/w2v_model_limited")  # write separately=[] for all_in_one model

    def save_fast_text_to_w2v(self):
        print("Loading and saving FastText pretrained model to Word2Vec model")
        word2vec_model = gensim.models.fasttext.load_facebook_vectors("full_models/cswiki/cc.cs.300.bin.gz", encoding="utf-8")
        print("FastText loaded...")
        word2vec_model.fill_norms()
        word2vec_model.save_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
        print("Fast text saved...")

    def fill_recommended_for_all_posts(self, skip_already_filled, full_text=True, random_order=False, reversed=False):

        database = Database()
        database.connect()
        if skip_already_filled is False:
            posts = database.get_all_posts()
        else:
            posts = database.get_not_prefilled_posts(full_text)

        number_of_inserted_rows = 0

        if reversed is True:
            print("Reversing list of posts...")
            posts.reverse()

        if random_order is True:
            print("Starting random iteration...")
            t.sleep(5)
            random.shuffle(posts)

        for post in posts:
            if len(posts) < 1:
                break
            print("post")
            print(post[22])
            post_id = post[0]
            slug = post[3]
            if full_text is False:
                current_recommended = post[22]
            else:
                current_recommended = post[23]

            print("Searching similar articles for article: " + slug)

            if skip_already_filled is True:
                if current_recommended is None:
                    if full_text is False:
                        actual_recommended_json = self.get_similar_word2vec(slug)
                    else:
                        actual_recommended_json = self.get_similar_word2vec_full_text(slug)
                    actual_recommended_json = json.dumps(actual_recommended_json)
                    if full_text is False:
                        try:
                            database.insert_recommended_word2vec_json(articles_recommended_json=actual_recommended_json,
                                                                      article_id=post_id)
                        except:
                            print("Error in DB insert. Skipping.")
                            pass
                    else:
                        try:
                            database.insert_recommended_word2vec_full_json(
                                articles_recommended_json=actual_recommended_json, article_id=post_id)
                        except:
                            print("Error in DB insert. Skipping.")
                            pass
                    number_of_inserted_rows += 1
                    if number_of_inserted_rows > 20:
                        print("Refreshing list of posts for finding only not prefilled posts.")
                        if full_text is False:
                            self.fill_recommended_for_all_posts(skip_already_filled=True, full_text=False)
                        else:
                            self.fill_recommended_for_all_posts(skip_already_filled=True, full_text=True)
                    # print(str(number_of_inserted_rows) + " rows insertd.")
                else:
                    print("Skipping.")
            else:
                if full_text is False:
                    actual_recommended_json = self.get_similar_word2vec(slug)
                else:
                    actual_recommended_json = self.get_similar_word2vec_full_text(slug)
                actual_recommended_json = json.dumps(actual_recommended_json)
                if full_text is False:
                    database.insert_recommended_word2vec_json(articles_recommended_json=actual_recommended_json,
                                                              article_id=post_id)
                else:
                    database.insert_recommended_word2vec_full_json(articles_recommended_json=actual_recommended_json,
                                                                   article_id=post_id)
                number_of_inserted_rows += 1
                # print(str(number_of_inserted_rows) + " rows insertd.")

    def prefilling_job(self, full_text, reverse, random=False):
        if full_text is False:
            for i in range(100):
                while True:
                    try:
                        self.fill_recommended_for_all_posts(skip_already_filled=True, full_text=False)
                    except psycopg2.OperationalError:
                        print("DB operational error. Waiting few seconds before trying again...")
                        t.sleep(30)  # wait 30 seconds then try again
                        continue
                    break
        else:
            for i in range(100):
                while True:
                    try:
                        self.fill_recommended_for_all_posts(skip_already_filled=True, full_text=True)
                    except psycopg2.OperationalError:
                        print("DB operational error. Waiting few seconds before trying again...")
                        t.sleep(30)  # wait 30 seconds then try again
                        continue
                    break

    def refresh_model(self):
        self.save_fast_text_to_w2v()
        print("Loading word2vec model...")
        self.save_full_model_to_smaller()

    def eval_word2vec(self, texts):
            model_variants = [0,1] # sg parameter: 0 = CBOW; 1 = Skip-Gram
            hs_softmax_variants = [0,1] # 1 = Hierarchical SoftMax
            negative_sampling_variants = range(5,20,5) # 0 = no negative sampling
            no_negative_sampling = 0 # use with hs_soft_max
            vector_size_range = [50,100,158,200,250,300,450]
            window_range = [1,2,4,5,8,12,16,20]
            min_count_range = [0,2,4,5,8,12,16,20]
            epochs_range = [2,5,10,15,20,25,30]
            sample_range = [0.0, 1.0*(10.0**-1.0), 1.0*(10.0**-2.0), 1.0*(10.0**-3.0), 1.0*(10.0**-4.0), 1.0*(10.0**-5.0)] # useful range is (0, 1e-5) acording to : https://radimrehurek.com/gensim/models/word2vec.html

            corpus_title = ['100% Corpus']
            model_results = {'Validation_Set': [],
                             'Model_Variant': [],
                             'Softmax': [],
                             'Negative': [],
                             'Vector_size': [],
                             'Window': [],
                             'Epochs': [],
                             'Sample': [],
                             'Word_pairs_test_Pearson_coeff': [],
                             'Word_pairs_test_Pearson_p-val': [],
                             'Word_pairs_test_Spearman_coeff': [],
                             'Word_pairs_test_Spearman_p-val': [],
                             'Word_pairs_test_Out-of-vocab_ratio': [],
                             'Analogies_test': []
                             }  # Can take a long time to run
            pbar = tqdm.tqdm(total=540)
            for model_variant in model_variants:
                for negative_sampling_variant in negative_sampling_variants:
                    for vector_size in vector_size_range:
                        for window in window_range:
                            for min_count in min_count_range:
                                for epochs in epochs_range:
                                    for sample in sample_range:
                                        for hs_softmax in hs_softmax_variants:
                                            if hs_softmax == 1:
                                                word_pairs_eval, analogies_eval = self.compute_eval_values(sentences=texts, model_variant=model_variant, negative_sampling_variant=no_negative_sampling,
                                                                         vector_size=vector_size, window=window, min_count=min_count,
                                                                         epochs=epochs, sample=sample)
                                            else:
                                                word_pairs_eval, analogies_eval = self.compute_eval_values(sentences=texts, model_variant=model_variant, negative_sampling_variant=negative_sampling_variant,
                                                                                    vector_size=vector_size, window=window, min_count=min_count,
                                                                                    epochs=epochs, sample=sample)

                                            print(word_pairs_eval[0][0])
                                            model_results['Validation_Set'].append("iDnes.cz " + corpus_title[0])
                                            model_results['Model_Variant'].append(model_variant)
                                            model_results['Softmax'].append(hs_softmax)
                                            model_results['Negative'].append(negative_sampling_variant)
                                            model_results['Vector_size'].append(vector_size)
                                            model_results['Window'].append(window)
                                            model_results['Epochs'].append(epochs)
                                            model_results['Sample'].append(sample)
                                            model_results['Word_pairs_test_Pearson_coeff'].append(word_pairs_eval[0][0])
                                            model_results['Word_pairs_test_Pearson_p-val'].append(word_pairs_eval[0][1])
                                            model_results['Word_pairs_test_Spearman_coeff'].append(word_pairs_eval[1][0])
                                            model_results['Word_pairs_test_Spearman_p-val'].append(word_pairs_eval[1][1])
                                            model_results['Word_pairs_test_Out-of-vocab_ratio'].append(word_pairs_eval[2])
                                            model_results['Analogies_test'].append(analogies_eval)

                                            pbar.update(1)
                                            pd.DataFrame(model_results).to_csv('word2vec_tuning_results.csv', index=False,
                                                                               mode="a")
                                            print("Saved training results...")
            pbar.close()

    def compute_eval_values(self, sentences, model_variant, negative_sampling_variant, vector_size, window, min_count,
                                                                     epochs, sample, force_update_model=True):
        if os.path.isfile("models/w2v_model_idnes.model") is False or force_update_model is True:

            print("Started training on iDNES.cz dataset...")
            # w2v_model = Word2Vec(sentences=sentences, sg=model_variant, negative=negative_sampling_variant, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, sample=sample, workers=7)
            w2v_model = Word2Vec(sentences=sentences, sg=model_variant, negative=negative_sampling_variant, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, sample=sample, workers=7)

            # DEFAULT:
            # model = Word2Vec(sentences=texts, vector_size=158, window=5, min_count=5, workers=7, epochs=15)
            w2v_model.save("models/w2v_model_idnes.model")
        else:
            print("Loading Word2Vec iDNES.cz model from saved model file")
            w2v_model = Word2Vec.load("models/w2v_model_idnes.model")

        import pandas as pd
        df = pd.read_csv('research/word2vec/similarities/WordSim353-cs.csv', usecols=['cs_word_1', 'cs_word_2', 'cs mean'])
        df.to_csv('research/word2vec/similarities/WordSim353-cs-cropped.tsv', sep='\t', encoding='utf-8', index=False)

        print("Word pairs evaluation iDnes.cz model:")
        word_pairs_eval = w2v_model.wv.evaluate_word_pairs('research/word2vec/similarities/WordSim353-cs-cropped.tsv')
        print(word_pairs_eval)

        overall_score, _ = w2v_model.wv.evaluate_word_analogies('research/word2vec/analogies/questions-words-cs.txt')
        print("Analogies evaluation of iDnes.cz model:")
        print(overall_score)

        return word_pairs_eval, overall_score

    def eval_wiki(self):
        topn = 30
        limit = 350000

        print("Loading Wikipedia full model...")
        wiki_full_model = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full", limit=limit)

        searched_word = 'hokej'
        sims = wiki_full_model.most_similar(searched_word, topn=topn)  # get other similar words
        print("TOP " + str(topn) + " similar Words from Wikipedia.cz for word " + searched_word + ":")
        print(sims)

        searched_word = 'zimák'
        if searched_word in wiki_full_model.key_to_index:
            print(searched_word)
            print("Exists in vocab")
        else:
            print(searched_word)
            print("Doesn't exists in vocab")

        helper = Helper()
        # self.save_tuple_to_csv("research/word2vec/most_similar_words/cswiki_top_10_similar_words_to_hokej.csv", sims)
        self.save_tuple_to_csv("cswiki_top_" + str(topn) + "_similar_words_to_hokej_limit_" + str(limit) + ".csv", sims)

        print("Word pairs evaluation FastText on Wikipedia.cz model:")
        print(wiki_full_model.evaluate_word_pairs('research/word2vec/similarities/WordSim353-cs-cropped.tsv'))

        overall_analogies_score, _ = wiki_full_model.evaluate_word_analogies("research/word2vec/analogies/questions-words-cs.txt")
        print("Analogies evaluation of FastText on Wikipedia.cz model:")
        print(overall_analogies_score)

    def save_tuple_to_csv(self, path, data):
        with open(path, 'w+') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['word', 'sim'])
            for row in data:
                csv_out.writerow(row)

def main():
    searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"  # print(doc2vecClass.get_similar_doc2vec(slug))
    # searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    # searched_slug = "facr-o-slavii-a-rangers-verime-v-objektivni-vysetreni-odmitame-rasismus"

    word2vecClass = Word2VecClass()
    start = t.time()
    print(word2vecClass.get_similar_word2vec(searched_slug))
    end = t.time()
    print("Elapsed time: " + str(end - start))
    start = t.time()
    print(word2vecClass.get_similar_word2vec_full_text(searched_slug))
    end = t.time()
    print("Elapsed time: " + str(end - start))
