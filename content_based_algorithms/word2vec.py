import json
import random

import gensim
import psycopg2
from gensim.models import KeyedVectors

from content_based_algorithms.data_queries import RecommenderMethods
from content_based_algorithms.doc_sim import DocSim
from content_based_algorithms.helper import NumpyEncoder
import pandas as pd
import time as t

from data_conenction import Database


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
        documents_df["features_to_use"] = self.df["keywords"] + '||' + self.df["title_y"] + ' ' + self.df[
            "all_features_preprocessed"]
        documents_df["slug"] = self.df["slug_x"]
        found_post = found_post_dataframe['features_to_use'].iloc[0]

        del self.df
        del found_post_dataframe

        word2vec_embedding = KeyedVectors.load("models/w2v_model_limited")

        ds = DocSim(word2vec_embedding)

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
        word2vec_embedding = KeyedVectors.load_word2vec_format("full_models/w2v_model_full", limit=87000)  #
        word2vec_embedding.save("models/w2v_model_limited")  # write separately=[] for all_in_one model

    def save_fast_text_to_w2v(self):
        print("Loading and saving FastText pretrained model to Word2Vec model")
        word2vec_model = gensim.models.fasttext.load_facebook_vectors("full_models/cc.cs.300.bin.gz", encoding="utf-8")
        print("FastText loaded...")
        word2vec_model.fill_norms()
        word2vec_model.save_word2vec_format("full_models/w2v_model_full")
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
