import csv
import gc
import json
import logging
import os
import pickle
import random
import re
import string
import time
from collections import defaultdict

import gensim
import majka
import numpy as np
import psycopg2
import pymongo as pymongo
import regex
import tqdm
from gensim import corpora
from gensim.models import KeyedVectors, Word2Vec, LdaModel
from gensim.test.utils import common_texts
from gensim.utils import deaccent
from html2text import html2text
from nltk import FreqDist, RegexpTokenizer

from content_based_algorithms import data_queries
from content_based_algorithms.data_queries import RecommenderMethods
from content_based_algorithms.doc_sim import DocSim
from content_based_algorithms.helper import NumpyEncoder, Helper
import pandas as pd
import time as t

from data_connection import Database
from preprocessing.cz_preprocessing import CzPreprocess, cz_stopwords, general_stopwords

myclient = pymongo.MongoClient('localhost', 27017)
db = myclient.test
mongo_db = myclient["idnes"]
mongo_collection = mongo_db["preprocessed_articles"]
mongo_collection_stopwords_free = mongo_db["preprocessed_articles_stopwords_free"]
mongo_collection_bigrams = mongo_db["preprocessed_articles_bigrams"]


def save_to_mongo(data, number_of_processed_files, mongo_collection):
    dict_to_insert = dict({"number": number_of_processed_files, "text": data})
    mongo_collection.insert_one(dict_to_insert)


class MyCorpus(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __iter__(self):
        print("Loading bigrams from preprocessed articles...")
        reader = MongoReader(dbName='idnes', collName='preprocessed_articles_bigrams')
        print("Creating Doc2Bow...")
        i = 1
        for doc in reader.iterate():
            # assume there's one document per line, tokens separated by whitespace
            print("\rDoc. num. " + str(i), end='')
            yield self.dictionary.doc2bow(doc.get('text'))
            i = i + 1


@DeprecationWarning
class CzLemma:

    def __init__(self):
        self.df = None
        self.categories_df = None

    # pre-worked
    def preprocess(self, sentence, stemming=False, lemma=True):
        # print(sentence)
        sentence = str(sentence)
        sentence = sentence.lower()
        sentence = sentence.replace('{html}', "")
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        cleantext.translate(str.maketrans('', '', string.punctuation))  # removing punctuation

        a_string = cleantext.split('=References=')[0]  # remove references and everything afterwards
        a_string = html2text(a_string).lower()  # remove HTML tags, convert to lowercase
        a_string = re.sub(r'https?:\/\/.*?[\s]', '', a_string)  # remove URLs

        # 'ToktokTokenizer' does divide by '|' and '\n', but retaining this
        #   statement seems to improve its speed a little
        a_string = a_string.replace('|', ' ').replace('\n', ' ')

        rem_url = re.sub(r'http\S+', '', cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        # print("rem_num")
        # print(rem_num)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)
        # print("tokens")
        # print(tokens)

        tokens = [w for w in tokens if '=' not in w]  # remove remaining tags and the like
        string_punctuation = list(string.punctuation)
        tokens = [w for w in tokens if not
        all(x.isdigit() or x in string_punctuation for x in w)] # remove tokens that are all punctuation
        tokens = [w.strip(string.punctuation) for w in tokens]  # remove stray punctuation attached to words
        tokens = [w for w in tokens if len(w) > 1]  # remove single characters
        tokens = [w for w in tokens if not any(x.isdigit() for x in w)]  # remove everything with a digit in it

        edited_words = [self.cz_lemma(w) for w in tokens]
        edited_words = list(filter(None, edited_words))  # empty strings removal

        # removing stopwords
        edited_words = [word for word in edited_words if word not in cz_stopwords]
        edited_words = [word for word in edited_words if word not in general_stopwords]

        return " ".join(edited_words)

    @DeprecationWarning
    def preprocess_single_post_find_by_slug(self, slug, json=False, stemming=False):
        recommenderMethods = RecommenderMethods()
        post_dataframe = recommenderMethods.find_post_by_slug(slug)
        post_dataframe["title"] = post_dataframe["title"].map(lambda s: self.preprocess(s, stemming))
        post_dataframe["excerpt"] = post_dataframe["excerpt"].map(lambda s: self.preprocess(s, stemming))
        if json is False:
            return post_dataframe
        else:
            return recommenderMethods.convert_df_to_json(post_dataframe)

    @DeprecationWarning
    def preprocess_feature(self, feature_text, stemming=False):
        post_excerpt_preprocessed = self.preprocess(feature_text, stemming)
        return post_excerpt_preprocessed

    @DeprecationWarning
    def cz_lemma(self, string, json=False):
        morph = majka.Majka('morphological_database/majka.w-lt')

        morph.flags |= majka.ADD_DIACRITICS  # find word forms with diacritics
        morph.flags |= majka.DISALLOW_LOWERCASE  # do not enable to find lowercase variants
        morph.flags |= majka.IGNORE_CASE  # ignore the word case whatsoever
        morph.flags = 0  # unset all flags

        morph.tags = False  # return just the lemma, do not process the tags
        morph.tags = True  # turn tag processing back on (default)

        morph.compact_tag = True  # return tag in compact form (as returned by Majka)
        morph.compact_tag = False  # do not return compact tag (default)

        morph.first_only = True  # return only the first entry
        morph.first_only = False  # return all entries (default)

        morph.tags = False
        morph.first_only = True
        morph.negative = "ne"

        ls = morph.find(string)

        if json is not True:
            if not ls:
                return string
            else:
                # # print(ls[0]['lemma'])
                return str(ls[0]['lemma'])
        else:
            return ls


class Reader(object):
    ''' Source reader object feeds other objects to iterate through a source. '''
    def __init__(self):

        ''' init '''
        # exclude_stops = set(('.', '(', ')'))
        # self.stop = set(stopwords.words('english')) - exclude_stops
        self.wn_lemmatizer = CzPreprocess()

    def prepare_words(self, text):
        ''' Prepare text
        '''
        # lower cased all text
        texts = text.lower()
        # tokenize
        texts = re.split(r'(?![\.|\$])[^\w\d]', texts)
        texts = [w.strip('.') for w in texts]
        # remove words that are too short
        texts = [w for w in texts if not len(w)<3]
        # remove words that are not alphanumeric and does not contain at least one character
        texts = [w for w in texts if w.isalnum()]
        # remove numbers only
        texts = [w for w in texts if not w.isdigit()]
        # remove stopped words
        # texts = [w for w in texts if not w in self.stop]
        # remove duplicates
        seen = set()
        seen_add = seen.add
        texts = [w for w in texts if not (w in seen or seen_add(w)) ]
        # lemmatize
        texts = [self.wn_lemmatizer.cz_lemma(w) for w in texts]
        return texts

    def iterate(self):
        ''' virtual method '''
        pass


class MongoReader(Reader):
    def __init__(self, dbName=None, collName=None,
                 mongoURI="mongodb://localhost:27017", limit=None):
        ''' init
            :param mongoURI: mongoDB URI. default: localhost:27017
            :param dbName: MongoDB database name.
            :param collName: MongoDB Collection name.
            :param limit: query limit
        '''
        Reader.__init__(self)
        self.conn = None
        self.mongoURI = mongoURI
        self.dbName = dbName
        self.collName = collName
        self.limit = limit
        self.fields = []
        self.key_field = 'text'
        self.return_fields = ['text']

    def get_value(self, value):
        ''' convinient method to retrive value.
        '''
        if not value:
            return value
        if isinstance(value, list):
            return ' '.join([v.encode('utf-8', 'replace').decode('utf-8', 'replace') for v in value])
        else:
            return value.encode('utf-8', 'replace').decode('utf-8', 'replace')

    def iterate(self):
        ''' Iterate through the source reader '''
        if not self.conn:
            try:
                self.conn = pymongo.MongoClient(self.mongoURI)[self.dbName][self.collName]
            except Exception as ex:
                raise Exception("ERROR establishing connection: %s" % ex)

        if self.limit:
            cursor = self.conn.find().limit(self.limit)
        else:
            cursor = self.conn.find({}, self.fields)

        for doc in cursor:
            content = ""
            for f in self.return_fields:
                content +=" %s" % (self.get_value(doc.get(f)))
            texts = self.prepare_words(content)
            # tags = doc.get(self.key_field).split(',')
            # tags = [t.strip() for t in tags]
            doc = { "text": texts }
            yield doc


class Word2VecClass:
    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/w2v_embedding_all_in_one"

    def __init__(self):
        self.documents = None
        self.df = None
        self.database = Database()

    @DeprecationWarning
    def get_posts_dataframe(self):
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    @DeprecationWarning
    def get_categories_dataframe(self):
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe(pd)
        self.database.disconnect()
        return self.categories_df

    @DeprecationWarning
    def join_posts_ratings_categories(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'post_title', 'slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
             'all_features_preprocessed']]

    @DeprecationWarning
    def join_posts_ratings_categories_full_text(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'post_title', 'slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
             'all_features_preprocessed', 'body_preprocessed', 'full_text']]

    def prepare_word2vec_eval(self):
        recommenderMethods = RecommenderMethods()

        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        del self.posts_df
        del self.categories_df

        documents_df = pd.DataFrame()
        documents_training_df = pd.DataFrame()

        documents_df["features_to_use"] = self.df["category_title"] + " " + self.df["keywords"] + ' ' + self.df[
            "all_features_preprocessed"]
        documents_df["slug"] = self.df["slug"]

        documents_training_df["features_to_use"] = self.df["category_title"] + " " + self.df["keywords"] + " " + self.df[
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

        # word2vec_embedding = KeyedVectors.load_texts(self.amazon_bucket_url)
        # self.amazon_bucket_url#

        print("Loading Word2Vec FastText (Wikipedia) model...")
        # word2vec_embedding = KeyedVectors.load_texts("models/w2v_model_limited")
        self.find_optimal_model_idnes(texts)

    # @profile
    def get_similar_word2vec(self, searched_slug, model="idnes"):
        recommenderMethods = RecommenderMethods()

        self.posts_df = recommenderMethods.get_posts_dataframe()
        self.categories_df = recommenderMethods.get_categories_dataframe()
        self.df = recommenderMethods.join_posts_ratings_categories()

        print("self.posts_df")
        print(self.posts_df)
        print("self.categories_df")
        print(self.categories_df)
        print("self.df")
        print(self.df)

        found_post_dataframe = recommenderMethods.find_post_by_slug(searched_slug)
        found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                  found_post_dataframe.iloc[0]['category_title'] + " " + \
                                                  found_post_dataframe.iloc[0]['all_features_preprocessed']
        del self.posts_df
        del self.categories_df

        documents_df = pd.DataFrame()

        documents_df["features_to_use"] = self.df["category_title"] + " " + self.df["keywords"] + ' ' + self.df[
            "all_features_preprocessed"]
        documents_df["slug"] = self.df["slug"]
        found_post = found_post_dataframe['features_to_use'].iloc[0]

        del self.df
        del found_post_dataframe

        documents_df['features_to_use'] = documents_df['features_to_use'] + "; " + documents_df['slug']
        list_of_document_features = documents_df["features_to_use"].tolist()
        print("list_of_document_features")
        print(list_of_document_features)
        del documents_df
        # https://github.com/v1shwa/document-similarity with my edits

        if model == "wiki":
            model_wiki = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
            print("Similarities on Wikipedia.cz model:")
            ds = DocSim(model_wiki)
            most_similar_articles_with_scores = ds.calculate_similarity_wiki_model_gensim(found_post,
                                                                                          list_of_document_features)[:21]
        elif model == "idnes":
            model_idnes = KeyedVectors.load("models/w2v_idnes.model")
            print("Similarities on iDNES.cz model:")
            ds = DocSim(model_idnes)
            print("found_post")
            print(found_post)
            print("list_of_document_features")
            print(list_of_document_features)
            most_similar_articles_with_scores = ds.calculate_similarity_idnes_model_gensim(found_post,
                                                                                           list_of_document_features)[:21]
        print("most_similar_articles_with_scores:")
        print(most_similar_articles_with_scores)
        # removing post itself
        if len(most_similar_articles_with_scores) > 0:
            del most_similar_articles_with_scores[0]  # removing post itself

            # workaround due to float32 error in while converting to JSON
            return json.loads(json.dumps(most_similar_articles_with_scores, cls=NumpyEncoder))
        else:
            return None

    # @profile
    def get_similar_word2vec_full_text(self, searched_slug):
        recommenderMethods = RecommenderMethods()

        self.posts_df = recommenderMethods.get_posts_dataframe()
        self.categories_df = recommenderMethods.get_categories_dataframe(rename_title=True)
        self.df = recommenderMethods.join_posts_ratings_categories_full_text()

        print("self.posts_df")
        print(self.df.head(10).to_string())
        print("self.categories_df")
        print(self.df.head(10).to_string())
        print("self.df")
        print(self.df.head(10).to_string())

        # search_terms = 'Domácí. Zemřel poslední krkonošský nosič Helmut Hofer, ikona Velké Úpy. Ve věku 88 let zemřel potomek slavného rodu vysokohorských nosičů Helmut Hofer z Velké Úpy. Byl posledním žijícím nosičem v Krkonoších, starodávným řemeslem se po staletí živili generace jeho předků. Jako nosič pracoval pro Českou boudu na Sněžce mezi lety 1948 až 1953.'
        found_post_dataframe = recommenderMethods.find_post_by_slug(searched_slug, force_update=True)

        print("found_post_dataframe")
        print(found_post_dataframe)

        # TODO: If this works well on production, add also to short text version
        if found_post_dataframe is None:
            return []
        else:
            print("found_post_dataframe.iloc[0]")
            print(found_post_dataframe.iloc[0])
            found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
            found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                      found_post_dataframe.iloc[0]['category_title'] + " " + \
                                                      found_post_dataframe.iloc[0]['all_features_preprocessed'] + " " + \
                                                      found_post_dataframe.iloc[0]['body_preprocessed']

            del self.posts_df
            del self.categories_df

            # cols = ["category_title", "post_title", "excerpt", "keywords", "slug", "all_features_preprocessed"]
            documents_df = pd.DataFrame()

            documents_df["features_to_use"] = self.df["keywords"] + '||' + self.df["category_title"] + ' ' + self.df[
                "all_features_preprocessed"] + ' ' + self.df["body_preprocessed"]
            documents_df["slug"] = self.df["post_slug"]

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

            most_similar_articles_with_scores = ds.calculate_similarity_idnes_model(found_post,
                                                                                    list_of_document_features)[:21]
            # removing post itself
            del most_similar_articles_with_scores[0]  # removing post itself

            # workaround due to float32 error in while converting to JSON
            return json.loads(json.dumps(most_similar_articles_with_scores, cls=NumpyEncoder))

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def save_full_model_to_smaller(self, model="wiki"):
        print("Saving full model to limited model...")
        if model == "wiki":
            word2vec_embedding = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full",
                                                                   limit=87000)  #
        elif model == "idnes":
            word2vec_embedding = KeyedVectors.load_word2vec_format("full_models/idnes/word2vec/w2v_model_full",
                                                                   limit=87000)  #
        else:
            word2vec_embedding = None
        word2vec_embedding.save("models/w2v_model_limited")  # write separately=[] for all_in_one model

    def save_fast_text_to_w2v(self):
        print("Loading and saving FastText pretrained model to Word2Vec model")
        word2vec_model = gensim.models.fasttext.load_facebook_vectors("full_models/cswiki/word2vec/cc.cs.300.bin.gz",
                                                                      encoding="utf-8")
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
            posts = database.get_not_prefilled_posts(full_text, method="tfidf")

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
                    if actual_recommended_json is None:
                        print("No recommended post found. Skipping.")
                        continue
                    actual_recommended_json = json.dumps(actual_recommended_json)
                    if full_text is False:
                        try:
                            database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                             article_id=post_id, full_text=False, db="pgsql",
                                                             method="word2vec")
                        except:
                            print("Error in DB insert. Skipping.")
                            pass
                    else:
                        try:
                            database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                             article_id=post_id, full_text=True, db="pgsql",
                                                             method="word2vec")
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
                else:
                    print("Skipping.")
            else:
                if full_text is False:
                    actual_recommended_json = self.get_similar_word2vec(slug)
                else:
                    actual_recommended_json = self.get_similar_word2vec_full_text(slug)
                actual_recommended_json = json.dumps(actual_recommended_json)
                if full_text is False:
                    database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                     article_id=post_id, full_text=False, db="pgsql",
                                                     method="word2vec")
                else:
                    database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                     article_id=post_id, full_text=True, db="pgsql",
                                                     method="word2vec")
                number_of_inserted_rows += 1

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

    # TODO: texts by generator
    # TODO: create bigrams
    def find_optimal_model_idnes(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Enabling Word2Vec logging
        logging.basicConfig(filename='model_callbacks.log',
                            format="%(asctime)s:%(levelname)s:%(message)s",
                            level=logging.NOTSET)
        logger = logging.getLogger()  # get the root logger
        logger.info("Testing file write")

        reader = MongoReader(dbName='idnes', collName='preprocessed_articles_stopwords_free')
        print("Building sentences...")
        sentences = [doc.get('text') for doc in reader.iterate()]


        model_variants = [0, 1]  # sg parameter: 0 = CBOW; 1 = Skip-Gram
        hs_softmax_variants = [0, 1]  # 1 = Hierarchical SoftMax
        negative_sampling_variants = range(5, 20, 5)  # 0 = no negative sampling
        no_negative_sampling = 0  # use with hs_soft_max
        vector_size_range = [50, 100, 158, 200, 250, 300, 450]
        window_range = [1, 2, 4, 5, 8, 12, 16, 20]
        min_count_range = [0, 2, 4, 5, 8, 12, 16, 20]
        epochs_range = [2, 5, 10, 15, 20, 25, 30]
        sample_range = [0.0, 1.0 * (10.0 ** -1.0), 1.0 * (10.0 ** -2.0), 1.0 * (10.0 ** -3.0), 1.0 * (10.0 ** -4.0),
                        1.0 * (10.0 ** -5.0)]  # useful range is (0, 1e-5) acording to : https://radimrehurek.com/gensim/models/word2vec.html

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
                                            word_pairs_eval, analogies_eval = self.compute_eval_values_idnes(sentences=sentences,
                                                                                                             model_variant=model_variant,
                                                                                                             negative_sampling_variant=no_negative_sampling,
                                                                                                             vector_size=vector_size,
                                                                                                             window=window,
                                                                                                             min_count=min_count,
                                                                                                             epochs=epochs,
                                                                                                             sample=sample)
                                        else:
                                            word_pairs_eval, analogies_eval = self.compute_eval_values_idnes(sentences=sentences,
                                                                                                             model_variant=model_variant,
                                                                                                             negative_sampling_variant=negative_sampling_variant,
                                                                                                             vector_size=vector_size,
                                                                                                             window=window,
                                                                                                             min_count=min_count,
                                                                                                             epochs=epochs,
                                                                                                             sample=sample)

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

    def compute_eval_values_idnes(self, sentences, model_variant, negative_sampling_variant, vector_size, window, min_count,
                                  epochs, sample, force_update_model=True):
        if os.path.isfile("models/w2v_idnes.model") is False or force_update_model is True:
            print("Started training on iDNES.cz dataset...")
            # w2v_model = Word2Vec(sentences=sentences, sg=model_variant, negative=negative_sampling_variant, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, sample=sample, workers=7)
            w2v_model = Word2Vec(sentences=sentences, sg=model_variant, negative=negative_sampling_variant,
                                 vector_size=vector_size, window=window, min_count=min_count, epochs=epochs,
                                 sample=sample, workers=7)
            # DEFAULT:
            # model = Word2Vec(sentences=texts, vector_size=158, window=5, min_count=5, workers=7, epochs=15)
            w2v_model.save("models/w2v_idnes.model")
        else:
            print("Loading Word2Vec iDNES.cz model from saved model file")
            w2v_model = Word2Vec.load("models/w2v_idnes.model")

        import pandas as pd
        df = pd.read_csv('research/word2vec/similarities/WordSim353-cs.csv',
                         usecols=['cs_word_1', 'cs_word_2', 'cs mean'])
        df.to_csv('research/word2vec/similarities/WordSim353-cs-cropped.tsv', sep='\t', encoding='utf-8', index=False)

        print("Word pairs evaluation iDnes.cz model:")
        word_pairs_eval = w2v_model.wv.evaluate_word_pairs('research/word2vec/similarities/WordSim353-cs-cropped.tsv')
        print(word_pairs_eval)

        overall_score, _ = w2v_model.wv.evaluate_word_analogies('research/word2vec/analogies/questions-words-cs.txt')
        print("Analogies evaluation of iDnes.cz model:")
        print(overall_score)

        return word_pairs_eval, overall_score

    def eval_idnes_basic(self):
        topn = 30
        limit = None

        print("Loading iDNES.cz full model...")
        idnes_model = KeyedVectors.load_word2vec_format("models/w2v_model_full.model", limit=limit)

        searched_word = 'hokej'
        sims = idnes_model.most_similar(searched_word, topn=topn)  # get other similar words
        print("TOP " + str(topn) + " similar Words from iDNES.cz for word " + searched_word + ":")
        print(sims)

        searched_word = 'zimák'
        if searched_word in idnes_model.key_to_index:
            print(searched_word)
            print("Exists in vocab")
        else:
            print(searched_word)
            print("Doesn't exists in vocab")

        helper = Helper()
        self.save_tuple_to_csv("idnes_top_" + str(topn) + "_similar_words_to_hokej_limit_" + str(limit) + ".csv", sims)

        print("Word pairs evaluation FastText on iDNES.cz model:")
        print(idnes_model.evaluate_word_pairs('research/word2vec/similarities/WordSim353-cs-cropped.tsv'))

        overall_analogies_score, _ = idnes_model.evaluate_word_analogies(
            "research/word2vec/analogies/questions-words-cs.txt")
        print("Analogies evaluation of FastText on iDNES.cz model:")
        print(overall_analogies_score)

    def train_word2vec_idnes_model(self, data):

        data_words_nostops = data_queries.remove_stopwords(data['tokenized'])
        data_words_bigrams = self.build_bigrams_and_trigrams(data_words_nostops)

        self.df.assign(tokenized=data_words_bigrams)

        # View
        all_words = [word for item in self.df['tokenized'] for word in item]
        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)
        k = 15000
        top_k_words = zip(*fdist.most_common(k))
        self.top_k_words = set(top_k_words)

        self.df['tokenized'] = self.df['tokenized'].apply(self.keep_top_k_words)

        # document length
        self.df['doc_len'] = self.df['tokenized'].apply(lambda x: len(x))
        self.df.drop(labels='doc_len', axis=1, inplace=True)

        minimum_amount_of_words = 5

        self.df = self.df[self.df['tokenized'].map(len) >= minimum_amount_of_words]
        # make sure all tokenized items are lists
        self.df = self.df[self.df['tokenized'].map(type) == list]
        self.df.reset_index(drop=True, inplace=True)

        # View
        dictionary = corpora.Dictionary(data_words_bigrams)
        dictionary.filter_extremes()
        # dictionary.compactify()
        corpus = [dictionary.doc2bow(doc) for doc in data_words_bigrams]
        self.save_corpus_dict(corpus, dictionary)

    def eval_wiki(self):
        topn = 30
        limit = None

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

        overall_analogies_score, _ = wiki_full_model.evaluate_word_analogies(
            "research/word2vec/analogies/questions-words-cs.txt")
        print("Analogies evaluation of FastText on Wikipedia.cz model:")
        print(overall_analogies_score)

    def build_bigrams_and_trigrams(self, force_update=False):
        cursor_any_record = mongo_collection_bigrams.find_one()
        if cursor_any_record is not None and force_update is False:
            print("There are already records in MongoDB. Skipping bigrams building.")
            pass
        else:
            print("Building bigrams...")
            mongo_collection_bigrams.delete_many({})
            print("Loading stopwords free documents...")
            # using 80% training set
            """
            print("Building sentences...")
            sentences = [doc.get('text') for doc in reader.iterate()]
            first_sentence = next(iter(sentences))
            print("first_sentence[:10]")
            print(first_sentence[:10])
            """

            reader = MongoReader(dbName='idnes', collName='preprocessed_articles_stopwords_free')
            print("Building sentences...")
            sentences = [doc.get('text') for doc in reader.iterate()]
            # Working but possibly slow
            phrase_model = gensim.models.Phrases(sentences, min_count=1, threshold=1)  # higher threshold fewer phrases.

            cursor = mongo_collection_stopwords_free.find({})
            i = 1
            for doc in cursor:
                print("Building bigrams for document number " + str(i))
                bigram_text = phrase_model[doc['text']]
                print("bigram_text:")
                print(bigram_text)
                save_to_mongo(number_of_processed_files=i, data=bigram_text,
                              mongo_collection=mongo_collection_bigrams)
                i = i + 1

    def keep_top_k_words(self, text):
        return [word for word in text if word in self.top_k_words]

    def save_tuple_to_csv(self, path, data):
        with open(path, 'w+') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['word', 'sim'])
            for row in data:
                csv_out.writerow(row)

    def save_corpus_dict(self, corpus, dictionary):
        print("Saving corpus and dictionary...")
        pickle.dump(corpus, open('precalc_vectors/corpus.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        dictionary.save('precalc_vectors/dictionary.gensim')

    def remove_stopwords_mongodb(self, force_update=False):
        cursor_any_record = mongo_collection_stopwords_free.find_one()
        if cursor_any_record is not None and force_update is False:
            print("There are already records in MongoDB. Skipping stopwords removal.")
            pass
        else:
            print("Removing stopwords...")
            cursor = mongo_collection.find({})
            number = 1
            # clearing collection from all documents
            mongo_collection_stopwords_free.delete_many({})
            for doc in cursor:
                print("Before:")
                print(doc['text'])
                removed_stopwords = data_queries.remove_stopwords(doc['text'])
                removed_stopwords = self.flatten(removed_stopwords)
                print("After removal:")
                print(removed_stopwords)
                save_to_mongo(number_of_processed_files=number, data=removed_stopwords, mongo_collection=mongo_collection_stopwords_free)
                number = number + 1

    def preprocess_idnes_corpus(self, force_update=False):

        print("Corpus lines are above")
        cursor_any_record = mongo_collection.find_one()
        if cursor_any_record is not None and force_update is False:
            print("There are already records in MongoDB. Skipping Idnes preprocessing (1st phase)")
            pass
        else:
            path_to_corpus = 'full_models/idnes/unprocessed/idnes.mm'
            path_to_pickle = 'full_models/idnes/unprocessed/idnes.pkl'
            corpus = pickle.load(open(path_to_pickle, 'rb'))
            print("Corpus length:")
            print(len(corpus))
            time.sleep(120)
            # preprocessing steps
            czpreprocessing = CzPreprocess()
            helper = Helper()
            processed_data = []

            last_record = mongo_db.mongo_collection.find()
            print("last_record")
            print(last_record)
            number_of_documents = 0
            cursor = mongo_collection.find({})
            print("Fetching records for DB...")
            cursor_any_record = mongo_collection.find_one()
            # Checking the cursor is empty or not
            number_of_documents = 0
            if cursor_any_record is None:
                number_of_documents = 0
            else:
                """
                for record in cursor:
                    number_of_documents = record['number']
                """
                number_of_documents = mongo_collection.estimated_document_count()
                print("Number_of_docs already in DB:")
                print(number_of_documents)

            if number_of_documents == 0:
                print("No file with preprocessed articles was found. Starting from 0.")
            else:
                print("Starting another preprocessing from document where it was halted.")
                print("Starting from doc. num: " + str(number_of_documents))

            i = 0
            num_of_preprocessed_docs = number_of_documents
            # clearing collection from all documents
            mongo_collection.delete_many({})
            for doc in helper.generate_lines_from_mmcorpus(corpus):
                if number_of_documents > 0:
                    number_of_documents -= 1
                    print("Skipping doc.")
                    print(doc[:10])
                    continue
                print("Processing doc. num. " + str(num_of_preprocessed_docs))
                print("Before:")
                print(doc)
                doc_string = ' '.join(doc)
                doc_string_preprocessed = deaccent(czpreprocessing.preprocess(doc_string))
                # tokens = doc_string_preprocessed.split(' ')

                # removing words in greek, azbuka or arabian
                # use only one of the following lines, whichever you prefer
                tokens = [i for i in doc_string_preprocessed.split(' ') if regex.sub(r'[^\p{Latin}]', u'', i)]
                # processed_data.append(tokens)
                print("After:")
                print(tokens)
                i = i + 1
                num_of_preprocessed_docs = num_of_preprocessed_docs + 1
                # saving list to pickle evey Nth document

                print("Preprocessing Idnes.cz doc. num. " + str(num_of_preprocessed_docs))
                save_to_mongo(tokens, num_of_preprocessed_docs, mongo_collection)

            print("Preprocessing Idnes has (finally) ended. All articles were preprocessed.")

    def create_dictionary_from_dataframe(self, force_update=False, filter_extremes=False):
        path_to_dict = "full_models/idnes/unprocessed/idnes.dict"
        path_to_corpus = "full_models/idnes/unprocessed/idnes.mm"
        if os.path.exists(path_to_dict) is False or os.path.exists(path_to_corpus) is False or force_update is True:
            recommenderMethods = RecommenderMethods()
            cz_preprocess = CzPreprocess()
            post_df = recommenderMethods.join_posts_ratings_categories()
            post_df['full_text'] = post_df['full_text'].replace([None], '')

            post_df['all_features_preprocessed'] = post_df['all_features_preprocessed'] + ' ' + post_df['full_text']

            gc.collect()

            # Preprocessing to small to calling
            # post_df['all_features_preprocessed'] = post_df['all_features_preprocessed'].map(cz_preprocess.preprocess)
            post_df['all_features_preprocessed'] = post_df.all_features_preprocessed.apply(lambda x: x.split(' '))
            post_df['all_features_preprocessed'] = post_df[['all_features_preprocessed']]
            print("post_df")
            print(post_df['all_features_preprocessed'][:100])

            all_features_preprocessed_list = post_df['all_features_preprocessed'].to_numpy()

            path_to_pickle = 'full_models/idnes/unprocessed/idnes.pkl'
            pickle.dump(all_features_preprocessed_list, open(path_to_pickle, 'wb'))
            print("all_features_preprocessed_list:")
            print(all_features_preprocessed_list[:100])
            time.sleep(60)
            tokenzized_texts = [document for document in all_features_preprocessed_list]

            # remove words that appear only once
            frequency = defaultdict(int)
            for text in tokenzized_texts:
                for token in text:
                    frequency[token] += 1

            tokenzized_texts = [
                [token for token in text if frequency[token] > 1]
                for text in tokenzized_texts
            ]
            # all_features_preprocessed_list_of_lists = [[i] for i in all_features_preprocessed_list]
            print("list of lists:")
            print(tokenzized_texts[:1000])
            time.sleep(50)
            dictionary = corpora.Dictionary(line for line in tokenzized_texts)
            path_to_dict = 'full_models/idnes/unprocessed/idnes.dict'
            path_to_dict_folder = 'full_models/idnes/unprocessed/'
            if not os.path.isfile(path_to_dict):
                os.makedirs(path_to_dict_folder)
            dictionary.save(path_to_dict)
            corpus = [dictionary.doc2bow(text, allow_update=True) for text in tokenzized_texts]
            print(corpus[:100])
            word_counts = [[(dictionary[id], count) for id, count in line] for line in corpus]
            print(word_counts[:100])
            print("Serializing...")
            corpora.MmCorpus.serialize(path_to_corpus, corpus)  # store to disk, for later use
            print("Dictionary and  Corpus successfully saved on disk")

        else:
            print("Dictionary and corpus already exists")


    def test_with_and_without_extremes(self):
        # TODO: Test this
        return False

    def create_dictionary_idnes(self, sentences=None, force_update=False, filter_extremes=False):
        # a memory-friendly iterator
        if os.path.isfile("full_models/idnes/lda/preprocessed/dictionary") is False or force_update is True:
            if sentences is None:
                reader = MongoReader(dbName='idnes', collName='preprocessed_articles_bigrams')
                sentences = [doc.get('text') for doc in reader.iterate()]
            print("Creating dictionary...")
            preprocessed_dictionary = corpora.Dictionary(line for line in sentences)
            del sentences
            gc.collect()
            if filter_extremes is True:
                preprocessed_dictionary.filter_extremes()
            print("Saving dictionary...")
            preprocessed_dictionary.save("full_models/idnes/lda/preprocessed/dictionary")
            return preprocessed_dictionary
        else:
            print("Dictionary already exists. Loading...")
            loaded_dict = corpora.Dictionary.load("full_models/idnes/lda/preprocessed/dictionary")
            return loaded_dict

    def create_corpus_from_mongo_idnes(self, dictionary, sentences=None, force_update=False, preprocessed=False):
        path_part_1 = "full_models/idnes/"
        path_part_2 = "/idnes.mm"
        if preprocessed is True:
            path_to_corpus = path_part_1 + "preprocessed" + path_part_2
        else:
            path_to_corpus = path_part_1 + "unpreprocessed" + path_part_2
        if os.path.isfile(path_to_corpus) is False or force_update is True:
            corpus = MyCorpus(dictionary)
            del sentences
            gc.collect()
            print("Saving preprocessed corpus...")
            corpora.MmCorpus.serialize(path_to_corpus, corpus)
        else:
            print("Corpus already exists. Loading...")
            corpus = corpora.MmCorpus(path_to_corpus)
        return corpus

@DeprecationWarning
def run():
    searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"  # print(doc2vecClass.get_similar_doc2vec(slug))
    # searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    # searched_slug = "facr-o-slavii-a-rangers-verime-v-objektivni-vysetreni-odmitame-rasismus"

    """
    start = t.time()
    print(word2vecClass.get_similar_word2vec(searched_slug))
    end = t.time()
    print("Elapsed time: " + str(end - start))
    start = t.time()
    print(word2vecClass.get_similar_word2vec_full_text(searched_slug))
    end = t.time()
    print("Elapsed time: " + str(end - start))
    """

    # word2vecClass.find_optimal_model_idnes()

    # 1. Create and save Dictionary
    # 2. Create and save Corpus from Dictionary
    # 3. Preprocessing
    # 4. Preprocessing + stopwords free
    # 5. Preprocessing + bigrams