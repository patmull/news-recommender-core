import gc
import json
import os
import re
import string
import time
import nltk
from collections import defaultdict
from pathlib import Path
import pickle
import dropbox
import gensim
import majka
import numpy as np
import pandas as pd
from gensim import corpora
from gensim import models
from gensim import similarities

# import smart_open
# import boto3
# import psutil
"""
from memory_profiler import profile
import cProfile
import io
import pstats
import psutil
"""
# from guppy import hpy

from gensim.models import TfidfModel, KeyedVectors, LdaModel, fasttext, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import RegexpTokenizer, FreqDist, word_tokenize
from scipy import sparse
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from doc_sim import DocSim
from data_conenction import Database
from text_classification.czech_stemmer import cz_stem

# AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
# AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
# AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

"""
def profile(fnc):
    # A decorator that uses cProfile to profile a function

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        # print(s.getvalue())
        return retval

    return inner
"""

word2vec_embedding = None
doc2vec_model = None
lda_model = None


def load_models():
    print("Loading Word2Vec model")
    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/w2v_embedding_all_in_one"

    global word2vec_embedding
    """
    s3 = boto3.client('s3')
    destination_file = "w2v_embedding_all_in_one"
    bucket_name = "moje-clanky"
    s3.download_file(bucket_name, destination_file, destination_file)
    """
    word2vec_embedding = KeyedVectors.load("models/w2v_model")

    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/d2v_all_in_one.model"
    print("Loading Doc2Vec model")
    global doc2vec_model
    # doc2vec_model = pickle.load(smart_open.smart_open(amazon_bucket_url))
    # doc2vec_model = Doc2Vec.load("d2v_all_in_one.model")
    doc2vec_model = Doc2Vec.load("models/d2v.model")

    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/lda_all_in_one"
    print("Loading LDA model")

    global lda_model
    # lda_model = pickle.load(smart_open.smart_open(amazon_bucket_url))
    # lda_model = Lda.load("lda_all_in_one")
    lda_model = LdaModel.load("models/lda_model")


global cz_stopwords


def load_stopwords():
    filename = "text_classification/czech_stopwords.txt"
    with open(filename, encoding="utf-8") as file:
        global cz_stopwords
        cz_stopwords = file.readlines()
        cz_stopwords = [line.rstrip() for line in cz_stopwords]


class Helper:
    # helper functions

    def get_id_from_title(self, title, df):
        return self.df[df.title == title]["row_num"].values[0]

    def get_id_from_slug(self, slug, df):
        return self.df[df.slug == slug]["row_num"].values[0]

    def get_title_from_id(self, id, df):
        data_frame_row = df.loc[df['row_num'] == id]
        return data_frame_row["title"].values[0]

    def get_slug_from_id(self, id, df):
        data_frame_row = df.loc[df['row_num'] == id]
        return data_frame_row["slug"].values[0]


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class CosineTransform:

    def __init__(self):
        self.cosine_sim = None
        self.count_matrix = None
        self.similar_articles = None
        self.sorted_similar_articles = None
        self.database = Database()
        self.helper = Helper()
        self.posts_df = None

    def simple_example(self):
        text = ["London Paris London", "Paris Paris London"]
        cv = CountVectorizer()

        count_matrix = cv.fit_transform(text)

        # print(count_matrix.toarray())

        similarity_scores = cosine_similarity(count_matrix)

        # print(similarity_scores)

    def combine_features(self, row):
        return row['title'] + " " + row['keywords'] + " " + row['excerpt']

    def combine_current_posts(self):

        ##Step 1
        self.database.set_row_var()
        # EXTRACT RESULTS FROM CURSOR
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        ##Step 2: Select Features
        features = ['title', 'excerpt', 'keywords']
        ##Step 3: Create a column in DF which combines all selected features
        for feature in features:
            self.posts_df[feature] = self.posts_df[feature].fillna('')

        helper = Helper()
        self.posts_df["combined_features"] = self.posts_df.apply(self.combine_features, axis=1)

        ##Step 4: Create count matrix from this new combined column
        cv = CountVectorizer()

        self.count_matrix = cv.fit_transform(self.posts_df["combined_features"])

    def cosine_similarity(self):
        ##Step 5: Compute the Cosine Similarity based on the count_matrix
        try:
            # self.cosine_sim = self.cosine_similarity_n_space(self.count_matrix)
            self.cosine_sim = cosine_similarity(self.count_matrix)
        except Exception as ex:
            # print(ex)
            pass
        ## print(self.cosine_sim)

    ## Step 6: Get id of this article from its title

    def article_user_likes(self, slug):
        helper = Helper()
        article_id = helper.get_id_from_slug(slug, self.posts_df)
        # print("article_user_likes: " + slug)
        # print("article_id: ")
        ## print(article_id)
        try:
            self.similar_articles = list(enumerate(self.cosine_sim[article_id]))
        except TypeError as te:
            # print(te)
            pass
        ## print(self.similar_articles)

    ## Step 7: Get a list of similar articles in descending order of similarity score
    def get_list_of_similar_articles(self):
        try:
            self.sorted_similar_articles = sorted(self.similar_articles, key=lambda x: x[1], reverse=True)
        except TypeError as te:
            # print(te)
            pass
        ## print(self.sorted_similar_articles)
        return self.sorted_similar_articles

    ## Step 8: Print titles of first 10 articles
    def get_similar(self):
        i = 0
        list_of_article_slugs = []
        list_returned_dict = {}

        for element in self.sorted_similar_articles:

            list_returned_dict['slug'] = self.helper.get_slug_from_id(element[0], self.posts_df)
            list_returned_dict['coefficient'] = element[1]
            list_of_article_slugs.append(list_returned_dict.copy())
            i = i + 1
            if i > 5:
                break

        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs)
        return json.dumps(list_of_article_slugs)

    def cosine_similarity_n_space(self, m1, m2=None, batch_size=100):
        assert m1.shape[1] == m2.shape[1] and isinstance(batch_size, int) == True

        ret = np.ndarray((m1.shape[0], m2.shape[0]))

        batches = m1.shape[0] // batch_size

        if m1.shape[0] % batch_size != 0:
            batches = batches + 1

        for row_i in range(0, batches):
            start = row_i * batch_size
            end = min([(row_i + 1) * batch_size, m1.shape[0]])
            rows = m1[start: end]
            sim = cosine_similarity(rows, m2)
            ret[start: end] = sim

        return ret

    def fill_recommended_for_all_posts(self, skip_already_filled):
        database = Database()
        database.connect()
        all_posts = database.get_all_posts()

        for post in all_posts:

            post_id = post[0]
            slug = post[3]
            current_recommended = post[15]

            if skip_already_filled is True:
                if current_recommended is None:
                    actual_recommended_json = self.get_by_param(slug)

                    database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                     article_id=post_id)
                else:
                    print("Skipping.")
            else:
                actual_recommended_json = self.get_by_param(slug)

                database.insert_recommended_json(articles_recommended_json=actual_recommended_json, article_id=post_id)


class GenSim:

    def __init__(self):
        self.posts_df = None
        self.categories_df = None
        self.df = None
        self.database = Database()
        self.documents = None

    def get_posts_dataframe(self):
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    def join_posts_ratings_categories(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description']]

    def find_post_by_slug(self, slug):
        post_dataframe = self.df.loc[self.df['slug_x'] == slug]
        doc = post_dataframe["title_y"] + " " + post_dataframe["keywords"] + " " + post_dataframe["title_x"] + " " + \
              post_dataframe["excerpt"]
        return str(doc.tolist())
        # return self.get_posts_dataframe().loc[self.get_posts_dataframe()['slug'] == slug]
        # return str(self.df['title_y','title_x','excerpt','keywords','slug'].loc[self.df['slug'] == slug].values[0])

    def get_categories_dataframe(self):
        self.categories_df = self.database.get_categories_dataframe(pd)
        return self.categories_df

    def recommend_posts_by_all_features(self, slug):

        gensimClass = GenSim()

        gensimClass.get_posts_dataframe()  # load posts do dataframe
        gensimClass.get_categories_dataframe()  # load categories to dataframe
        # tfidf.get_ratings_dataframe() # load post rating do dataframe

        gensimClass.join_posts_ratings_categories()  # joining posts and categories into one table
        fit_by_title_matrix = gensimClass.get_fit_by_feature('title_x', 'title_y')  # prepended by category
        fit_by_excerpt_matrix = gensimClass.get_fit_by_feature('excerpt')
        fit_by_keywords_matrix = gensimClass.get_fit_by_feature('keywords')

        tuple_of_fitted_matrices = (fit_by_title_matrix, fit_by_excerpt_matrix, fit_by_keywords_matrix)
        post_recommendations = gensimClass.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del gensimClass
        return post_recommendations

    def get_fit_by_feature(self, feature_name, second_feature=None):
        fit_by_feature = self.get_tfIdfVectorizer(feature_name, second_feature)
        return fit_by_feature

    def recommend_by_more_features(self, slug, tupple_of_fitted_matrices):
        # combining results of all feature types
        # combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices) # creating sparse matrix containing mostly zeroes from combined feature tupples
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)
        """
        Example 1: solving linear system A*x=b where A is 5000x5000 but is block diagonal matrix constructed of 500 5x5 blocks. Setup code:

        As = sparse(rand(5, 5));
        for(i=1:999)
           As = blkdiag(As, sparse(rand(5,5)));
        end;                         %As is made up of 500 5x5 blocks along diagonal
        Af = full(As); b = rand(5000, 1);

        Then you can test speed difference:

        As \ b % operation on sparse As takes .0012 seconds
        Af \ b % solving with full Af takes about 2.3 seconds

        """
        # # print(combined_matrix1.shape)
        # computing cosine similarity
        self.set_cosine_sim_use_own_matrix(combined_matrix1)
        # # print("self.cosine_sim_df")
        # # print(self.cosine_sim_df)

        # getting posts with highest similarity
        combined_all = self.get_recommended_posts(slug, self.cosine_sim_df,
                                                  self.df[['slug_x']])
        # print("combined_all:")
        # print(combined_all)
        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})
        # print("df_renamed:")
        # print(df_renamed)

        # json conversion
        json = self.convert_posts_to_json(df_renamed, slug)

        return json

    def load(self):

        # preprocessing
        # self.df["title_x"] = self.df["title_x"].map(lambda s: self.preprocess(s, stemming=False, lemma=False))
        # self.df["excerpt"] = self.df["excerpt"].map(lambda s: self.preprocess(s, stemming=False, lemma=False))

        # converting pandas columns to list of lists and through map to list of string joined by space ' '
        self.documents = list(map(' '.join, self.df[["keywords", "title_y", "title_x", "excerpt"]].values.tolist()))

        print("self.documents")
        print(self.documents)
        filename = "text_classification/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)
        texts = [
            [word for word in document.lower().split() if word not in cz_stopwords and len(word) > 1]
            for document in self.documents
        ]

        print("texts")
        # print(texts)

        # remove words that appear only once
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [
            [token for token in text if frequency[token] > 1]
            for text in texts
        ]

        print("texts")
        print(texts)

        return texts

    def get_recommended_by_slug(self, slug):

        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        texts = self.load()

        # does it support czech language?
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

        doc = self.find_post_by_slug(slug)
        # post_dataframe["title_x"] = post_dataframe["title_x"].map(lambda s: self.preprocess(s))
        # post_dataframe["excerpt"] = post_dataframe["excerpt"].map(lambda s: self.preprocess(s))

        doc = ''.join(doc)
        doc = str(doc)
        print("doc")
        print(type(doc))
        print(doc)

        vec_bow = dictionary.doc2bow(doc.lower().split())

        # WIP (Work in Progress)
        # vec_bow = dictionary.doc2bow(self.preprocess_single_post(slug))
        vec_lsi = lsi[vec_bow]  # convert the query to LSI space

        sims = self.get_similarities(lsi, corpus, vec_lsi)

        for doc_position, doc_score in sims:
            print(doc_score, self.documents[doc_position])

    def get_similarities(self, lsi, corpus, vec_lsi, N=10):
        index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
        Path("/tmp").mkdir(parents=True, exist_ok=True)
        index.save('/tmp/deerwester.index')
        index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
        sims = index[vec_lsi]  # perform a similarity query against the corpus
        print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims[:N]

    # pre-worked
    def preprocess(self, sentence, stemming=False, lemma=True):
        # print(sentence)
        sentence = str(sentence)
        sentence = sentence.lower()
        sentence = sentence.replace('{html}', "")
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        cleantext.translate(str.maketrans('', '', string.punctuation))  # removing punctuation
        rem_url = re.sub(r'http\S+', '', cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        # print("rem_num")
        # print(rem_num)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)
        # print("tokens")
        # print(tokens)
        if stemming is True:
            edited_words = [cz_stem(w, True) for w in tokens if len(w) > 1]  # aggresive
            edited_words = list(filter(None, edited_words))  # empty strings removal
            return " ".join(edited_words)

        elif lemma is True:
            edited_words = [self.cz_lemma(w) for w in tokens if len(w) > 1]
            edited_words_list = list(filter(None, edited_words))  # empty strings removal
            return " ".join(edited_words_list)
        else:
            return tokens
        # print(lemma_words)

    def preprocess_single_post(self, slug, json=False, stemming=False):
        post_dataframe = self.find_post_by_slug(slug)
        post_dataframe["title"] = post_dataframe["title"].map(lambda s: self.preprocess(s, stemming))
        post_dataframe["excerpt"] = post_dataframe["excerpt"].map(lambda s: self.preprocess(s, stemming))
        if json is False:
            return post_dataframe
        else:
            return self.convert_df_to_json(post_dataframe)

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


class RecommenderMethods:

    def __init__(self):
        self.database = Database()

    def get_posts_dataframe(self):
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    def get_categories_dataframe(self):
        self.categories_df = self.database.get_categories_dataframe(pd)
        return self.categories_df

    def join_posts_ratings_categories(self):
        self.get_categories_dataframe()
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
             'all_features_preprocessed', 'full_text']]
        return self.df

    def get_fit_by_feature(self, feature_name, second_feature=None):
        fit_by_feature = self.get_tfIdfVectorizer(feature_name, second_feature)

        return fit_by_feature

    def recommend_by_keywords(self, keywords, tupple_of_fitted_matrices):
        # combining results of all feature types to sparse matrix
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices, dtype=np.float16)

        # # print(combined_matrix1.shape)
        # computing cosine similarity using matrix with combined features
        self.set_cosine_sim_use_own_matrix(combined_matrix1)
        # # print("self.cosine_sim_df")
        # # print(self.cosine_sim_df)
        combined_all = self.get_recommended_posts_for_keywords(keywords, self.cosine_sim_df,
                                                               self.df[['keywords']])
        # print("combined_all:")
        # print(combined_all)
        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})
        # print("DF RENAMED:")
        # print(df_renamed)

        json = self.convert_to_json_keyword_based(df_renamed)

        return json

    def find_post_by_slug(self, slug):
        recommenderMethods = RecommenderMethods()
        return recommenderMethods.get_posts_dataframe().loc[recommenderMethods.get_posts_dataframe()['slug'] == slug]

    def get_tfIdfVectorizer(self, fit_by, fit_by_2=None, stemming=False):

        self.set_tfIdfVectorizer()
        # print("self.df[fit_by]")
        # print(self.df[fit_by])

        # self.preprocess_dataframe()

        # self.df[fit_by] = self.df[fit_by].map(lambda s:self.preprocess(s))

        # # print("PREPROCESSING: self.df[fit_by]")
        # pd.set_option('display.max_columns', None)
        # # print(self.df[fit_by].to_string())

        if fit_by_2 is None:
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[
                                                                        fit_by])  # Metoda fit: výpočet průměru a rozptylu jednotlivých sloupců z dat. Metoda transformace: # transformuje všechny prvky pomocí příslušného průměru a rozptylu.
        else:
            self.df[fit_by] = self.df[fit_by_2] + ". " + self.df[fit_by]
            # # print(self.df[fit_by])
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[
                                                                        fit_by])  # Metoda fit: výpočet průměru a rozptylu jednotlivých sloupců z dat. Metoda transformace: # transformuje všechny prvky pomocí příslušného průměru a rozptylu.
        # print("Fitted by: " + str(fit_by) + " " + str(fit_by_2))
        # print(self.tfidf_tuples)
        # Outputing results:
        print("self.tfidf_tuples")
        print(self.tfidf_tuples)
        return self.tfidf_tuples  # tuples of (document_id, token_id) and tf-idf score for it

    def set_tfIdfVectorizer(self):
        # load czech stopwords from file
        filename = "text_classification/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)

        tfidf_vectorizer = TfidfVectorizer(dtype=np.float32,
                                           stop_words=cz_stopwords)  # transforms text to feature vectors that can be used as input to estimator
        print("tfidf_vectorizer")
        print(tfidf_vectorizer)
        self.tfidf_vectorizer = tfidf_vectorizer

    # # @profile
    def recommend_by_more_features(self, slug, tupple_of_fitted_matrices):
        # combining results of all feature types
        # combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices) # creating sparse matrix containing mostly zeroes from combined feature tupples
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)
        """
        Example 1: solving linear system A*x=b where A is 5000x5000 but is block diagonal matrix constructed of 500 5x5 blocks. Setup code:

        As = sparse(rand(5, 5));
        for(i=1:999)
           As = blkdiag(As, sparse(rand(5,5)));
        end;                         %As is made up of 500 5x5 blocks along diagonal
        Af = full(As); b = rand(5000, 1);

        Then you can test speed difference:

        As \ b % operation on sparse As takes .0012 seconds
        Af \ b % solving with full Af takes about 2.3 seconds

        """
        # # print(combined_matrix1.shape)
        # computing cosine similarity
        self.set_cosine_sim_use_own_matrix(combined_matrix1)
        # # print("self.cosine_sim_df")
        # # print(self.cosine_sim_df)

        # getting posts with highest similarity
        combined_all = self.get_recommended_posts(slug, self.cosine_sim_df,
                                                  self.df[['slug_x']])
        # print("combined_all:")
        # print(combined_all)
        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})
        # print("df_renamed:")
        # print(df_renamed)

        # json conversion
        json = self.convert_datframe_posts_to_json(df_renamed, slug)

        return json

    def recommend_by_more_features_with_full_text(self, slug, tupple_of_fitted_matrices):
        # combining results of all feature types
        # combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices) # creating sparse matrix containing mostly zeroes from combined feature tupples
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)
        """
        Example 1: solving linear system A*x=b where A is 5000x5000 but is block diagonal matrix constructed of 500 5x5 blocks. Setup code:

        As = sparse(rand(5, 5));
        for(i=1:999)
           As = blkdiag(As, sparse(rand(5,5)));
        end;                         %As is made up of 500 5x5 blocks along diagonal
        Af = full(As); b = rand(5000, 1);

        Then you can test speed difference:

        As \ b % operation on sparse As takes .0012 seconds
        Af \ b % solving with full Af takes about 2.3 seconds

        """
        # # print(combined_matrix1.shape)
        # computing cosine similarity
        self.set_cosine_sim_use_own_matrix(combined_matrix1)
        # # print("self.cosine_sim_df")
        # # print(self.cosine_sim_df)

        # getting posts with highest similarity
        combined_all = self.get_recommended_posts(slug, self.cosine_sim_df,
                                                  self.df[['slug_x']])
        # print("combined_all:")
        # print(combined_all)
        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})
        # print("df_renamed:")
        # print(df_renamed)

        # json conversion
        json = self.convert_datframe_posts_to_json(df_renamed, slug)

        return json

    def set_cosine_sim_use_own_matrix(self, own_tfidf_matrix):
        own_tfidf_matrix_csr = sparse.csr_matrix(own_tfidf_matrix.astype(dtype=np.float16)).astype(dtype=np.float16)
        cosine_sim = self.cosine_similarity_n_space(own_tfidf_matrix_csr, own_tfidf_matrix_csr)
        # cosine_sim = cosine_similarity(own_tfidf_matrix_csr) # computing cosine similarity
        cosine_sim_df = pd.DataFrame(cosine_sim, index=self.df['slug_x'],
                                     columns=self.df['slug_x'])  # finding original record of post belonging to slug
        del cosine_sim
        self.cosine_sim_df = cosine_sim_df

    def cosine_similarity_n_space(self, m1, m2=None, batch_size=100):
        assert m1.shape[1] == m2.shape[1] and isinstance(batch_size, int) == True

        ret = np.ndarray((m1.shape[0], m2.shape[0]), dtype=np.float16)

        # iterating through dataframe by batches
        batches = m1.shape[0] // batch_size

        if m1.shape[0] % batch_size != 0:
            batches = batches + 1

        for row_i in range(0, batches):
            start = row_i * batch_size
            end = min([(row_i + 1) * batch_size, m1.shape[0]])
            rows = m1[start: end]
            sim = cosine_similarity(rows, m2)
            ret[start: end] = sim

        return ret

    def get_recommended_posts(self, find_by_string, data_frame, items, k=20):
        ix = data_frame.loc[:, find_by_string].to_numpy().argpartition(range(-1, -k, -1))
        closest = data_frame.columns[ix[-1:-(k + 2):-1]]
        # print("closest")
        # print(closest)
        # drop post itself
        closest = closest.drop(find_by_string, errors='ignore')

        # print("pd.DataFrame(closest).merge(items).head(k)")
        # print(pd.DataFrame(closest).merge(items).head(k))
        return pd.DataFrame(closest).merge(items).head(k)

    def convert_datframe_posts_to_json(self, post_recommendations, slug):
        list_of_article_slugs = []
        list_of_coefficients = []

        for index, row in post_recommendations.iterrows():
            # finding coefficient belonging to recommended posts compared to original post (for which we want to find recommendations)
            list_of_coefficients.append(self.cosine_sim_df.at[row['slug'], slug])

        post_recommendations['coefficient'] = list_of_coefficients

        dict = post_recommendations.to_dict('records')

        list_of_article_slugs.append(dict.copy())
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs[0])
        return list_of_article_slugs[0]


class TfIdf:

    def __init__(self):
        self.posts_df = None
        self.ratings_df = None
        self.categories_df = None
        self.df = None
        self.database = Database()
        self.user_categories_df = None
        self.tfidf_tuples = None
        self.tfidf_vectorizer = None
        self.cosine_sim_df = None

    def demo(self):
        # print("----------------------------------")
        # print("----------------------------------")
        # print("Multi-document comparison")
        # print("----------------------------------")
        # print("----------------------------------")
        sentence_1 = "S dnešním Mašinfírou se vrátíme do Jeseníků. Projedeme se po dráze, která začíná přímo pod úbočím Pradědu. Po povodních v roce 1997 málem zanikla, zachránili ji ale soukromníci. Prototypem řady 816 se projedeme z Vrbna pod Pradědem do Milotic nad Opavou."
        sentence_2 = "Za necelé dva týdny chce Milan Světlík začít veslovat napříč Atlantikem z New Yorku do Evropy. Pokud trasu dlouhou víc než 5 000 kilometrů zvládne, stane se prvním člověkem, který to dokázal sólo. „Bude to kombinace spousty extrémních věcí, ale nebavilo by mě dělat něco běžného, co už přede mnou někdo zvládl,“ svěřil se v rozhovoru pro magazín Víkend DNES."

        sentence_splitted_1 = sentence_1.split(" ")
        sentence_splitted_2 = sentence_2.split(" ")

        union_of_sentences = set(sentence_splitted_1).union(set(sentence_splitted_2))

        # print("union_of_sentences:")
        # print(union_of_sentences)

        wordDictA = dict.fromkeys(union_of_sentences, 0)
        wordDictB = dict.fromkeys(union_of_sentences, 0)

        for word in sentence_splitted_1:
            wordDictA[word] += 1

        for word in sentence_splitted_2:
            wordDictB[word] += 1

        # print(pd.DataFrame([wordDictA, wordDictB]))

        tf_computed1 = self.computeTF(wordDictA, sentence_splitted_1)
        tf_computed2 = self.computeTF(wordDictB, sentence_splitted_2)

        tfidf_vectorizer = pd.DataFrame([tf_computed1, tf_computed2])
        # print("tfidf_vectorizer:")
        # print(tfidf_vectorizer)

        idfs = self.computeIDF([wordDictA, wordDictB])

        idf_computed_1 = self.computeTFIDF(tf_computed1, idfs)
        idf_computed_2 = self.computeTFIDF(tf_computed2, idfs)

        idf = pd.DataFrame([idf_computed_1, idf_computed_2])

        # print("IDF")
        # print(idf)

        vectorize = TfidfVectorizer()
        vectors = vectorize.fit_transform(sentence_splitted_1)

        names = vectorize.get_feature_names()
        data = vectors.todense().tolist()
        df = pd.DataFrame(data, columns=names)
        # print("SKLEARN:")
        """
        for i in df.iterrows():
            print(i[1].sort_values(ascending=False)[:10])
        """

    def keyword_based_comparison(self, keywords):

        # print("----------------------------------")
        # print("----------------------------------")
        # print("Keyword based comparison")
        # print("----------------------------------")
        # print("----------------------------------")

        keywords_splitted_1 = keywords.split(" ")  # splitting sentence into list of keywords by space

        # creating dictionary of words
        wordDictA = dict.fromkeys(keywords_splitted_1, 0)

        for word in keywords_splitted_1:
            wordDictA[word] += 1

        # keywords_df = pd.DataFrame([wordDictA])

        recommenderMethods = RecommenderMethods()

        recommenderMethods.get_posts_dataframe()
        recommenderMethods.get_categories_dataframe()
        # tfidf.get_ratings_dataframe()
        recommenderMethods.join_posts_ratings_categories()

        # same as "classic" tf-idf
        fit_by_title_matrix = recommenderMethods.get_fit_by_feature('title_x', 'title_y')  # prepended by category
        fit_by_excerpt_matrix = recommenderMethods.get_fit_by_feature('excerpt')
        fit_by_keywords_matrix = recommenderMethods.get_fit_by_feature('keywords')

        tuple_of_fitted_matrices = (fit_by_title_matrix, fit_by_excerpt_matrix, fit_by_keywords_matrix)

        post_recommendations = recommenderMethods.recommend_by_keywords(keywords, tuple_of_fitted_matrices)

        del recommenderMethods
        return post_recommendations

    def get_ratings_dataframe(self):
        self.ratings_df = self.database.get_ratings_dataframe(pd)
        return self.ratings_df

    def get_user_categories(self):
        user_categories_df = self.database.get_user_categories(pd)
        return user_categories_df

    def join_post_ratings_categories(self, dataframe):
        recommenderMethods = RecommenderMethods()
        recommenderMethods.get_categories_dataframe()
        dataframe = dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        dataframe = dataframe[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description']]
        # print("dataframe afer joining with category")
        # print(dataframe.iloc[0])
        return dataframe.iloc[0]

    """

    def get_most_favourite_categories(self):
        self.user_categories_df = self.get_user_categories()

        self.user_categories_df = self.user_categories_df.merge(self.categories_df, left_on='category_id', right_on='id')
        topic_popularity = (self.user_categories_df.title
                            .value_counts()
                            .sort_values(ascending=False))
        topic_popularity.head(10)
        return topic_popularity

    def get_weighted_average(self):
        self.compute_weighted_average_score(self.df).head(10)

    def compute_weighted_average_score(self, df, k=0.8):
        # n_views = self.df.groupby('id_x', sort=False).id_x.count()
        n_views = self.df['views']
        ratings = self.df.groupby('id_x', sort=False).value.mean()
        scores = ((1-k)*(n_views/n_views.max()) + k*(ratings/ratings.max())).to_numpy().argsort()[::-1]
        df_deduped = df.groupby('id_x', sort=False).agg({
            'title_x':'first',
            'title_y':'first', # category title
        })

        return df_deduped.assign(views=n_views).iloc[scores]

    def computeTF(self, wordDict, doc):
        tfDict = {}
        corpusCount = len(doc)
        for word, count in wordDict.items():
            tfDict[word] = count / float(corpusCount)
        return (tfDict)

    def computeIDF(self, docList):
        idfDict = {}
        N = len(docList)
        idfDict = dict.fromkeys(docList[0].keys(), 0)
        for word, val in idfDict.items():
            if val > 0:
                idfDict[word] += 1

        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / (float(val) + 1));

        return idfDict

    def computeTFIDF(self, tfBow, idfs):
        tfidf = {}
        for word, val in tfBow.items():
            tfidf[word] = val * idfs[word]
        return tfidf

    """

    # https://datascience.stackexchange.com/questions/18581/same-tf-idf-vectorizer-for-2-data-inputs
    def set_tfidf_vectorizer_combine_features(self):
        tfidf_vectorizer = TfidfVectorizer()
        self.df.drop_duplicates(subset=['title_x'], inplace=True)
        tf_train_data = pd.concat([self.df['title_y'], self.df['keywords'], self.df['title_x'], self.df['excerpt']])
        tfidf_vectorizer.fit_transform(tf_train_data)

        tf_idf_title_x = tfidf_vectorizer.transform(self.df['title_x'])
        tf_idf_title_y = tfidf_vectorizer.transform(self.df['title_y'])  # category title
        tf_idf_keywords = tfidf_vectorizer.transform(self.df['keywords'])
        tf_idf_excerpt = tfidf_vectorizer.transform(self.df['excerpt'])

        model = LogisticRegression()
        model.fit([tf_idf_title_x.shape, tf_idf_title_y.shape, tf_idf_keywords.shape, tf_idf_excerpt.shape],
                  self.df['excerpt'])

    def set_cosine_sim(self):
        cosine_sim = cosine_similarity(self.tfidf_tuples)
        cosine_sim_df = pd.DataFrame(cosine_sim, index=self.df['slug_x'], columns=self.df['slug_x'])
        self.cosine_sim_df = cosine_sim_df

    # # @profile

    # @profile
    def get_cleaned_text(self, df, row):
        return row

    def get_recommended_posts_for_keywords(self, keywords, data_frame, items, k=10):

        keywords_list = []
        keywords_list.append(keywords)
        txt_cleaned = self.get_cleaned_text(self.df,
                                            self.df['title_x'] + self.df['title_y'] + self.df['keywords'] + self.df[
                                                'excerpt'])
        tfidf = self.tfidf_vectorizer.fit_transform(txt_cleaned)
        tfidf_keywords_input = self.tfidf_vectorizer.transform(keywords_list)
        cosine_similarities = cosine_similarity(tfidf_keywords_input, tfidf).flatten()
        # cosine_similarities = linear_kernel(tfidf_keywords_input, tfidf).flatten()

        data_frame['coefficient'] = cosine_similarities

        # related_docs_indices = cosine_similarities.argsort()[:-(number+1):-1]
        related_docs_indices = cosine_similarities.argsort()[::-1][:k]

        # print('index:', related_docs_indices)

        # print('similarity:', cosine_similarities[related_docs_indices])

        # print('\n--- related_docs_indices ---\n')

        # print(data_frame.iloc[related_docs_indices])

        # print('\n--- sort_values ---\n')
        # print(data_frame.sort_values('coefficient', ascending=False)[:k])

        closest = data_frame.sort_values('coefficient', ascending=False)[:k]

        closest.reset_index(inplace=True)
        # closest = closest.set_index('index1')
        closest['index1'] = closest.index
        # closest.index.name = 'index1'
        closest.columns.name = 'index'
        # # print("closest.columns.tolist()")
        # # print(closest.columns.tolist())
        # print("index name:")
        # print(closest.index.name)
        # print("""closest["slug_x","coefficient"]""")
        # print(closest[["slug_x","coefficient"]])

        return closest[["slug_x", "coefficient"]]
        # return pd.DataFrame(closest).merge(items).head(k)

    # @profile
    def recommend_posts_by_all_features_preprocessed(self, slug):

        recommenderMethods = RecommenderMethods()

        recommenderMethods.get_posts_dataframe()  # load posts to dataframe
        # print("posts dataframe:")
        # print(recommenderMethods.get_posts_dataframe())
        # print("posts categories:")
        # print(recommenderMethods.get_categories_dataframe())
        recommenderMethods.get_categories_dataframe()  # load categories to dataframe
        # tfidf.get_ratings_dataframe() # load post rating to dataframe

        recommenderMethods.join_posts_ratings_categories()  # joining posts and categories into one table
        print("posts ratings categories dataframe:")
        print(recommenderMethods.join_posts_ratings_categories())

        # feature tuples of (document_id, token_id) and coefficient
        # fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature('all_features_preprocessed')
        fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature('all_features_preprocessed')
        fit_by_title = recommenderMethods.get_fit_by_feature('title_y')

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title)
        print("tuple_of_fitted_matrices[0]")
        print(str(tuple_of_fitted_matrices[0]))
        gc.collect()
        post_recommendations = recommenderMethods.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del recommenderMethods
        return post_recommendations

    # @profile
    def recommend_posts_by_all_features_preprocessed_with_full_text(self, slug):

        recommenderMethods = RecommenderMethods()

        recommenderMethods.get_posts_dataframe()  # load posts to dataframe
        gc.collect()
        # print("posts dataframe:")
        # print(recommenderMethods.get_posts_dataframe())
        # print("posts categories:")
        # print(recommenderMethods.get_categories_dataframe())
        recommenderMethods.get_categories_dataframe()  # load categories to dataframe
        # tfidf.get_ratings_dataframe() # load post rating to dataframe

        recommenderMethods.join_posts_ratings_categories()  # joining posts and categories into one table
        print("posts ratings categories dataframe:")
        print(recommenderMethods.join_posts_ratings_categories())

        # replacing None values with empty strings
        recommenderMethods.df['full_text'] = recommenderMethods.df['full_text'].replace([None], '')

        # feature tuples of (document_id, token_id) and coefficient
        # fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature('all_features_preprocessed')
        fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature('all_features_preprocessed')
        fit_by_title = recommenderMethods.get_fit_by_feature('title_y')
        fit_by_full_text = recommenderMethods.get_fit_by_feature('full_text')

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_title, fit_by_all_features_matrix, fit_by_full_text)
        del fit_by_title
        del fit_by_all_features_matrix
        del fit_by_full_text
        gc.collect()

        print("tuple_of_fitted_matrices[0]")
        print(str(tuple_of_fitted_matrices[0]))

        post_recommendations = recommenderMethods.recommend_by_more_features_with_full_text(slug,
                                                                                            tuple_of_fitted_matrices)

        del recommenderMethods
        return post_recommendations

    # # @profile
    def recommend_posts_by_all_features(self, slug):

        recommenderMethods = RecommenderMethods()

        recommenderMethods.get_posts_dataframe()  # load posts to dataframe
        # print("posts dataframe:")
        # print(recommenderMethods.get_posts_dataframe())
        # print("posts categories:")
        # print(recommenderMethods.get_categories_dataframe())
        recommenderMethods.get_categories_dataframe()  # load categories to dataframe
        # tfidf.get_ratings_dataframe() # load post rating to dataframe

        recommenderMethods.join_posts_ratings_categories()  # joining posts and categories into one table
        print("posts ratings categories dataframe:")
        print(recommenderMethods.join_posts_ratings_categories())

        # preprocessing

        # feature tuples of (document_id, token_id) and coefficient
        fit_by_post_title_matrix = recommenderMethods.get_fit_by_feature('title_x', 'title_y')
        print("fit_by_post_title_matrix")
        print(fit_by_post_title_matrix)
        # fit_by_category_matrix = recommenderMethods.get_fit_by_feature('title_y')
        fit_by_excerpt_matrix = recommenderMethods.get_fit_by_feature('excerpt')
        print("fit_by_excerpt_matrix")
        print(fit_by_excerpt_matrix)
        fit_by_keywords_matrix = recommenderMethods.get_fit_by_feature('keywords')
        print("fit_by_keywords_matrix")
        print(fit_by_keywords_matrix)

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_post_title_matrix, fit_by_excerpt_matrix, fit_by_keywords_matrix)
        print("tuple_of_fitted_matrices[0]")
        print(str(tuple_of_fitted_matrices[0]))
        post_recommendations = recommenderMethods.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del recommenderMethods
        return post_recommendations

    """
        predictions_json = post_recommendations.to_json(orient="split")

        predictions_json_parsed = json.loads(predictions_json)

        return predictions_json_parsed
    """

    def convert_df_to_json(self, dataframe):
        result = dataframe[["title", "excerpt", "body"]].to_json(orient="records", lines=True)
        parsed = json.loads(result)
        return parsed

    def convert_to_json_one_row(self, key, value):
        list_for_json = []
        dict_for_json = {key: value}
        list_for_json.append(dict_for_json)
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_for_json)
        return list_for_json

    def convert_to_json_keyword_based(self, post_recommendations):

        list_of_article_slugs = []

        # post_recommendations['coefficient'] = list_of_coefficients

        dict = post_recommendations.to_dict('records')

        list_of_article_slugs.append(dict.copy())
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs[0])
        return list_of_article_slugs[0]

    def fill_recommended_for_all_posts(self, skip_already_filled):

        database = Database()
        database.connect()
        all_posts = database.get_all_posts()

        number_of_inserted_rows = 0

        for post in all_posts:

            post_id = post[0]
            slug = post[3]
            current_recommended = post[15]

            if skip_already_filled is True:
                if current_recommended is None:
                    actual_recommended_json = self.recommend_posts_by_all_features(slug)
                    actual_recommended_json = json.dumps(actual_recommended_json)
                    database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                     article_id=post_id)
                    number_of_inserted_rows += 1
                    # print(str(number_of_inserted_rows) + " rows insertd.")
                else:
                    print("Skipping.")
            else:
                actual_recommended_json = self.recommend_posts_by_all_features(slug)
                actual_recommended_json = json.dumps(actual_recommended_json)
                database.insert_recommended_json(articles_recommended_json=actual_recommended_json, article_id=post_id)
                number_of_inserted_rows += 1
                # print(str(number_of_inserted_rows) + " rows insertd.")

    def preprocess_dataframe(self):

        # preprocessing

        self.df["title_y"] = self.df["title_y"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["title_x"] = self.df["title_x"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"].map(lambda s: self.preprocess(s, stemming=False))

        print("datframe preprocessing:")
        print(self.df)

    def preprocess(self, sentence, stemming=False):
        # print(sentence)
        sentence = str(sentence)
        sentence = sentence.lower()
        sentence = sentence.replace('{html}', "")
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        cleantext.translate(str.maketrans('', '', string.punctuation))  # removing punctuation
        rem_url = re.sub(r'http\S+', '', cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        # print("rem_num")
        # print(rem_num)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)
        print("tokens:")
        print(tokens)
        filename = "text_classification/czech_stopwords.txt"

        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)

        tokens = [
            [word for word in document.lower().split() if word not in cz_stopwords]
            for document in tokens
        ]

        tokens = [item for sublist in tokens for item in sublist if len(sublist) > 0]

        print("tokens after lemma:")
        print(tokens)

        if stemming is True:
            print("STEMMING...")
            edited_words = [cz_stem(w, True) for w in tokens]  # aggresive
            edited_words = list(filter(None, edited_words))  # empty strings removal
        else:
            print("LEMMA...")
            edited_words = []
            for w in tokens:
                lemmatized_word = self.cz_lemma(str(w))
                if lemmatized_word is w:
                    lemmatized_word = self.cz_lemma(lemmatized_word.capitalize())
                    lemmatized_word = lemmatized_word.lower()
                edited_words.append(lemmatized_word)
            print(edited_words)
            edited_words = list(filter(None, edited_words))  # empty strings removal
            print(edited_words)
        # print(lemma_words)
        print("edited_words:")
        print(edited_words)
        return " ".join(edited_words)

    def preprocess_single_post(self, slug, json=False, stemming=False):
        post_dataframe = self.find_post_by_slug(slug)
        post_dataframe_joined = self.join_post_ratings_categories(post_dataframe)
        print("post_dataframe_joined:")
        print(post_dataframe_joined)
        post_dataframe["title_x"] = self.preprocess(post_dataframe_joined["title_x"], stemming)
        post_dataframe["excerpt"] = self.preprocess(post_dataframe_joined["excerpt"], stemming)
        post_dataframe["title_y"] = post_dataframe_joined["title_y"]
        print("preprocessing single post:")
        print(post_dataframe)
        if json is False:
            return post_dataframe
        else:
            return self.convert_df_to_json(post_dataframe)

    def cz_stem(self, string, aggresive="False", json=False):
        if aggresive == "False":
            aggressive_bool = False
        else:
            aggressive_bool = True

        if json is True:
            return self.convert_to_json_one_row("stem", cz_stem(string, aggressive_bool))
        else:
            return cz_stem(string, aggressive_bool)

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

    def get_document_scores(self):
        search_terms = 'Domácí. Zemřel poslední krkonošský nosič Helmut Hofer, ikona Velké Úpy. Ve věku 88 let zemřel potomek slavného rodu vysokohorských nosičů Helmut Hofer z Velké Úpy. Byl posledním žijícím nosičem v Krkonoších, starodávným řemeslem se po staletí živili generace jeho předků. Jako nosič pracoval pro Českou boudu na Sněžce mezi lety 1948 až 1953.'

        self.df["title_y"] = self.df["title_y"]
        self.df["title_x"] = self.df["title_x"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"]

        cols = ["keywords", "title_y", "title_x", "excerpt", "slug_x"]
        documents_df = pd.DataFrame()
        documents_df['features_combined'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        documents = list(map(' '.join, documents_df[['features_combined']].values.tolist()))

        filename = "text_classification/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)

        doc_vectors = TfidfVectorizer(dtype=np.float32, stop_words=cz_stopwords).fit_transform(
            [search_terms] + documents)

        cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()
        document_scores = [item.item() for item in cosine_similarities[1:]]

        return document_scores


""" -- Prepared:

    def fill_recommended_by_keywords_for_all_users(self):
        database = Database()
        database.connect()
        all_users = database.get_all_users()

        for user in all_users:

            user_id = user[0]

            database.get_user_tags(user_id)
            current_recommended = user[11]

            if current_recommended is None:
                actual_recommended_json = self.get_by_param(slug)
                # print("actual_recommended_json:")
                # print(actual_recommended_json)
                database.insert_recommended_json(articles_recommended_json=actual_recommended_json,article_id=post_id)
"""


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
        self.categories_df = self.database.get_categories_dataframe(pd)
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
        # post_found = self.(search_slug)

        # search_terms = 'Domácí. Zemřel poslední krkonošský nosič Helmut Hofer, ikona Velké Úpy. Ve věku 88 let zemřel potomek slavného rodu vysokohorských nosičů Helmut Hofer z Velké Úpy. Byl posledním žijícím nosičem v Krkonoších, starodávným řemeslem se po staletí živili generace jeho předků. Jako nosič pracoval pro Českou boudu na Sněžce mezi lety 1948 až 1953.'
        found_post_dataframe = recommenderMethods.find_post_by_slug(searched_slug)
        found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        print("post_dataframe.iloc[0]")
        # print(found_post_dataframe.iloc[0])
        found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                  found_post_dataframe.iloc[0]['title_y'] + " " + \
                                                  found_post_dataframe.iloc[0]['all_features_preprocessed']

        del self.posts_df
        del self.categories_df
        """
        self.df["title_y"] = self.df["title_y"]
        self.df["title_x"] = self.df["title_x"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"]
        """

        # cols = ["title_y", "title_x", "excerpt", "keywords", "slug_x", "all_features_preprocessed"]
        cols = ["all_features_preprocessed"]
        documents_df = pd.DataFrame()
        documents_df["features_to_use"] = self.df["keywords"] + '||' + self.df["title_y"] + ' ' + self.df[
            "all_features_preprocessed"]
        documents_df["slug"] = self.df["slug_x"]
        found_post = found_post_dataframe['features_to_use'].iloc[0]

        del self.df
        del found_post_dataframe
        # documents_df['features_combined'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        # documents = list(map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        # Uncomment for change of model
        # self.save_fast_text_to_w2v()
        # print("Loading word2vec model...")
        # self.save_full_model_to_smaller()

        # word2vec_embedding = KeyedVectors.load(self.amazon_bucket_url)
        # self.amazon_bucket_url#

        # word2vec_embedding = KeyedVectors.load(self.amazon_bucket_url)
        # global word2vec_embedding
        word2vec_embedding = KeyedVectors.load("models/w2v_model")

        # print("Model loaded...")
        # word2vec_embedding = KeyedVectors.load_word2vec_format("w2v_model",binary=False,unicode_errors='ignore')
        # print(test_word in word2vec_model.key_to_index))
        ds = DocSim(word2vec_embedding)
        # del word2vec_embedding
        # documents_df['features_to_use'] = documents_df.replace(',','', regex=True)
        documents_df['features_to_use'] = documents_df['features_to_use'] + "; " + documents_df['slug']
        list_of_document_features = documents_df["features_to_use"].tolist()
        del documents_df
        print("Calculating similarity...")
        # https://github.com/v1shwa/document-similarity with my edits
        print("list_of_document_features")
        most_similar_articles_with_scores = ds.calculate_similarity(found_post,
                                                                    list_of_document_features)[:21]
        print("sim_scores")
        print(most_similar_articles_with_scores)
        # removing post itself
        del most_similar_articles_with_scores[0]  # removing post itself

        # workaround due to float32 error in while converting to JSON
        return json.loads(json.dumps(most_similar_articles_with_scores, cls=NumpyEncoder))

    def get_similar_word2vec_full_text(self, searched_slug):
        tfidf = TfIdf()

        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories_full_text()
        # post_found = self.(search_slug)

        # search_terms = 'Domácí. Zemřel poslední krkonošský nosič Helmut Hofer, ikona Velké Úpy. Ve věku 88 let zemřel potomek slavného rodu vysokohorských nosičů Helmut Hofer z Velké Úpy. Byl posledním žijícím nosičem v Krkonoších, starodávným řemeslem se po staletí živili generace jeho předků. Jako nosič pracoval pro Českou boudu na Sněžce mezi lety 1948 až 1953.'
        found_post_dataframe = tfidf.find_post_by_slug(searched_slug)
        found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        print("post_dataframe.iloc[0]")
        # print(found_post_dataframe.iloc[0])
        found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                  found_post_dataframe.iloc[0]['title_y'] + " " + \
                                                  found_post_dataframe.iloc[0]['all_features_preprocessed'] + " " + \
                                                  found_post_dataframe.iloc[0]['body_preprocessed']

        del self.posts_df
        del self.categories_df
        """
        self.df["title_y"] = self.df["title_y"]
        self.df["title_x"] = self.df["title_x"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"]
        """

        # cols = ["title_y", "title_x", "excerpt", "keywords", "slug_x", "all_features_preprocessed"]
        cols = ["all_features_preprocessed"]
        documents_df = pd.DataFrame()
        print('self.df["keywords"]')
        print(self.df["keywords"])

        documents_df["features_to_use"] = self.df["keywords"] + '||' + self.df["title_y"] + ' ' + self.df[
            "all_features_preprocessed"] + self.df["body_preprocessed"]
        documents_df["slug"] = self.df["slug_x"]
        found_post = found_post_dataframe['features_to_use'].iloc[0]

        del self.df
        del found_post_dataframe
        # documents_df['features_combined'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        # documents = list(map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        # Uncomment for change of model
        # self.save_fast_text_to_w2v()
        # print("Loading word2vec model...")
        # self.save_full_model_to_smaller()

        # word2vec_embedding = KeyedVectors.load(self.amazon_bucket_url)
        # self.amazon_bucket_url#

        # word2vec_embedding = KeyedVectors.load(self.amazon_bucket_url)
        # global word2vec_embedding

        dropbox_access_token = "njfHaiDhqfIAAAAAAAAAAX_9zCacCLdpxxXNThA69dVhAsqAa_EwzDUyH1ZHt5tY"
        dropbox_file_download(dropbox_access_token, "models/w2v_model.vectors.npy", "/w2v_model.vectors.npy")
        word2vec_embedding = KeyedVectors.load("models/w2v_model")

        # print("Model loaded...")
        # word2vec_embedding = KeyedVectors.load_word2vec_format("w2v_model",binary=False,unicode_errors='ignore')
        # print(test_word in word2vec_model.key_to_index))
        ds = DocSim(word2vec_embedding)
        # del word2vec_embedding
        # documents_df['features_to_use'] = documents_df.replace(',','', regex=True)
        documents_df['features_to_use'] = documents_df['features_to_use'].str.replace(';', ' ')
        documents_df['features_to_use'] = documents_df['features_to_use'].str.replace(r'\r\n', '', regex=True)
        documents_df['features_to_use'] = documents_df['features_to_use'] + "; " + documents_df['slug']
        list_of_document_features = documents_df["features_to_use"].tolist()
        del documents_df
        print("Calculating similarity...")
        # https://github.com/v1shwa/document-similarity with my edits
        print("type(list_of_document_features)")
        print(type(list_of_document_features))
        print("list_of_document_features")
        most_similar_articles_with_scores = ds.calculate_similarity(found_post,
                                                                    list_of_document_features)[:21]
        print("sim_scores")
        print(most_similar_articles_with_scores)
        # removing post itself
        del most_similar_articles_with_scores[0]  # removing post itself

        # workaround due to float32 error in while converting to JSON
        return json.loads(json.dumps(most_similar_articles_with_scores, cls=NumpyEncoder))

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def save_full_model_to_smaller(self):
        word2vec_embedding = KeyedVectors.load_word2vec_format("full_models/w2v_model_full", limit=87000)  #
        word2vec_embedding.save("models/w2v_model")  # write separately=[] for all_in_one model

    def save_fast_text_to_w2v(self):
        word2vec_model = gensim.models.fasttext.load_facebook_vectors("full_models/cc.cs.300.bin.gz", encoding="utf-8")
        print("FastText loaded...")
        word2vec_model.fill_norms()
        word2vec_model.save_word2vec_format("full_models/w2v_model_full")
        print("Fast text saved...")


class Doc2VecClass:
    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/d2v_all_in_one.model"

    def __init__(self):
        self.documents = None
        self.df = None
        self.posts_df = None
        self.categories_df = None
        self.database = Database()

    def get_posts_dataframe(self):
        # self.database.insert_post_dataframe_to_cache() # uncomment for UPDATE of DB records
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    def get_categories_dataframe(self):
        self.categories_df = self.database.get_categories_dataframe(pd)
        return self.categories_df

    def join_posts_ratings_categories(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
             'all_features_preprocessed', 'body_preprocessed']]

    def train_doc2vec(self, documents_all_features_preprocessed):

        tagged_data = [TaggedDocument(words=_d.split(", "), tags=[str(i)]) for i, _d in
                       enumerate(documents_all_features_preprocessed)]

        print("tagged_data:")
        print(tagged_data)

        d2v_model = self.train_full_text(tagged_data)
        self.train_full_text(tagged_data)

        pickle.dump(d2v_model, open("d2v_all_in_one_with_full_text.model", "wb"))
        global doc2vec_model
        # model = pickle.load(smart_open.smart_open(self.amazon_bucket_url))
        # to find the vector of a document which is not in training data

        doc2vec_model = Doc2Vec.load("models/d2v_full_text.model")
        doc2vec_model.save("models/d2v_full_text_limited")  # write separately=[] for all_in_one model

    def get_similar_doc2vec(self, slug, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        ## merge more columns!
        # cols = ["title_y","title_x","excerpt","keywords"]
        tfidf = TfIdf()
        """
        self.df["title_y"] = self.df["title_y"]
        self.df["title_x"] = self.df["title_x"].map(lambda s: tfidf.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: tfidf.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"]
        """
        # cols = ["title_y","title_x","excerpt","keywords","slug_x"]

        cols = ['keywords', 'all_features_preprocessed']
        documents_df = pd.DataFrame()
        self.df['all_features_preprocessed'] = self.df['all_features_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))
        documents_df['all_features_preprocessed'] = self.df[cols].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                        axis=1)
        documents_df['all_features_preprocessed'] = self.df['title_y'] + ', ' + documents_df[
            'all_features_preprocessed']
        print("documents_df['all_features_preprocessed'].iloc[0]")
        print(documents_df['all_features_preprocessed'].iloc[0])

        documents_all_features_preprocessed = list(
            map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        del documents_df
        gc.collect()

        print("documents_all_features_preprocessed[0]")
        print(documents_all_features_preprocessed[0])
        del documents_all_features_preprocessed
        gc.collect()
        # print("documents_keywords[0]")
        # print(documents_keywords[0])
        documents_slugs = self.df['slug_x'].tolist()

        # documents = list(map(' '.join, self.df[["title_y","title_x","excerpt","keywords"]].values.tolist()))
        # documents_title = list(map(' '.join, self.df[["title_x"]].values.tolist()))
        # documents_keywords = list(map(' '.join, self.df[["keywords"]].values.tolist()))

        # print("document_keywords:")
        # print(document_keywords)
        filename = "text_classification/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

        # documents_df = pd.DataFrame(documents, columns=['title_x', 'keywords'])

        # self.train_doc2vec(documents_all_features_preprocessed)

        # print("tagged_data:")
        # print(tagged_data)

        # d2v_model = self.train(tagged_data)

        # pickle.dump(d2v_model,open("d2v_all_in_one.model", "wb" ))
        # global doc2vec_model
        doc2vec_model = Doc2Vec.load("models/d2v.model")
        # model = pickle.load(smart_open.smart_open(self.amazon_bucket_url))
        # to find the vector of a document which is not in training data
        recommenderMethods = RecommenderMethods()

        # not necessary
        post_found = recommenderMethods.find_post_by_slug(slug)
        # print("post_preprocessed:")
        # print(post_preprocessed)
        print(post_found.iloc[0])
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")
        tokens = keywords_preprocessed + all_features_preprocessed
        # post_features_to_find = post_preprocessed.iloc[0]['title']
        """
        print(post_features_to_find)
        print("post_features_to_find")
        """
        # tokens = post_features_to_find.split()
        print("tokens:")
        print(tokens)
        vector_source = doc2vec_model.infer_vector(tokens)
        print("vector_source:")
        print(vector_source)
        most_similar = doc2vec_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)
        """
        print("most_similar:")
        print(most_similar)
        print(self.get_similar_posts_slug(most_similar,documents_slugs,number_of_recommended_posts))
        """
        return self.get_similar_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_doc2vec_with_full_text(self, slug, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        ## merge more columns!
        # cols = ["title_y","title_x","excerpt","keywords"]
        tfidf = TfIdf()
        """
        self.df["title_y"] = self.df["title_y"]
        self.df["title_x"] = self.df["title_x"].map(lambda s: tfidf.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: tfidf.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"]
        """
        # cols = ["title_y","title_x","excerpt","keywords","slug_x"]

        cols = ['keywords', 'all_features_preprocessed', 'body_preprocessed']
        documents_df = pd.DataFrame()
        self.df['all_features_preprocessed'] = self.df['all_features_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))

        self.df.fillna("", inplace=True)

        self.df['body_preprocessed'] = self.df['body_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))
        documents_df['all_features_preprocessed'] = self.df[cols].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                        axis=1)
        # self.df['body_preprocessed'] = self.df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        documents_df['all_features_preprocessed'] = self.df['title_y'] + ', ' + documents_df[
            'all_features_preprocessed'] + ", " + self.df['body_preprocessed']
        print("documents_df['all_features_preprocessed'].iloc[0]")
        print(documents_df['all_features_preprocessed'].iloc[0])

        documents_all_features_preprocessed = list(
            map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        del documents_df
        gc.collect()

        print("documents_all_features_preprocessed[0]")
        print(documents_all_features_preprocessed[0])
        del documents_all_features_preprocessed
        gc.collect()
        # print("documents_keywords[0]")
        # print(documents_keywords[0])
        documents_slugs = self.df['slug_x'].tolist()

        # documents = list(map(' '.join, self.df[["title_y","title_x","excerpt","keywords"]].values.tolist()))
        # documents_title = list(map(' '.join, self.df[["title_x"]].values.tolist()))
        # documents_keywords = list(map(' '.join, self.df[["keywords"]].values.tolist()))

        # print("document_keywords:")
        # print(document_keywords)

        # self.train_doc2vec(documents_all_features_preprocessed)

        filename = "text_classification/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

        doc2vec_model = Doc2Vec.load("models/d2v_full_text_limited")

        recommendMethods = RecommenderMethods()

        # not necessary
        post_found = recommendMethods.find_post_by_slug(slug)
        # print("post_preprocessed:")
        # print(post_preprocessed)
        print(post_found.iloc[0])
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")
        full_text = post_found.iloc[0]['body_preprocessed'].split(" ")
        tokens = keywords_preprocessed + all_features_preprocessed + full_text
        # post_features_to_find = post_preprocessed.iloc[0]['title']
        """
        print(post_features_to_find)
        print("post_features_to_find")
        """
        # tokens = post_features_to_find.split()
        print("tokens:")
        print(tokens)
        vector_source = doc2vec_model.infer_vector(tokens)
        print("vector_source:")
        print(vector_source)
        most_similar = doc2vec_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)
        """
        print("most_similar:")
        print(most_similar)
        print(self.get_similar_posts_slug(most_similar,documents_slugs,number_of_recommended_posts))
        """
        return self.get_similar_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_doc2vec_by_keywords(self, slug, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        ## merge more columns!
        # cols = ["title_y","title_x","excerpt","keywords"]
        tfidf = TfIdf()
        """
        self.df["title_y"] = self.df["title_y"]
        self.df["title_x"] = self.df["title_x"].map(lambda s: tfidf.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: tfidf.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"]
        """
        # cols = ["title_y","title_x","excerpt","keywords","slug_x"]

        cols = ["keywords"]
        documents_df = pd.DataFrame()
        documents_df['keywords'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        documents = list(map(' '.join, documents_df[['keywords']].values.tolist()))
        documents_slugs = self.df['slug_x'].tolist()

        # documents = list(map(' '.join, self.df[["title_y","title_x","excerpt","keywords"]].values.tolist()))
        # documents_title = list(map(' '.join, self.df[["title_x"]].values.tolist()))
        # documents_keywords = list(map(' '.join, self.df[["keywords"]].values.tolist()))

        # print("document_keywords:")
        # print(document_keywords)
        filename = "text_classification/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

        # documents_df = pd.DataFrame(documents, columns=['title_x', 'keywords'])

        # documents_df = pd.DataFrame(documents)
        # title_df = pd.DataFrame(documents_title, columns=['title_x'])
        # slug_df = pd.DataFrame(documents_title, columns=['slug_x'])

        """
        documents_df['documents_cleaned'] = documents_df.title_x.apply(lambda x: " ".join(
            re.sub(r'(?![\d_])\w', ' ', w).lower() for w in x.split() if
            # re.sub(r'[^a-zA-Z]', ' ', w).lower() not in cz_stopwords))

        documents_df['keywords_cleaned'] = documents_df.keywords.apply(lambda x: " ".join(
            w.lower() for w in x.split()
            if w.lower() not in cz_stopwords))

        documents_df['title_x'] = title_df.title_x.apply(lambda x: " ".join(
            w.lower() for w in x.split()
            if w.lower() not in cz_stopwords))

        documents_df['slug_x'] = slug_df.slug_x
        """
        # keywords_cleaned = list(map(' '.join, documents_df[['keywords_cleaned']].values.tolist()))
        # print("keywords_cleaned:")
        # print(keywords_cleaned[:20])
        """
        self.train()
        """

        # model = Doc2Vec.load()
        # to find the vector of a document which is not in training data
        tfidf = TfIdf()
        # post_preprocessed = word_tokenize("Zemřel poslední krkonošský nosič Helmut Hofer, ikona Velké Úpy. Ve věku 88 let zemřel potomek slavného rodu vysokohorských nosičů Helmut Hofer z Velké Úpy. Byl posledním žijícím nosičem v Krkonoších, starodávným řemeslem se po staletí živili generace jeho předků. Jako nosič pracoval pro Českou boudu na Sněžce mezi lety 1948 až 1953.".lower())
        # post_preprocessed = tfidf.preprocess_single_post("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy")

        # not necessary
        post_preprocessed = tfidf.preprocess_single_post(slug)
        # print("post_preprocessed:")
        # print(post_preprocessed)
        # print(post_preprocessed.iloc[0])
        post_features_to_find = post_preprocessed.iloc[0]['keywords']
        """
        print(post_features_to_find)
        print("post_features_to_find")
        """
        tokens = post_features_to_find.split()
        """
        print("tokens:")
        print(tokens)
        """
        global doc2vec_model

        doc2vec_model = Doc2Vec.load("models/d2v.models")
        vector = doc2vec_model.infer_vector(tokens)
        """
        print("vector:")
        print(vector)
        """
        most_similar = doc2vec_model.docvecs.most_similar([vector], topn=number_of_recommended_posts)
        """
        print("most_similar:")
        print(most_similar)
        print(self.get_similar_posts_slug(most_similar,documents_slugs,number_of_recommended_posts))
        """
        return self.get_similar_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_posts_slug(self, most_similar, documents_slugs, number_of_recommended_posts):
        print('\n')

        post_recommendations = pd.DataFrame()
        list_of_article_slugs = []
        list_of_coefficients = []

        most_similar = most_similar[1:number_of_recommended_posts]

        # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD-MOST', 2), ('FOURTH-MOST', 3), ('FIFTH-MOST', 4), ('MEDIAN', len(most_similar) // 2), ('LEAST', len(most_similar) - 1)]:
        for index in range(0, len(most_similar)):
            print(u'%s: %s\n' % (most_similar[index][1], documents_slugs[int(most_similar[index][0])]))
            list_of_article_slugs.append(documents_slugs[int(most_similar[index][0])])
            list_of_coefficients.append(most_similar[index][1])
        print('=====================\n')

        post_recommendations['slug'] = list_of_article_slugs
        post_recommendations['coefficient'] = list_of_coefficients

        dict = post_recommendations.to_dict('records')

        list_of_articles = []

        list_of_articles.append(dict.copy())
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs[0])
        return self.flatten(list_of_articles)

    def train(self, tagged_data):

        max_epochs = 20
        vec_size = 150
        alpha = 0.025
        minimum_alpha = 0.0025
        reduce_alpha = 0.0002

        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha,
                        min_count=0,
                        dm=0)

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

        model.save("models/d2v.model")
        print("LDA model Saved")

    def train_full_text(self, tagged_data):

        max_epochs = 20
        vec_size = 150
        alpha = 0.025
        minimum_alpha = 0.0025
        reduce_alpha = 0.0002

        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha,
                        min_count=0,
                        dm=0, max_vocab_size=87000)

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

        model.save("models/d2v_full_text_limited.model")
        print("LDA model Saved")

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def most_similar(self, model, search_term):

        inferred_vector = model.infer_vector(search_term)

        sims = model.docvecs.most_similar([inferred_vector], topn=20)

        res = []
        for elem in sims:
            inner = {}
            inner['index'] = elem[0]
            inner['distance'] = elem[1]
            res.append(inner)

        return (res[:20])


class Lda:
    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/lda_all_in_one"

    def __init__(self):
        self.documents = None
        self.df = None
        self.posts_df = None
        self.categories_df = None
        self.database = Database()

    def join_posts_ratings_categories(self):
        self.get_categories_dataframe()
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
             'all_features_preprocessed', 'body_preprocessed']]
        return self.df

    def get_categories_dataframe(self):
        self.categories_df = self.database.get_categories_dataframe(pd)
        return self.categories_df

    def get_posts_dataframe(self):
        # self.database.insert_posts_dataframe_to_cache()  # uncomment for UPDATE of DB records
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    def apply_tokenize(self, text):
        print("text")
        return text.split(" ")

    @profile
    def get_similar_lda(self, searched_slug, N=21):
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()

        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))
        print("self.df['tokenized_keywords'][0]")
        print(self.df['tokenized_keywords'][0])

        gc.collect()

        self.df['tokenized_all_features_preprocessed'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))

        # self.df['tokenized'] = self.df.all_features_preprocessed_stopwords_clear.apply(lambda x: x.split(' '))
        self.df['tokenized'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))
        print("self.df['tokenized']")
        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed']

        """

        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        # self.df['tokenized'] = self.df.all_features_preprocessed_stopwords_clear.apply(lambda x: x.split(' '))
        self.df['tokenized'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))
        print("self.df['tokenized']")
        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized']
        all_words = [word for item in list(self.df["tokenized"]) for word in item]

        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)
        # self.lda_stats(fdist=fdist,searched_slug=searched_slug)
        k = 15000
        top_k_words, _ = zip(*fdist.most_common(k))
        self.top_k_words = set(top_k_words)

        self.df['tokenized'] = self.df['tokenized'].apply(self.keep_top_k_words)
        print("self.df['tokenized']")
        print(self.df['tokenized'])

        # document length
        self.df['doc_len'] = self.df['tokenized'].apply(lambda x: len(x))
        doc_lengths = list(self.df['doc_len'])
        self.df.drop(labels='doc_len', axis=1, inplace=True)

        minimum_amount_of_words = 30

        self.df = self.df[self.df['tokenized'].map(len) >= minimum_amount_of_words]
        # make sure all tokenized items are lists
        self.df = self.df[self.df['tokenized'].map(type) == list]
        self.df.reset_index(drop=True, inplace=True)
        print("After cleaning and excluding short articles, the dataframe now has:", len(self.df), "articles")
        print("df head:")
        print(self.df.head)

        

        print("dictionary, corpus, lda")

        # bow = dictionary.doc2bow(self.df.iloc[searched_doc_id, 11])
        # print("new_bow")
        # print(new_bow)
        # searched_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=bow)])
        # print("new_doc_distribution")
        # print(new_doc_distribution)
        """
        # self.train_lda(self.df)
        dictionary, corpus, lda = self.load_lda(self.df)

        searched_doc_id_list = self.df.index[self.df['slug_x'] == searched_slug].tolist()
        searched_doc_id = searched_doc_id_list[0]
        print(self.df.columns)
        new_bow = dictionary.doc2bow(self.df.iloc[searched_doc_id, 11])
        new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])

        # new_bow = dictionary.doc2bow(self.df.iloc[searched_doc_id, 11])
        # new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])

        print("most_sim_ids, most_sim_coefficients")
        # most_sim_ids = self.get_most_similar_documents(new_doc_distribution, doc_topic_dist)[0]

        doc_topic_dist = np.load('precalc_vectors/lda_doc_topic_dist.npy')
        most_sim_ids, most_sim_coefficients = self.get_most_similar_documents(new_doc_distribution, doc_topic_dist, N)
        print("most_sim_ids")
        print(most_sim_ids)

        # most_sim_ids = np.insert(most_sim_ids, 0, searched_doc_id) # appending post itself
        # most_similar_df = self.df[self.df.index.isin(most_sim_ids)]

        most_similar_df = self.df.iloc[most_sim_ids]
        print(most_similar_df)
        most_similar_df = most_similar_df.iloc[1:, :]
        list_of_coefficients = []
        """
        print("most_sim_coefficients[:K]")
        print(most_sim_coefficients[:N])
        """
        post_recommendations = pd.DataFrame()
        post_recommendations['slug'] = most_similar_df['slug_x'].iloc[:N]
        post_recommendations['coefficient'] = most_sim_coefficients[:N - 1]

        print("post_recommendations")
        print(post_recommendations)
        dict = post_recommendations.to_dict('records')

        list_of_articles = []

        list_of_articles.append(dict.copy())
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs[0])
        return self.flatten(list_of_articles)


    @profile
    def get_similar_lda_full_text(self, searched_slug, N=21):
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()

        """

        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))
        print("self.df['tokenized_keywords'][0]")
        print(self.df['tokenized_keywords'][0])

        self.df.fillna("", inplace=True)
        
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)

        self.df['tokenized_full_text'] = self.df.apply(
            lambda row: row['body_preprocessed'].replace(str(row['tokenized']), ''),
            axis=1)
        self.df['tokenized_all_features_preprocessed'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))

        gc.collect()

        self.df['tokenized_full_text'] = self.df.tokenized_full_text.apply(lambda x: x.split(' '))
        print("self.df['tokenized']")
        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed'] + self.df['tokenized_full_text']
        print("all_words")
        all_words = [word for item in list(self.df["tokenized"]) for word in item]
        print("freq_dist")
        fdist = FreqDist(all_words)
        # self.lda_stats(fdist=fdist,searched_slug=searched_slug)
        k = 15000
        print("zip(*fdist.most_common(k))")
        top_k_words, _ = zip(*fdist.most_common(k))
        self.top_k_words = set(top_k_words)

        print("self.df['tokenized']")
        self.df['tokenized'] = self.df['tokenized'].apply(self.keep_top_k_words)
        """
        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))

        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)

        self.df['tokenized_full_text'] = self.df.apply(
            lambda row: row['body_preprocessed'].replace(str(row['tokenized']), ''),
            axis=1)
        self.df['tokenized_all_features_preprocessed'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))

        gc.collect()

        self.df['tokenized_full_text'] = self.df.tokenized_full_text.apply(lambda x: x.split(' '))
        print("self.df['tokenized']")

        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed'] + self.df[
            'tokenized_full_text']

        print("searched_doc_id_list")
        searched_doc_id_list = self.df.index[self.df['slug_x'] == searched_slug].tolist()
        searched_doc_id = searched_doc_id_list[0]

        print("dictionary,corpus, lda")
        dictionary, corpus, lda = self.load_lda_full_text(self.df)
        print("self.df")
        print(self.df)
        new_bow = dictionary.doc2bow(self.df.iloc[searched_doc_id, 11])
        print("new_doc_distribution")
        new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])

        print("most_sim_ids, most_sim_coefficients")
        doc_topic_dist = np.load('precalc_vectors/lda_doc_topic_dist_full_text.npy')

        most_sim_ids, most_sim_coefficients = self.get_most_similar_documents(new_doc_distribution, doc_topic_dist, N)

        print("most_sim_ids")
        print(most_sim_ids)
        print("most_similar_df")

        most_similar_df = self.df.iloc[most_sim_ids]
        print(most_similar_df)
        most_similar_df = most_similar_df.iloc[1:, :]
        list_of_coefficients = []
        post_recommendations = pd.DataFrame()
        post_recommendations['slug'] = most_similar_df['slug_x'].iloc[:N]
        post_recommendations['coefficient'] = most_sim_coefficients[:N - 1]

        print("post_recommendations")
        print(post_recommendations)
        dict = post_recommendations.to_dict('records')

        list_of_articles = []

        list_of_articles.append(dict.copy())
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs[0])
        return self.flatten(list_of_articles)

    def lda_stats(self, fdist, doc_distribution, searched_slug):
        print("len(fdist)")
        print(len(fdist))  # number of unique words

        print("Most common words:")
        print(top_k_words[:10])

        """
        # bar plot of topic distribution for this document
        fig, ax = plt.subplots(figsize=(12, 6));
        # the histogram of the data
        patches = ax.bar(np.arange(len(doc_distribution)), doc_distribution)
        ax.set_xlabel('Topic ID', fontsize=15)
        ax.set_ylabel('Topic Contribution', fontsize=15)
        ax.set_title("Topic Distribution for Article " + str(searched_slug), fontsize=20)
        ax.set_xticks(np.linspace(10, 100, 10))
        fig.tight_layout()
        plt.show()
        """

        print("TOP 10 TOPICS:")
        """
        for i in new_doc_distribution.argsort()[-5:][::-1]:
            print(i, lda.show_topic(topicid=i, topn=10), "\n")
        """

        # print(top_k_words)
        # print("top_k_words")
        # print("top_k_words[-10:]")
        # print(top_k_words[-10:])


    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def get_most_similar_documents(self, query, matrix, k=20):
        """
        This function implements the Jensen-Shannon distance above
        and retruns the top k indices of the smallest jensen shannon distances
        """
        sims = self.jensen_shannon(query, matrix)  # list of jensen shannon distances

        sorted_k_result = sims.argsort()[:k]
        sims = sorted(sims, reverse=True)
        print("sims")
        # print(sims)
        print("sorted_k_result")
        print(sorted_k_result)
        return sorted_k_result, sims  # the top k positional index of the smallest Jensen Shannon distances

    def jensen_shannon(self, query, matrix):
        """
        This function implements a Jensen-Shannon similarity
        between the input query (an LDA topic distribution for a document)
        and the entire corpus of topic distributions.
        It returns an array of length M where M is the number of documents in the corpus
        """
        # lets keep with the p,q notation above
        p = query[None, :].T  # take transpose
        q = matrix.T  # transpose matrix
        m = 0.5 * (p + q)
        return np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))

    def keep_top_k_words(self, text):
        return [word for word in text if word in self.top_k_words]

    def load_lda(self, data, training_now=False):
        """
        print("dictionary")
        dictionary = corpora.Dictionary(data['tokenized'])
        print("corpus")
        corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]

        self.save_corpus_dict(corpus,dictionary)
        """
        try:
            lda_model = LdaModel.load("models/lda_model")
        except FileNotFoundError:
            dropbox_access_token = "njfHaiDhqfIAAAAAAAAAAX_9zCacCLdpxxXNThA69dVhAsqAa_EwzDUyH1ZHt5tY"
            dropbox_file_download(dropbox_access_token, "models/lda_model", "/lda_model")
            dropbox_file_download(dropbox_access_token, "models/lda_model.expElogbeta.npy", "/lda_model.expElogbeta.npy")
            dropbox_file_download(dropbox_access_token, "models/lda_model.id2word", "/lda_model.id2word")
            dropbox_file_download(dropbox_access_token, "models/lda_model.state", "/lda_model.state")
            dropbox_file_download(dropbox_access_token, "models/lda_model.state.sstats.npy", "/lda_model.state.sstats.npy")

            lda_model = LdaModel.load("models/lda_model")
        try:
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary.gensim')
            corpus = pickle.load(open('precalc_vectors/corpus.pkl', 'rb'))
        except FileNotFoundError:
            if training_now is False:
                self.train_lda(data)

            lda_model = LdaModel.load("models/lda_model")
            print("dictionary")
            dictionary = corpora.Dictionary(data['tokenized'])
            print("corpus")
            corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]

            self.save_corpus_dict(corpus, dictionary)

        return dictionary, corpus, lda_model

    def save_corpus_dict(self, corpus, dictionary):

        print("Saving corpus and dictionary...")
        pickle.dump(corpus, open('precalc_vectors/corpus.pkl', 'wb'))
        dictionary.save('precalc_vectors/dictionary.gensim')

    def load_lda_full_text(self, data, training_now=False):

        try:
            lda_model = LdaModel.load("models/lda_model_full_text")
        except FileNotFoundError:
            print("Downloading LDA model files...")
            dropbox_access_token = "njfHaiDhqfIAAAAAAAAAAX_9zCacCLdpxxXNThA69dVhAsqAa_EwzDUyH1ZHt5tY"
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text", "/lda_model_full_text")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.expElogbeta.npy", "/lda_model_full_text.expElogbeta.npy")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.id2word", "/lda_model_full_text.id2word")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.state", "/lda_model_full_text.state")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.state.sstats.npy", "/lda_model_full_text.state.sstats.npy")
            print("LDA Model files downloaded")

            lda_model = LdaModel.load("models/lda_model_full_text")
        try:
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_full_text.gensim')
            corpus = pickle.load(open('precalc_vectors/corpus_full_text.pkl', 'rb'))
        except FileNotFoundError:
            # COMMENT FOR FASTER PERFORMANCE
            if training_now is False:
                self.train_lda_full_text(data)

            lda_model = LdaModel.load("models/lda_model_full_text")
            print("dictionary")
            dictionary = corpora.Dictionary(data['tokenized'])
            print("corpus")
            corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]

            self.save_corpus_dict_full_text(corpus,dictionary)

        return dictionary, corpus, lda_model

    def save_corpus_dict_full_text(self, corpus, dictionary):

        print("Saving corpus and dictionary...")

        pickle.dump(corpus, open('precalc_vectors/corpus_full_text.pkl', "wb"))
        dictionary.save('precalc_vectors/dictionary_full_text.gensim')

    def train_lda(self, data):
        """
        2 passes of the data since this is a small dataset, so we want the distributions to stabilize
        """
        dictionary = corpora.Dictionary(data['tokenized'])
        corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
        num_topics = 400
        chunksize = 300
        t1 = time.time()

        # low alpha means each document is only represented by a small number of topics, and vice versa
        # low eta means each topic is only represented by a small number of words, and vice versa

        print("LDA training...")
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, minimum_probability=0.0,
                             chunksize=chunksize, alpha='auto', eta='auto',
                             passes=2)
        t2 = time.time()
        print("Time to train LDA model on ", len(self.df), "articles: ", (t2 - t1) / 60, "min")

        # native gensim method (abandoned due to not storing to single file like it should with separately=[] option)
        lda_model.save("models/lda_model")
        # pickle.dump(lda_model_local, open("lda_all_in_one", "wb"))
        print("Model Saved")
        # Compute Perplexity
        # print('\nPerplexity: ', lda_model.log_perplexity(corpus))
        # a measure of how good the model is. lower the better.
        """
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data['tokenized'], dictionary=dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        """
        # native gensim method (abandoned due to not storing to single file like it should with separately=[] option)
        # lda_model = LdaModel.load("models/lda_model")
        # lda_model_local = pickle.load(smart_open.smart_open("lda_all_in_one"))
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()

        documents_df = pd.DataFrame()
        # print("documents_df:")
        # print(documents_df)

        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))
        print("self.df['tokenized_keywords'][0]")
        print(self.df['tokenized_keywords'][0])
        """
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed_stopwords_clear'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        """
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        # self.df['tokenized'] = self.df.all_features_preprocessed_stopwords_clear.apply(lambda x: x.split(' '))
        self.df['tokenized'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))
        print("self.df['tokenized']")
        # print(self.df['tokenized'].iloc[0])
        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized']
        # documents = list(map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))
        # self.df['tokenized'] = self.df['all_features_preprocessed']
        # print("self.df['tokenized'].iloc[0]")
        # print(self.df['tokenized'].iloc[0])
        all_words = [word for item in list(self.df["tokenized"]) for word in item]
        # print("all_words[:10]")
        # print(all_words[:10])
        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)
        # print("fdist")
        # print(fdist)
        """
        print("len(fdist)")
        print(len(fdist))  # number of unique words
        """
        k = 15000
        top_k_words = fdist.most_common(k)
        # print(top_k_words)
        # print("top_k_words")
        # print("top_k_words[-10:]")
        # print(top_k_words[-10:])
        top_k_words, _ = zip(*fdist.most_common(k))

        self.top_k_words = set(top_k_words)
        # print("Bottom of the top " + str(k) + " words")
        # print(self.top_k_words)
        # print("Most common words:")
        # print(top_k_words[:10])

        self.df['tokenized'] = self.df['tokenized'].apply(self.keep_top_k_words)
        print("self.df['tokenized']")
        print(self.df['tokenized'])

        # document length
        self.df['doc_len'] = self.df['tokenized'].apply(lambda x: len(x))
        doc_lengths = list(self.df['doc_len'])
        self.df.drop(labels='doc_len', axis=1, inplace=True)

        minimum_amount_of_words = 30

        self.df = self.df[self.df['tokenized'].map(len) >= minimum_amount_of_words]
        # make sure all tokenized items are lists
        self.df = self.df[self.df['tokenized'].map(type) == list]
        self.df.reset_index(drop=True, inplace=True)
        print("After cleaning and excluding short aticles, the dataframe now has:", len(self.df), "articles")
        print("df head:")
        print(self.df.head)

        dictionary, corpus, lda = self.load_lda(self.df,training_now=True)

        # print("Most common topics found:")
        # print(lda.show_topics(num_topics=10, num_words=20))
        # print(lda.show_topic(topicid=4, topn=20))

        print("doc_topic_dist")
        doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
        print("np.save")
        # save doc_topic_dist
        # https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        np.save('precalc_vectors/lda_doc_topic_dist.npy',
                doc_topic_dist)  # IndexError: index 14969 is out of bounds for axis 1 with size 14969
        print("LDA model and documents topic distribution saved")

    def train_lda_full_text(self, data, lst=None):
        """
        2 passes of the data since this is a small dataset, so we want the distributions to stabilize
        """
        dictionary = corpora.Dictionary(data['tokenized'])
        corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
        num_topics = 400
        chunksize = 300
        t1 = time.time()

        # low alpha means each document is only represented by a small number of topics, and vice versa
        # low eta means each topic is only represented by a small number of words, and vice versa

        print("LDA training...")
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, minimum_probability=0.0,
                             chunksize=chunksize, alpha='auto', eta='auto',
                             passes=2)
        t2 = time.time()
        print("Time to train LDA model on ", len(self.df), "articles: ", (t2 - t1) / 60, "min")

        # native gensim method (abandoned due to not storing to single file like it should with separately=[] option)
        lda_model.save("models/lda_model_full_text")
        # pickle.dump(lda_model_local, open("lda_all_in_one", "wb"))
        print("Model Saved")
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))
        # a measure of how good the model is. lower the better.
        """
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data['tokenized'], dictionary=dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        """

        self.get_posts_dataframe()
        self.join_posts_ratings_categories()

        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))
        print("self.df['tokenized_keywords'][0]")
        print(self.df['tokenized_keywords'][0])
        self.df.fillna("", inplace=True)

        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        self.df['tokenized_full_text'] = self.df.apply(
            lambda row: row['body_preprocessed'].replace(str(row['tokenized']), ''),
            axis=1)
        # self.df['tokenized'] = self.df.all_features_preprocessed_stopwords_clear.apply(lambda x: x.split(' '))
        self.df['tokenized_all_features_preprocessed'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))
        gc.collect()
        self.df['tokenized_full_text'] = self.df.tokenized_full_text.apply(lambda x: x.split(' '))

        print("self.df['tokenized']")
        # print(self.df['tokenized'].iloc[0])
        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed'] + self.df['tokenized_full_text']
        all_words = [word for item in list(self.df["tokenized"]) for word in item]

        fdist = FreqDist(all_words)
        """
        print("len(fdist)")
        print(len(fdist))  # number of unique words
        """
        k = 15000
        top_k_words = fdist.most_common(k)
        # print(top_k_words)
        # print("top_k_words")
        # print("top_k_words[-10:]")
        # print(top_k_words[-10:])
        top_k_words, _ = zip(*fdist.most_common(k))

        self.top_k_words = set(top_k_words)
        # print("Bottom of the top " + str(k) + " words")
        # print(self.top_k_words)
        # print("Most common words:")
        # print(top_k_words[:10])

        self.df['tokenized'] = self.df['tokenized'].apply(self.keep_top_k_words)
        print("self.df['tokenized']")
        print(self.df['tokenized'])

        # document length
        self.df['doc_len'] = self.df['tokenized'].apply(lambda x: len(x))
        doc_lengths = list(self.df['doc_len'])
        self.df.drop(labels='doc_len', axis=1, inplace=True)

        minimum_amount_of_words = 30

        self.df = self.df[self.df['tokenized'].map(len) >= minimum_amount_of_words]
        # make sure all tokenized items are lists
        self.df = self.df[self.df['tokenized'].map(type) == list]
        self.df.reset_index(drop=True, inplace=True)
        print("After cleaning and excluding short aticles, the dataframe now has:", len(self.df), "articles")
        print("df head:")
        print(self.df.head)

        dictionary, corpus, lda = self.load_lda_full_text(self.df, training_now=True)

        # print("Most common topics found:")
        # print(lda.show_topics(num_topics=10, num_words=20))
        # print(lda.show_topic(topicid=4, topn=20))

        doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])

        # save doc_topic_dist
        # https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        np.save('precalc_vectors/lda_doc_topic_dist_full_text.npy', doc_topic_dist)
        print("LDA model and documents topic distribution saved")

    def load_stopwords(self):
        filename = "text_classification/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
            return cz_stopwords


def dropbox_file_download(access_token, dropbox_file_path, local_folder_name):
    try:
        dbx = dropbox.Dropbox(access_token)

        """
        print("Dropbox Files:")
        for entry in dbx.files_list_folder('').entries:
            print(entry.path_lower)
        """
        dbx.files_download_to_file(dropbox_file_path, local_folder_name)

    except Exception as e:
        print(e)
        return False


def main():
    # database = Database()
    # database.insert_posts_dataframe_to_cache() # for update
    searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"  # print(doc2vecClass.get_similar_doc2vec(slug))

    # tfidf = TfIdf()

    # print(tfidf.recommend_posts_by_all_features_preprocessed(searched_slug))
    # print(tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(searched_slug))
    # print(tfidf.recommend_posts_by_all_features('sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik'))
    # print(tfidf.preprocess("Vítkovice prohrály důležitý zápas s Třincem po prodloužení"))
    # print(tfidf.recommend_posts_by_all_features_preprocessed('sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik'))

    # keywords = "fotbal hokej sport slavia"
    # # print(tfidf.keyword_based_comparison(keywords))

    # STEMMING
    # word = "rybolovný"
    # # print(cz_stem(word))
    # # print(cz_stem(word,aggressive=True))

    # langdata = simplemma.load_data('cs')
    # # print(simplemma.lemmatize(word, langdata))
    # # print(tfidf.cz_lemma("nejnevhodnější"))

    # print(tfidf.preprocess_single_post("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy",json=True))

    # gensim = GenSim()
    # gensim.get_recommended_by_slug("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy")

    # searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    # searched_slug = "facr-o-slavii-a-rangers-verime-v-objektivni-vysetreni-odmitame-rasismus"

    # doc2vecClass = Doc2VecClass()
    # print(doc2vecClass.get_similar_doc2vec(searched_slug))
    # print(doc2vecClass.get_similar_doc2vec_with_full_text(searched_slug))

    lda = Lda()
    print("--------------LDA------------------")
    print(lda.get_similar_lda(searched_slug))
    print("--------------LDA FULL TEXT------------------")
    print(lda.get_similar_lda_full_text(searched_slug))

    # lda = Lda()
    # print(lda.get_similar_lda('krasa-se-skryva-v-exotickem-ovoci-kosmetika-kterou-na-podzim-musite-mit'))
    # print(lda.get_similar_lda_full_text('krasa-se-skryva-v-exotickem-ovoci-kosmetika-kterou-na-podzim-musite-mit'))

    # word2vecClass = Word2VecClass()
    # print(word2vecClass.get_similar_word2vec(searched_slug))
    # print(word2vecClass.get_similar_word2vec_full_text(searched_slug))

    # print(psutil.cpu_percent())
    # print(psutil.virtual_memory())  # physical memory usage
    # print('memory % used:', psutil.virtual_memory()[2])
    """
    h = hpy()
    print(h.heap())
    """


if __name__ == "__main__": main()
