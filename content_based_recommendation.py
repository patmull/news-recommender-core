import gc
import json
import logging
import random
import re
import string
import time
from collections import defaultdict
from pathlib import Path
import pickle
import dropbox
import gensim
# import majka
import numpy as np
import pandas as pd
import psycopg2
from gensim import corpora
from gensim import models
from gensim import similarities

# remove for production
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import tqdm
import matplotlib.pyplot as plt

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

from gensim.models import TfidfModel, KeyedVectors, LdaModel, fasttext, Word2Vec, CoherenceModel, LdaMulticore
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
from cz_stemmer.czech_stemmer import cz_stem

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

@DeprecationWarning
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
    word2vec_embedding = KeyedVectors.load("models/w2v_model_limited")

    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/d2v_all_in_one.model"
    print("Loading Doc2Vec model")
    global doc2vec_model
    # doc2vec_model = pickle.load(smart_open.smart_open(amazon_bucket_url))
    # doc2vec_model = Doc2Vec.load("d2v_all_in_one.model")
    doc2vec_model = Doc2Vec.load("models/d2v_limited.model")

    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/lda_all_in_one"
    print("Loading LDA model")

    global lda_model
    # lda_model = pickle.load(smart_open.smart_open(amazon_bucket_url))
    # lda_model = Lda.load("lda_all_in_one")
    lda_model = LdaModel.load("models/lda_model")


global cz_stopwords


def load_stopwords():
    filename = "cz_stemmer/czech_stopwords.txt"
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
        cv = CountVectorizer(analyzer='word', min_df=10, stop_words='czech', lowercase=True,
                             token_pattern='[a-zA-Z0-9]{3,}')

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
        cv = CountVectorizer(analyzer='word', min_df=10, stop_words='czech', lowercase=True,
                             token_pattern='[a-zA-Z0-9]{3,}')

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
        filename = "cz_stemmer/czech_stopwords.txt"
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


class RecommenderMethods:

    def __init__(self):
        self.database = Database()

    def get_posts_dataframe(self):
        self.database.connect()
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        self.database.disconnect()
        return self.posts_df

    def get_categories_dataframe(self):
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe(pd)
        self.database.disconnect()
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

        # computing cosine similarity using matrix with combined features
        print("Computing cosine similarity using matrix with combined features...")
        self.set_cosine_sim_use_own_matrix(combined_matrix1)
        combined_all = self.get_recommended_posts_for_keywords(keywords, self.cosine_sim_df,
                                                               self.df[['keywords']])
        # print("combined_all:")
        # print(combined_all)
        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})
        # print("DF RENAMED:")
        # print(df_renamed)

        json = self.convert_to_json_keyword_based(df_renamed)

        return json

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

    def get_recommended_posts_for_keywords(self, keywords, data_frame, items, k=10):

        keywords_list = [keywords]
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

        closest = data_frame.sort_values('coefficient', ascending=False)[:k]

        closest.reset_index(inplace=True)
        # closest = closest.set_index('index1')
        closest['index1'] = closest.index
        # closest.index.name = 'index1'
        closest.columns.name = 'index'

        return closest[["slug_x", "coefficient"]]
        # return pd.DataFrame(closest).merge(items).head(k)

    def find_post_by_slug(self, slug):
        recommenderMethods = RecommenderMethods()
        return recommenderMethods.get_posts_dataframe().loc[recommenderMethods.get_posts_dataframe()['slug'] == slug]

    def get_cleaned_text(self, df, row):
        return row

    def get_tfIdfVectorizer(self, fit_by, fit_by_2=None, stemming=False):

        self.set_tfIdfVectorizer()

        if fit_by_2 is None:
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[
                                                                        fit_by])  # Metoda fit: výpočet průměru a rozptylu jednotlivých sloupců z dat. Metoda transformace: # transformuje všechny prvky pomocí příslušného průměru a rozptylu.
        else:
            self.df[fit_by] = self.df[fit_by_2] + ". " + self.df[fit_by]
            # # print(self.df[fit_by])
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[
                                                                        fit_by])  # Metoda fit: výpočet průměru a rozptylu jednotlivých sloupců z dat. Metoda transformace: # transformuje všechny prvky pomocí příslušného průměru a rozptylu.

        return self.tfidf_tuples  # tuples of (document_id, token_id) and tf-idf score for it

    def set_tfIdfVectorizer(self):
        # load czech stopwords from file
        filename = "cz_stemmer/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)

        tfidf_vectorizer = TfidfVectorizer(dtype=np.float32,
                                           stop_words=cz_stopwords)  # transforms text to feature vectors that can be used as input to estimator
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

        # getting posts with highest similarity
        combined_all = self.get_recommended_posts(slug, self.cosine_sim_df,
                                                  self.df[['slug_x']])

        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})

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
        # computing cosine similarity
        print("Computing cosine simialirity")
        self.set_cosine_sim_use_own_matrix(combined_matrix1)

        # getting posts with highest similarity
        combined_all = self.get_recommended_posts(slug, self.cosine_sim_df,
                                                  self.df[['slug_x']])

        df_renamed = combined_all.rename(columns={'slug_x': 'slug'})

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

        # finding coefficient belonging to recommended posts compared to original post (for which we want to find recommendations)
        for index, row in post_recommendations.iterrows():
            list_of_coefficients.append(self.cosine_sim_df.at[row['slug'], slug])

        post_recommendations['coefficient'] = list_of_coefficients
        dict = post_recommendations.to_dict('records')
        list_of_article_slugs.append(dict.copy())
        return list_of_article_slugs[0]

    @DeprecationWarning
    def download_from_amazon(self, amazon_bucket_url):
        amazon_model = pickle.load(smart_open.smart_open(self.amazon_bucket_url))
        return amazon_model

    def dropbox_file_download(self, access_token, dropbox_file_path, local_folder_name):
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
        # Compute Sparsicity = Percentage of Non-Zero cells
        print("Sparsicity: ", ((vectors.todense() > 0).sum() / vectors.todense().size) * 100, "%")
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

        print("Loading posts.")

        recommenderMethods.get_posts_dataframe()  # load posts to dataframe
        recommenderMethods.get_categories_dataframe()  # load categories to dataframe
        recommenderMethods.join_posts_ratings_categories()  # joining posts and categories into one table

        fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature('all_features_preprocessed')
        fit_by_title = recommenderMethods.get_fit_by_feature('title_y')

        tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title) # join feature tuples into one matrix

        gc.collect()

        post_recommendations = recommenderMethods.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del recommenderMethods
        return post_recommendations

    # @profile
    def recommend_posts_by_all_features_preprocessed_with_full_text(self, slug):

        recommenderMethods = RecommenderMethods()
        print("Loading posts")
        recommenderMethods.get_posts_dataframe()  # load posts to dataframe
        gc.collect()
        recommenderMethods.get_categories_dataframe()  # load categories to dataframe
        recommenderMethods.join_posts_ratings_categories()  # joining posts and categories into one table

        # replacing None values with empty strings
        recommenderMethods.df['full_text'] = recommenderMethods.df['full_text'].replace([None], '')

        fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature('all_features_preprocessed')
        fit_by_title = recommenderMethods.get_fit_by_feature('title_y')
        fit_by_full_text = recommenderMethods.get_fit_by_feature('full_text')

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_title, fit_by_all_features_matrix, fit_by_full_text)
        del fit_by_title
        del fit_by_all_features_matrix
        del fit_by_full_text
        gc.collect()

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
        post_recommendations = recommenderMethods.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del recommenderMethods
        return post_recommendations

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
        dict = post_recommendations.to_dict('records')
        list_of_article_slugs.append(dict.copy())
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
        filename = "cz_stemmer/czech_stopwords.txt"

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

        filename = "cz_stemmer/czech_stopwords.txt"
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
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe(pd)
        self.database.disconnect()
        return self.categories_df

    def join_posts_ratings_categories(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
             'all_features_preprocessed', 'body_preprocessed']]

    def train_doc2vec(self, documents_all_features_preprocessed, body_text, limited=True):

        tagged_data = [TaggedDocument(words=_d.split(", "), tags=[str(i)]) for i, _d in
                       enumerate(documents_all_features_preprocessed)]

        print("tagged_data:")
        print(tagged_data)

        self.train_full_text(tagged_data, body_text, limited)

        # amazon loading
        # amazon_bucket_url = "..."
        # recommenderMethods = RecommenderMethods()
        # model = recommenderMethods.download_from_amazon(amazon_bucket_url)

        # to find the vector of a document which is not in training data

    def get_similar_doc2vec(self, slug, train=False, limited=True, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        cols = ['keywords', 'all_features_preprocessed']
        documents_df = pd.DataFrame()
        self.df['all_features_preprocessed'] = self.df['all_features_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))
        documents_df['all_features_preprocessed'] = self.df[cols].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                        axis=1)
        documents_df['all_features_preprocessed'] = self.df['title_y'] + ', ' + documents_df[
            'all_features_preprocessed']

        documents_all_features_preprocessed = list(
            map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        del documents_df
        gc.collect()

        documents_slugs = self.df['slug_x'].tolist()

        """
        filename = "cz_stemmer/czech_stopwords.txt"

        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        """

        if train is True:
            self.train_doc2vec(documents_all_features_preprocessed, False)

        del documents_all_features_preprocessed
        gc.collect()

        # pickle.dump(d2v_model,open("d2v_all_in_one.model", "wb" ))
        # global doc2vec_model

        if limited is True:
            doc2vec_model = Doc2Vec.load("models/d2v_limited.model")
        else:
            doc2vec_model = Doc2Vec.load("models/d2v.model")
        # model = pickle.load(smart_open.smart_open(self.amazon_bucket_url))
        # to find the vector of a document which is not in training data
        recommenderMethods = RecommenderMethods()
        # not necessary
        post_found = recommenderMethods.find_post_by_slug(slug)
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")

        tokens = keywords_preprocessed + all_features_preprocessed

        vector_source = doc2vec_model.infer_vector(tokens)

        most_similar = doc2vec_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)
        return self.get_similar_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_doc2vec_with_full_text(self, slug, train=False, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        cols = ['keywords', 'all_features_preprocessed', 'body_preprocessed']
        documents_df = pd.DataFrame()
        self.df['all_features_preprocessed'] = self.df['all_features_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))

        self.df.fillna("", inplace=True)

        self.df['body_preprocessed'] = self.df['body_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))
        documents_df['all_features_preprocessed'] = self.df[cols].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                        axis=1)

        documents_df['all_features_preprocessed'] = self.df['title_y'] + ', ' + documents_df[
            'all_features_preprocessed'] + ", " + self.df['body_preprocessed']

        documents_all_features_preprocessed = list(
            map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        del documents_df
        gc.collect()

        documents_slugs = self.df['slug_x'].tolist()

        if train is True:
            self.train_doc2vec(documents_all_features_preprocessed, True)
        del documents_all_features_preprocessed
        gc.collect()

        """
        filename = "cz_stemmer/czech_stopwords.txt"
        
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        """
        doc2vec_model = Doc2Vec.load("models/d2v_full_text_limited.model")

        recommendMethods = RecommenderMethods()

        # not necessary
        post_found = recommendMethods.find_post_by_slug(slug)
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")
        full_text = post_found.iloc[0]['body_preprocessed'].split(" ")
        tokens = keywords_preprocessed + all_features_preprocessed + full_text
        vector_source = doc2vec_model.infer_vector(tokens)

        most_similar = doc2vec_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)

        return self.get_similar_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_doc2vec_by_keywords(self, slug, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        # cols = ["title_y","title_x","excerpt","keywords","slug_x"]

        cols = ["keywords"]
        documents_df = pd.DataFrame()
        documents_df['keywords'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        documents_slugs = self.df['slug_x'].tolist()

        filename = "cz_stemmer/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

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
        post_features_to_find = post_preprocessed.iloc[0]['keywords']

        tokens = post_features_to_find.split()

        global doc2vec_model

        doc2vec_model = Doc2Vec.load("models/d2v.models")
        vector = doc2vec_model.infer_vector(tokens)

        most_similar = doc2vec_model.docvecs.most_similar([vector], topn=number_of_recommended_posts)
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

    def train_full_text(self, tagged_data, full_body, limited):

        max_epochs = 20
        vec_size = 150
        alpha = 0.025
        minimum_alpha = 0.0025
        reduce_alpha = 0.0002

        if limited is True:
            model = Doc2Vec(vector_size=vec_size,
                            alpha=alpha,
                            min_count=0,
                            dm=0, max_vocab_size=87000)
        else:
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

        if full_body is True:
            if limited is True:
                model.save("models/d2v_full_text_limited.model")
            else:
                model.save("models/d2v_full_text.model")
        else:
            if limited is True:
                model.save("models/d2v_limited.model")
            else:
                model.save("models/d2v.model")
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
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe(pd)
        self.database.disconnect()
        return self.categories_df

    def get_posts_dataframe(self):
        # self.database.insert_posts_dataframe_to_cache()  # uncomment for UPDATE of DB records
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    def apply_tokenize(self, text):
        print("text")
        return text.split(" ")

    # @profile

    def get_similar_lda(self, searched_slug, train=False, display_dominant_topics=False, N=21):
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()

        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))

        gc.collect()

        self.df['tokenized_all_features_preprocessed'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))

        self.df['tokenized'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))
        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed']

        gc.collect()

        print(self.df.head(10).to_string())
        # if there is no LDA model, training will run anyway due to load method handle
        if train is True:
            self.train_lda(self.df, display_dominant_topics=display_dominant_topics)

        dictionary, corpus, lda = self.load_lda(self.df)

        searched_doc_id_list = self.df.index[self.df['slug_x'] == searched_slug].tolist()
        searched_doc_id = searched_doc_id_list[0]
        new_bow = dictionary.doc2bow(self.df.iloc[searched_doc_id, 11])
        new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])

        doc_topic_dist = np.load('precalc_vectors/lda_doc_topic_dist.npy')

        most_sim_ids, most_sim_coefficients = self.get_most_similar_documents(new_doc_distribution, doc_topic_dist, N)

        most_similar_df = self.df.iloc[most_sim_ids]

        del self.df
        gc.collect()

        most_similar_df = most_similar_df.iloc[1:, :]

        post_recommendations = pd.DataFrame()
        post_recommendations['slug'] = most_similar_df['slug_x'].iloc[:N]
        post_recommendations['coefficient'] = most_sim_coefficients[:N - 1]

        dict = post_recommendations.to_dict('records')

        list_of_articles = []
        list_of_articles.append(dict.copy())

        return self.flatten(list_of_articles)

    # @profile
    def get_similar_lda_full_text(self, searched_slug, N=21, train=False, display_dominant_topics=True):
        self.database.connect()
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()
        self.database.disconnect()

        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        self.df['tokenized_full_text'] = self.df.apply(
            lambda row: row['body_preprocessed'].replace(str(row['tokenized']), ''),
            axis=1)

        gc.collect()

        self.df['tokenized_all_features_preprocessed'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))
        gc.collect()
        self.df['tokenized_full_text'] = self.df.tokenized_full_text.apply(lambda x: x.split(' '))
        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed'] + self.df[
            'tokenized_full_text']
        gc.collect()

        if train is True:
            self.train_lda_full_text(self.df, display_dominant_topics=display_dominant_topics)

        dictionary, corpus, lda = self.load_lda_full_text(self.df, retrain=train, display_dominant_topics=display_dominant_topics)

        searched_doc_id_list = self.df.index[self.df['slug_x'] == searched_slug].tolist()
        searched_doc_id = searched_doc_id_list[0]
        new_bow = dictionary.doc2bow(self.df.iloc[searched_doc_id, 11])
        new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])

        doc_topic_dist = np.load('precalc_vectors/lda_doc_topic_dist_full_text.npy')

        most_sim_ids, most_sim_coefficients = self.get_most_similar_documents(new_doc_distribution, doc_topic_dist, N)

        most_similar_df = self.df.iloc[most_sim_ids]
        del self.df
        gc.collect()
        most_similar_df = most_similar_df.iloc[1:, :]
        post_recommendations = pd.DataFrame()
        post_recommendations['slug'] = most_similar_df['slug_x'].iloc[:N]
        post_recommendations['coefficient'] = most_sim_coefficients[:N - 1]

        dict = post_recommendations.to_dict('records')
        list_of_articles = []
        list_of_articles.append(dict.copy())

        return self.flatten(list_of_articles)

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

        return sorted_k_result, sims  # the top k positional index of the smallest Jensen Shannon distances

    # return sorted_k_result, sims  # the top k positional index of the smallest Jensen Shannon distances
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
        """
        try:
        except FileNotFoundError:

            dropbox_access_token = "njfHaiDhqfIAAAAAAAAAAX_9zCacCLdpxxXNThA69dVhAsqAa_EwzDUyH1ZHt5tY"
            dropbox_file_download(dropbox_access_token, "models/lda_model", "/lda_model")
            dropbox_file_download(dropbox_access_token, "models/lda_model.expElogeta.npy", "/lda_model.expElogeta.npy")
            dropbox_file_download(dropbox_access_token, "models/lda_model.id2word", "/lda_model.id2word")
            dropbox_file_download(dropbox_access_token, "models/lda_model.state", "/lda_model.state")
            dropbox_file_download(dropbox_access_token, "models/lda_model.state.sstats.npy", "/lda_model.state.sstats.npy")

            lda_model = LdaModel.load("models/lda_model")
        """
        try:
            lda_model = LdaModel.load("models/lda_model")
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary.gensim')
            corpus = pickle.load(open('precalc_vectors/corpus.pkl', 'rb'))
        except Exception as e:
            print("Could not load LDA models or precalculated vectors. Reason:")
            print(e)
            self.train_lda(data)

            lda_model = LdaModel.load("models/lda_model")
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary.gensim')
            corpus = pickle.load(open('precalc_vectors/corpus.pkl', 'rb'))

        return dictionary, corpus, lda_model

    def save_corpus_dict(self, corpus, dictionary):

        print("Saving corpus and dictionary...")
        pickle.dump(corpus, open('precalc_vectors/corpus.pkl', 'wb'))
        dictionary.save('precalc_vectors/dictionary.gensim')

    def load_lda_full_text(self, data, display_dominant_topics, training_now=False, retrain=False, ):

        """
        try:
            lda_model = LdaModel.load("models/lda_model_full_text")
        except FileNotFoundError:
            print("Downloading LDA model files...")
            dropbox_access_token = "njfHaiDhqfIAAAAAAAAAAX_9zCacCLdpxxXNThA69dVhAsqAa_EwzDUyH1ZHt5tY"
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text", "/lda_model_full_text")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.expElogeta.npy", "/lda_model_full_text.expElogeta.npy")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.id2word", "/lda_model_full_text.id2word")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.state", "/lda_model_full_text.state")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.state.sstats.npy", "/lda_model_full_text.state.sstats.npy")
            print("LDA Model files downloaded")

            lda_model = LdaModel.load("models/lda_model_full_text")
        """
        try:
            lda_model = LdaModel.load("models/lda_model_full_text")
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_full_text.gensim')
            corpus = pickle.load(open('precalc_vectors/corpus_full_text.pkl', 'rb'))
        except Exception as e:
            print("Could not load LDA models or precalculated vectors. Reason:")
            print(e)
            self.train_lda_full_text(data, display_dominant_topics)

            lda_model = LdaModel.load("models/lda_model_full_text")
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_full_text.gensim')
            corpus = pickle.load(open('precalc_vectors/corpus_full_text.pkl', 'rb'))

        return dictionary, corpus, lda_model

    def save_corpus_dict_full_text(self, corpus, dictionary):

        print("Saving corpus and dictionary...")
        pickle.dump(corpus, open('precalc_vectors/corpus_full_text.pkl', "wb"))
        dictionary.save('precalc_vectors/dictionary_full_text.gensim')

    def train_lda(self, data, display_dominant_topics=True):
        data_words_nostops = self.remove_stopwords(data['tokenized'])
        data_words_bigrams = self.build_bigrams_and_trigrams(data_words_nostops)

        print("data_words_bigrams")
        print(data_words_bigrams)

        self.df.assign(tokenized=data_words_bigrams)
        print("data['tokenized']")
        print(data['tokenized'])
        dictionary = corpora.Dictionary(data['tokenized'])
        dictionary.filter_extremes()
        corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
        num_topics = 100 # set according visualise_lda() method (Coherence value) = 100
        chunksize = 1000
        iterations = 200
        passes = 20 # evaluated on 20
        t1 = time.time()

        # low alpha means each document is only represented by a small number of topics, and vice versa
        # low eta means each topic is only represented by a small number of words, and vice versa
        print("LDA training...")
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                             minimum_probability=0.00,chunksize=chunksize,
                             alpha='auto', eta='auto',
                             passes=passes, iterations=iterations)
        t2 = time.time()
        print("Time to train LDA model on ", len(self.df), "articles: ", (t2 - t1) / 60, "min")

        # native gensim method (abandoned due to not storing to single file like it should with separately=[] option)
        lda_model.save("models/lda_model")
        # pickle.dump(lda_model_local, open("lda_all_in_one", "wb"))
        print("Model Saved")
        # lda_model = LdaModel.load("models/lda_model")
        # lda_model_local = pickle.load(smart_open.smart_open("lda_all_in_one"))
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()

        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        self.df['tokenized'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))
        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized']
        all_words = [word for item in list(self.df["tokenized"]) for word in item]
        print(all_words[:50])

        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)
        k = 15000
        top_k_words, _ = zip(*fdist.most_common(k))
        print(top_k_words)
        print("top_k_words")
        self.top_k_words = set(top_k_words)

        self.df['tokenized'] = self.df['tokenized'].apply(self.keep_top_k_words)

        minimum_amount_of_words = 2
        self.df = self.df[self.df['tokenized'].map(len) >= minimum_amount_of_words]
        # make sure all tokenized items are lists
        self.df = self.df[self.df['tokenized'].map(type) == list]
        self.df.reset_index(drop=True, inplace=True)
        print("After cleaning and excluding short aticles, the dataframe now has:", len(self.df), "articles")
        print("df head:")
        print(self.df.head)

        self.save_corpus_dict(corpus, dictionary)

        lda = lda_model.load("models/lda_model")
        doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
        print("np.save")
        # save doc_topic_dist
        # https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        np.save('precalc_vectors/lda_doc_topic_dist.npy',
                doc_topic_dist)  # IndexError: index 14969 is out of bounds for axis 1 with size 14969
        print("LDA model and documents topic distribution saved")

    def train_lda_full_text(self, data, display_dominant_topics=True, lst=None):

        data_words_nostops = self.remove_stopwords(data['tokenized'])
        data_words_bigrams = self.build_bigrams_and_trigrams(data_words_nostops)

        self.df.assign(tokenized=data_words_bigrams)

        # View
        all_words = [word for item in self.df['tokenized'] for word in item]
        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)
        k = 15000
        top_k_words, _ = zip(*fdist.most_common(k))
        self.top_k_words = set(top_k_words)

        print("self.df['tokenized']")
        print(self.df['tokenized'])

        print("self.df['tokenized']")
        print(self.df['tokenized'])

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

        t1 = time.time()

        num_topics = 20  # set according visualise_lda() method (Coherence value) = 20
        chunksize = 1000
        passes = 2 # evaluated on 20
        workers = 7  # change when used on different computer/server according tu no. of CPU cores
        eta = 'auto'
        per_word_topics = True
        iterations = 200

        print("LDA training...")
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                             minimum_probability=0.0, chunksize=chunksize,
                             eta=eta, alpha='auto',
                             passes=passes, iterations=iterations)
        t2 = time.time()
        print("Time to train LDA model on ", len(self.df), "articles: ", (t2 - t1) / 60, "min")

        lda_model.save("models/lda_model_full_text")
        print("Model Saved")

        if display_dominant_topics is True:
            self.display_dominant_topics(optimal_model=lda_model, corpus=corpus, texts=data_words_bigrams)


        lda = lda_model.load("models/lda_model_full_text")
        doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
        print("np.save")
        # save doc_topic_dist
        # https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        np.save('precalc_vectors/lda_doc_topic_dist_full_text.npy',
                doc_topic_dist)  # IndexError: index 14969 is out of bounds for axis 1 with size 14969
        print("LDA model and documents topic distribution saved")

    def build_bigrams_and_trigrams(self, data_words):
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # See trigram example
        print(trigram_mod[bigram_mod[data_words[0]]])

        # Form Bigrams
        data_words_bigrams = self.make_bigrams(bigram_mod, data_words)

        return data_words_bigrams

    def make_bigrams(self, bigram_mod, texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(self, trigram_mod, bigram_mod, texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def remove_stopwords(self, texts):
        stop_words = self.load_stopwords()
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def load_stopwords(self):
        filename = "cz_stemmer/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
            return cz_stopwords

    def visualise_lda(self, lda_model, corpus, dictionary, data_words_bigrams):

        print("Keywords and topics:")
        print(lda_model.print_topics())
        # Compute Perplexity
        print('\nLog perplexity: ', lda_model.log_perplexity(corpus))
        # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        pyLDAvis.display(vis_data)
        pyLDAvis.save_html(vis_data, 'C:\Dokumenty\OSU\Diplomová práce\LDA_Visualization.html')

    def display_lda_stats(self):
        self.database.connect()
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()
        self.database.disconnect()

        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))

        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)

        self.df['tokenized_full_text'] = self.df.apply(
            lambda row: row['body_preprocessed'].replace(str(row['tokenized']), ''),
            axis=1)

        gc.collect()
        self.df['tokenized_all_features_preprocessed'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))

        gc.collect()

        self.df['tokenized_full_text'] = self.df.tokenized_full_text.apply(lambda x: x.split(' '))
        print("self.df['tokenized']")

        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed'] + self.df[
            'tokenized_full_text']
        data = self.df
        print("data['tokenized']")
        print(data['tokenized'])
        data_words_nostops = self.remove_stopwords(data['tokenized'])
        data_words_bigrams = self.build_bigrams_and_trigrams(data_words_nostops)
        # Term Document Frequency

        # View
        dictionary = corpora.Dictionary(data_words_bigrams)
        # dictionary.filter_extremes(no_below=20, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in data_words_bigrams]

        t1 = time.time()
        print("corpus[:1]")
        print(corpus[:1])
        print("words:")
        print([[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]])
        # low alpha means each document is only represented by a small number of topics, and vice versa
        # low eta means each topic is only represented by a small number of words, and vice versa

        print("LDA loading...")
        lda_model = LdaModel.load("models/lda_model_full_text")
        t2 = time.time()
        print("Time to load LDA model on ", len(self.df), "articles: ", (t2 - t1) / 60, "min")

        self.visualise_lda(lda_model, corpus, dictionary, data_words_bigrams)

    # supporting function
    def compute_coherence_values(self, corpus, dictionary, data_lemmatized, num_topics, alpha, eta, passes, iterations):

        # Make sure that by the final passes, most of the documents have converged. So you want to choose both passes and iterations to be high enough for this to happen.
        # After choosing the right passes, you can set to None because it evaluates model perplexity and this takes too much time
        eval_every = 1

        lda_model = gensim.models.LdaModel(corpus=corpus,
                                               id2word=dictionary,
                                               num_topics=num_topics,
                                               random_state=100,
                                               chunksize=2000,
                                               passes=passes,
                                               alpha=alpha,
                                               eta=eta,
                                               eval_every=eval_every,
                                               # callbacks=[l],
                                               # added
                                               # workers=7,
                                               iterations=iterations)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary,
                                             coherence='c_v')

        return coherence_model_lda.get_coherence()

    def find_optimal_model(self, body_text_model=True):
        self.database.connect()
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()
        self.database.disconnect()

        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        if body_text_model is True:
            self.df['tokenized_full_text'] = self.df.apply(
                lambda row: row['body_preprocessed'].replace(str(row['tokenized']), ''),
                axis=1)
            self.df['tokenized_full_text'] = self.df.tokenized_full_text.apply(lambda x: x.split(' '))
        gc.collect()
        self.df['tokenized_all_features_preprocessed'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))

        gc.collect()

        if body_text_model is True:
            self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed'] + \
                                   self.df[
                                       'tokenized_full_text']
        else:
            self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed']

        data = self.df
        print("data['tokenized']")
        print(data['tokenized'])
        data_words_nostops = self.remove_stopwords(data['tokenized'])
        data_words_bigrams = self.build_bigrams_and_trigrams(data_words_nostops)
        data_lemmatized = data_words_bigrams
        dictionary = corpora.Dictionary(data_lemmatized)
        dictionary.filter_extremes(no_below=20, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in data_lemmatized]

        # Enabling LDA logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # Setting parameters
        limit = 1500
        start = 10
        step = 100
        # Topics range
        min_topics = 2
        max_topics = 3
        step_size = 1
        topics_range = range(min_topics, max_topics, step_size)
        # alpha = list(np.arange(0.01, 1, 0.5))
        alpha = []
        # alpha_params = ['symmetric','asymmetric','auto']
        alpha_params = ['auto']
        alpha.extend(alpha_params)
        eta = []
        # eta_params = ['symmetric','asymmetric','auto']
        eta_params = ['auto']
        eta.extend(eta_params)
        min_passes = 1
        max_passes = 2
        step_size = 1
        passes_range = range(min_passes, max_passes, step_size)
        min_iterations = 1
        max_iterations = 2
        step_size = 1
        iterations_range = range(min_iterations, max_iterations, step_size)
        num_of_docs = len(corpus)
        corpus_sets = [
            # gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.05)),
            # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
            gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
            corpus]
        corpus_title = ['75% Corpus', '100% Corpus']
        model_results = {'Validation_Set': [],
                         'Topics': [],
                         'Alpha': [],
                         'eta': [],
                         'Coherence': [],
                         'Passes': [],
                         'Iterations': []
                         }  # Can take a long time to run

        pbar = tqdm.tqdm(total=540)

        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                for p in passes_range:
                    for i in iterations_range:
                        # iterate through alpha values
                        for a in alpha:
                            # iterare through eta values
                            for e in eta:
                                # get the coherence score for the given parameters
                                print(alpha)
                                print(eta)
                                cv = self.compute_coherence_values(corpus=corpus_sets[i], dictionary=dictionary,
                                                                   data_lemmatized=data_lemmatized,
                                                                   num_topics=k, alpha=a, eta=e, passes=p, iterations=i)
                                # Save the model results
                                model_results['Validation_Set'].append(corpus_title[i])
                                model_results['Topics'].append(k)
                                model_results['Alpha'].append(a)
                                model_results['eta'].append(e)
                                model_results['Coherence'].append(cv)
                                model_results['Passes'].append(p)
                                model_results['Iterations'].append(i)

                                pbar.update(1)

        pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
        pbar.close()


    def format_topics_sentences(self, lda_model, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(lda_model[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = lda_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    # https: // www.machinelearningplus.com / nlp / topic - modeling - gensim - python /  # 17howtofindtheoptimalnumberoftopicsforlda
    def display_dominant_topics(self, optimal_model, corpus, texts):

        df_topic_sents_keywords = self.format_topics_sentences(lda_model=optimal_model, corpus=corpus, texts=texts)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        # Show dominant topics
        pd.set_option('display.max_rows', 1000)
        print("Dominant topics:")
        print(df_dominant_topic.head(10).to_string())

        # Group top 5 sentences under each topic
        sent_topics_sorteddf = pd.DataFrame()

        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf = pd.concat([sent_topics_sorteddf,
                                                     grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                                    axis=0)

        # Reset Index
        sent_topics_sorteddf.reset_index(drop=True, inplace=True)
        # Format
        sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
        # Show
        sent_topics_sorteddf.head()

        # Number of Documents for Each Topic
        topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts / topic_counts.sum(), 4)

        # Topic Number and Keywords
        topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

        # Concatenate Column wise
        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

        # Change Column names
        df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

        df_dominant_topics.to_csv("exports/dominant_topics.csv", sep=';', encoding='iso8859_2', errors='replace')
        print("Results saved to csv")


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
    # searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    # searched_slug = "facr-o-slavii-a-rangers-verime-v-objektivni-vysetreni-odmitame-rasismus"

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

    # tfidf = TfIdf()
    # print(tfidf.recommend_posts_by_all_features_preprocessed(searched_slug))
    # print(tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(searched_slug))

    # print(tfidf.recommend_posts_by_all_features('sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik'))
    # print(tfidf.preprocess("Vítkovice prohrály důležitý zápas s Třincem po prodloužení"))
    # print(tfidf.recommend_posts_by_all_features_preprocessed('sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik'))

    # keywords = "fotbal hokej sport slavia"
    # # print(tfidf.keyword_based_comparison(keywords))

    # doc2vecClass = Doc2VecClass()
    # print(doc2vecClass.get_similar_doc2vec(searched_slug,train=False))
    # print(doc2vecClass.get_similar_doc2vec_with_full_text(searched_slug,train=False))

    """
    lda = Lda()
    print("--------------LDA------------------")
    print(lda.get_similar_lda(searched_slug))
    print("--------------LDA FULL TEXT------------------")
    print(lda.get_similar_lda_full_text(searched_slug))
    """
    lda = Lda()
    # print(lda.get_similar_lda('salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem', train=True, display_dominant_topics=True))
    # print(lda.get_similar_lda_full_text('salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem', train=False, display_dominant_topics=False))
    # lda.display_lda_stats()
    lda.find_optimal_model(body_text_model=True)
    """
    word2vecClass = Word2VecClass()
    start = time.time()
    print(word2vecClass.get_similar_word2vec(searched_slug))
    end = time.time()
    print("Elapsed time: " + str(end - start))
    start = time.time()
    # print(word2vecClass.get_similar_word2vec_full_text(searched_slug))
    end = time.time()
    print("Elapsed time: " + str(end - start))
    # print(psutil.cpu_percent())
    # print(psutil.virtual_memory())  # physical memory usage
    # print('memory % used:', psutil.virtual_memory()[2])
    """
    # word2vec = Word2VecClass()
    # word2vec.prefilling_job(full_text=True, reverse=False, random=True)

    # word2vec = Word2VecClass()
    # word2vec.prefilling_job(full_text=True, reverse=False)
    """
    h = hpy()
    print(h.heap())
    """

if __name__ == "__main__": main()
