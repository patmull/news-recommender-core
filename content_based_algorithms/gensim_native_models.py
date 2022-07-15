import re
import string
from collections import defaultdict
from pathlib import Path

import pandas as pd
from nltk import RegexpTokenizer

from content_based_algorithms.data_queries import RecommenderMethods
from preprocessing.cz_preprocessing import CzPreprocess
from preprocessing.czech_stemmer import cz_stem
from gensim import corpora, models, similarities
from scipy import sparse

from data_connection import Database


class GenSimMethods:

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
            ['id_x', 'post_title', 'slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description']]

    def find_post_by_slug(self, slug):
        post_dataframe = self.df.loc[self.df['post_slug'] == slug]
        doc = post_dataframe["category_title"] + " " + post_dataframe["keywords"] + " " + post_dataframe["post_title"] + " " + \
              post_dataframe["excerpt"]
        return str(doc.tolist())
        # return self.get_posts_dataframe().loc[self.get_posts_dataframe()['slug'] == slug]
        # return str(self.df['category_title','post_title','excerpt','keywords','slug'].loc[self.df['slug'] == slug].values[0])

    def get_categories_dataframe(self):
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe(pd)
        self.database.disconnect()
        return self.categories_df

    def recommend_posts_by_all_features(self, slug):

        gensimClass = GenSimMethods()

        gensimClass.get_posts_dataframe()  # load_texts posts do dataframe
        gensimClass.get_categories_dataframe()  # load_texts categories to dataframe
        # tfidf.get_ratings_dataframe() # load_texts post rating do dataframe
        recommendeMethods = RecommenderMethods()
        self.df = recommendeMethods.join_posts_ratings_categories()  # joining posts and categories into one table
        fit_by_title_matrix = gensimClass.get_fit_by_feature('post_title', 'category_title')  # prepended by category
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
                                                  self.df[['slug']])
        # print("combined_all:")
        # print(combined_all)
        df_renamed = combined_all.rename(columns={'slug': 'slug'})
        # print("df_renamed:")
        # print(df_renamed)

        # json conversion
        json = self.convert_posts_to_json(df_renamed, slug)

        return json

    def load_texts(self):
        recommender_methods = RecommenderMethods()
        self.posts_df = recommender_methods.get_posts_dataframe()
        self.categories_df = recommender_methods.get_categories_dataframe()
        self.df = recommender_methods.join_posts_ratings_categories()
        # preprocessing
        # self.df["post_title"] = self.df["post_title"].map(lambda s: self.preprocess(s, stemming=False, lemma=False))
        # self.df["excerpt"] = self.df["excerpt"].map(lambda s: self.preprocess(s, stemming=False, lemma=False))

        # converting pandas columns to list of lists and through map to list of string joined by space ' '
        self.documents = list(map(' '.join, self.df[["keywords", "category_title", "post_title", "excerpt"]].values.tolist()))

        filename = "preprocessing/stopwords/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)
        texts = [
            [word for word in document.lower().split() if word not in cz_stopwords and len(word) > 1]
            for document in self.documents
        ]

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

        return texts

    @DeprecationWarning
    def get_recommended_by_slug(self, slug):

        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        texts = self.load_texts()

        # does it support czech language?
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

        doc = self.find_post_by_slug(slug)
        # post_dataframe["post_title"] = post_dataframe["post_title"].map(lambda s: self.preprocess(s))
        # post_dataframe["excerpt"] = post_dataframe["excerpt"].map(lambda s: self.preprocess(s))

        doc = ''.join(doc)
        doc = str(doc)

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
        cz_preprocess = CzPreprocess()
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
            edited_words = [cz_preprocess.cz_lemma(w) for w in tokens if len(w) > 1]
            edited_words_list = list(filter(None, edited_words))  # empty strings removal
            return " ".join(edited_words_list)
        else:
            return tokens
        # print(lemma_words)

    def preprocess_single_post(self, slug, json=False, stemming=False):
        recommenderMethods = RecommenderMethods()
        post_dataframe = self.find_post_by_slug(slug)
        post_dataframe["title"] = post_dataframe["title"].map(lambda s: self.preprocess(s, stemming))
        post_dataframe["excerpt"] = post_dataframe["excerpt"].map(lambda s: self.preprocess(s, stemming))
        if json is False:
            return post_dataframe
        else:
            return recommenderMethods.convert_df_to_json(post_dataframe)

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