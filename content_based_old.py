import re
import time
import json

import nltk
from gensim import corpora
from gensim.models import LdaModel

from data_conenction import Database
import pandas as pd
import numpy as np
from scipy import sparse

from nltk import RegexpTokenizer, FreqDist, word_tokenize
import string
from text_classification.czech_stemmer import cz_stem
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, euclidean_distances
import majka
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer

"""
from memory_profiler import profile
import cProfile
import io
import pstats
import psutil
"""

import gc

nltk.download('punkt')

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

class TfIdfOld:


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

    def join_posts_ratings_categories(self):
        self.get_categories_dataframe()
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
             'all_features_preprocessed']]
        return self.df

    def recommend_posts_by_all_features_preprocessed(self, slug):

        tfidf = TfIdfOld()

        tfidf.get_posts_dataframe() # load posts to dataframe
        print("posts dataframe:")
        print(tfidf.get_posts_dataframe())
        print("posts categories:")
        print(tfidf.get_categories_dataframe())
        tfidf.get_categories_dataframe() # load categories to dataframe
        # tfidf.get_ratings_dataframe() # load post rating to dataframe

        tfidf.join_posts_ratings_categories() # joining posts and categories into one table
        print("posts ratings categories dataframe:")
        print(tfidf.join_posts_ratings_categories())

        # preprocessing

        # feature tuples of (document_id, token_id) and coefficient
        # fit_by_all_features_matrix = tfidf.get_fit_by_feature('all_features_preprocessed')
        fit_by_all_features_matrix = tfidf.get_fit_by_feature('all_features_preprocessed')
        fit_by_title = tfidf.get_fit_by_feature('title_y')

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_all_features_matrix,fit_by_title)
        print("tuple_of_fitted_matrices[0]")
        print(str(tuple_of_fitted_matrices[0]))
        post_recommendations = tfidf.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del tfidf
        return post_recommendations

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

    def set_cosine_sim_use_own_matrix(self,own_tfidf_matrix):
        own_tfidf_matrix_csr = sparse.csr_matrix(own_tfidf_matrix.astype(dtype=np.float16)).astype(dtype=np.float16)
        cosine_sim = self.cosine_similarity_n_space(own_tfidf_matrix_csr,own_tfidf_matrix_csr)
        # cosine_sim = cosine_similarity(own_tfidf_matrix_csr) # computing cosine similarity
        cosine_sim_df = pd.DataFrame(cosine_sim, index=self.df['slug_x'], columns=self.df['slug_x']) # finding original record of post belonging to slug
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

    def convert_datframe_posts_to_json(self, post_recommendations, slug):
        list_of_article_slugs = []
        list_of_coefficients = []

        for index, row in post_recommendations.iterrows():
            # finding coefficient belonging to recommended posts compared to original post (for which we want to find recommendations)
            list_of_coefficients.append(self.cosine_sim_df.at[row['slug'],slug])

        post_recommendations['coefficient'] = list_of_coefficients

        dict = post_recommendations.to_dict('records')

        list_of_article_slugs.append(dict.copy())
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs[0])
        return list_of_article_slugs[0]

    def get_fit_by_feature(self, feature_name, second_feature=None):
        fit_by_feature = self.get_tfIdfVectorizer(feature_name, second_feature)

        return fit_by_feature

    def find_post_by_slug(self, slug):
        return self.get_posts_dataframe().loc[self.get_posts_dataframe()['slug'] == slug]

    def get_tfIdfVectorizer(self, fit_by, fit_by_2=None, stemming=False):

        self.set_tfIdfVectorizer()

        # print("self.df[fit_by]")
        # print(self.df[fit_by])

        # preprocessing
        # self.df[fit_by] = self.df[fit_by].map(lambda s:self.preprocess(s))

        # # print("PREPROCESSING: self.df[fit_by]")
        # pd.set_option('display.max_columns', None)
        # # print(self.df[fit_by].to_string())

        if fit_by_2 is None:
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[fit_by])  # Metoda fit: výpočet průměru a rozptylu jednotlivých sloupců z dat. Metoda transformace: # transformuje všechny prvky pomocí příslušného průměru a rozptylu.
        else:
            self.df[fit_by] = self.df[fit_by_2] + ". " + self.df[fit_by]
            # # print(self.df[fit_by])
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[fit_by])  # Metoda fit: výpočet průměru a rozptylu jednotlivých sloupců z dat. Metoda transformace: # transformuje všechny prvky pomocí příslušného průměru a rozptylu.
        # print("Fitted by: " + str(fit_by) + " " + str(fit_by_2))
        # print(self.tfidf_tuples)
        # Outputing results:
        return self.tfidf_tuples # tuples of (document_id, token_id) and tf-idf score for it

    def set_tfIdfVectorizer(self):
        # load czech stopwords from file
        filename = "text_classification/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)

        tfidf_vectorizer = TfidfVectorizer(dtype=np.float32,stop_words=cz_stopwords) # transforms text to feature vectors that can be used as input to estimator
        self.tfidf_vectorizer = tfidf_vectorizer

    def get_posts_dataframe(self):
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'],inplace=True)
        return self.posts_df

    def get_categories_dataframe(self):
        self.categories_df = self.database.get_categories_dataframe(pd)
        return self.categories_df

    def join_post_ratings_categories(self,dataframe):
        self.get_categories_dataframe()
        dataframe = dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        dataframe = dataframe[['id_x','title_x','slug_x','excerpt','body','views','keywords','title_y','description']]
        # print("dataframe afer joining with category")
        # print(dataframe.iloc[0])
        return dataframe.iloc[0]

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
            edited_words = [cz_stem(w, True) for w in tokens]  # aggresive
            edited_words = list(filter(None, edited_words))  # empty strings removal
            return " ".join(edited_words)

        elif lemma is True:
            edited_words = [self.cz_lemma(w) for w in tokens]
            edited_words_list = list(filter(None, edited_words))  # empty strings removal
            return " ".join(edited_words_list)
        else:
            return tokens
        # print(lemma_words)

    def convert_df_to_json(self, dataframe):
        result = dataframe[["title", "excerpt", "body"]].to_json(orient="records", lines=True)
        parsed = json.loads(result)
        return parsed

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


class Doc2VecOld:

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
        self.df = self.df[['id_x','title_x','slug_x','excerpt','body','views','keywords','title_y','description','all_features_preprocessed']]


    def get_similar_doc2vec(self, slug, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        ## merge more columns!
        # cols = ["title_y","title_x","excerpt","keywords"]
        """
        self.df["title_y"] = self.df["title_y"]
        self.df["title_x"] = self.df["title_x"].map(lambda s: tfidf.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: tfidf.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"]
        """
        # cols = ["title_y","title_x","excerpt","keywords","slug_x"]

        cols = ["all_features_preprocessed"]
        documents_df = pd.DataFrame()
        documents_df['all_features_preprocessed'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)),
                                                                        axis=1)
        documents = list(map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))
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

        documents_df = pd.DataFrame(documents)
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
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(documents)]
        max_epochs = 100
        vec_size = 150
        size = 20
        alpha = 0.025
        minimum_alpha = 0.0025
        reduce_alpha = 0.0002
        model = Doc2Vec(vector_size=size,
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
        model.save("models/old_models/d2v.model")
        print("Model Saved")
        """

        model = Doc2Vec.load("models/old_models/d2v.model")
        # to find the vector of a document which is not in training data
        tfidf = TfIdfOld()
        # post_preprocessed = word_tokenize("Zemřel poslední krkonošský nosič Helmut Hofer, ikona Velké Úpy. Ve věku 88 let zemřel potomek slavného rodu vysokohorských nosičů Helmut Hofer z Velké Úpy. Byl posledním žijícím nosičem v Krkonoších, starodávným řemeslem se po staletí živili generace jeho předků. Jako nosič pracoval pro Českou boudu na Sněžce mezi lety 1948 až 1953.".lower())
        # post_preprocessed = tfidf.preprocess_single_post("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy")

        # not necessary
        post_preprocessed = tfidf.preprocess_single_post(slug)
        # print("post_preprocessed:")
        # print(post_preprocessed)
        post_features_to_find = post_preprocessed.iloc[0]['title_y'] + " " + post_preprocessed.iloc[0]['keywords'] + " " + \
                                post_preprocessed.iloc[0]['title_x'] + " " + post_preprocessed.iloc[0]['excerpt']
        # post_features_to_find = post_preprocessed.iloc[0]['title']
        """
        print(post_features_to_find)
        print("post_features_to_find")
        """
        tokens = post_features_to_find.split()
        """
        print("tokens:")
        print(tokens)
        """
        vector = model.infer_vector(tokens)
        """
        print("vector:")
        print(vector)
        """
        most_similar = model.docvecs.most_similar([vector], topn=number_of_recommended_posts)
        """
        print("most_similar:")
        print(most_similar)
        print(self.get_similar_posts_slug(most_similar,documents_slugs,number_of_recommended_posts))
        """
        return self.get_similar_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)


    def get_similar_posts_slug(self, most_similar,documents_slugs,number_of_recommended_posts):
        print('\n')

        post_recommendations = pd.DataFrame()
        list_of_article_slugs = []
        list_of_coefficients = []

        most_similar = most_similar[1:number_of_recommended_posts]

        # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD-MOST', 2), ('FOURTH-MOST', 3), ('FIFTH-MOST', 4), ('MEDIAN', len(most_similar) // 2), ('LEAST', len(most_similar) - 1)]:
        for index in range(0,len(most_similar)):
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

    def flatten(self,t):
        return [item for sublist in t for item in sublist]

class LdaOld:

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
             'all_features_preprocessed']]
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

    # @profile
    def get_similar_lda(self, searched_slug, N=21):
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()

        """
        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))
        print("self.df['tokenized_keywords'][0]")
        print(self.df['tokenized_keywords'][0])
        """
        """
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed_stopwords_clear'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        """
        """
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        """
        # self.df['tokenized'] = self.df.all_features_preprocessed_stopwords_clear.apply(lambda x: x.split(' '))
        self.df['tokenized'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))
        print("self.df['tokenized']")
        # print(self.df['tokenized'].iloc[0])
        # self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized']
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

        # minimum_amount_of_words = 30

        # self.df = self.df[self.df['tokenized'].map(len) >= minimum_amount_of_words]
        # make sure all tokenized items are lists
        self.df = self.df[self.df['tokenized'].map(type) == list]
        self.df.reset_index(drop=True, inplace=True)
        # print("After cleaning and excluding short articles, the dataframe now has:", len(self.df), "articles")
        print("df head:")
        print(self.df.head)

        # self.train_lda(self.df)
        dictionary, corpus, lda = self.load_lda(self.df)
        print("dictionary, corpus, lda")

        # bow = dictionary.doc2bow(self.df.iloc[searched_doc_id, 11])
        # print("new_bow")
        # print(new_bow)
        # searched_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=bow)])
        # print("new_doc_distribution")
        # print(new_doc_distribution)

        searched_doc_id_list = self.df.index[self.df['slug_x'] == searched_slug].tolist()
        searched_doc_id = searched_doc_id_list[0]

        print("self.df.columns")
        print(self.df.columns)

        new_bow = dictionary.doc2bow(self.df.iloc[searched_doc_id, 10])
        new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])
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
        # new_bow = dictionary.doc2bow(self.df.iloc[searched_doc_id, 11])
        # new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])

        print("most_sim_ids, most_sim_coefficients")
        # most_sim_ids = self.get_most_similar_documents(new_doc_distribution, doc_topic_dist)[0]

        doc_topic_dist = np.load('precalc_vectors/lda_old_doc_topic_dist.npy')

        most_sim_ids, most_sim_coefficients = self.get_most_similar_documents(new_doc_distribution, doc_topic_dist, N)

        print("most_sim_ids")
        print(most_sim_ids)
        # print("most_sim_coefficients")
        # print(most_sim_coefficients)
        """
        print("len(most_sim_coefficients)")
        print(len(most_sim_coefficients))
        """
        print("most_similar_df")
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
        """
        print("post_recommendations")
        print(post_recommendations)
        print("most_sim_coefficients:")
        print(most_sim_coefficients)
        """
        post_recommendations['coefficient'] = most_sim_coefficients[:N-1]
        """
        for index, row in post_recommendations.iterrows():
            # finding coefficient belonging to recommended posts compared to original post (for which we want to find recommendations)
            list_of_coefficients.append(self.cosine_sim_df.at[row['slug'], slug])
        """
        # post_recommendations['coefficient'] = list_of_coefficients
        # post_recommendations = post_recommendations.drop(post_recommendations.index[post_recommendations['slug'] == searched_slug])
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

    def load_lda(self, data):
        print("dictionary")
        dictionary = corpora.Dictionary(data['tokenized'])
        print("corpus")
        corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
        # lda_model_local = pickle.load(smart_open.smart_open(self.amazon_bucket_url))
        print("LdaModel.load")
        lda_model = LdaModel.load("models/old_models/lda_old_model")
        return dictionary, corpus, lda_model

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
                             chunksize=chunksize, alpha=1e-2, eta=0.5e-2,
                             passes=2)
        t2 = time.time()
        print("Time to train LDA model on ", len(self.df), "articles: ", (t2 - t1) / 60, "min")

        # native gensim method (abandoned due to not storing to single file like it should with separately=[] option)
        lda_model.save("models/old_models/lda_old_model")
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
        # native gensim method (abandoned due to not storing to single file like it should with separately=[] option)
        lda_model = LdaModel.load("models/old_models/lda_old_model")
        # lda_model_local = pickle.load(smart_open.smart_open("lda_all_in_one"))
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()

        documents_df = pd.DataFrame()
        # print("documents_df:")
        # print(documents_df)
        # documents_df["all_features_preprocessed"] = self.df["all_features_preprocessed"]
        # print("cz_stopwords:")
        # print(cz_stopwords)
        """
        cz_stopwords = self.load_stopwords()
        self.df['all_features_preprocessed_stopwords_clear'] = self.df['all_features_preprocessed'].apply(
            lambda x: ' '.join([item for item in x.split() if item not in cz_stopwords]))

        print("self.df['all_features_preprocessed_stopwords_clear']")
        print(self.df['all_features_preprocessed_stopwords_clear'])
        print("Removing words that already appears in keywords")
        print("self.df['keywords']")
        print(self.df['keywords'])
        """
        self.df['tokenized_keywords'] = self.df['keywords'].apply(lambda x: x.split(', '))
        print("self.df['tokenized_keywords'][0]")
        print(self.df['tokenized_keywords'][0])
        """
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed_stopwords_clear'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        """

        # self.df['tokenized'] = self.df.all_features_preprocessed_stopwords_clear.apply(lambda x: x.split(' '))
        self.df['tokenized'] = self.df.all_features_preprocessed.apply(lambda x: x.split(' '))
        print("self.df['tokenized']")
        # print(self.df['tokenized'].iloc[0])
        # self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized']
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

        """
        # plot a histogram of document length
        num_bins = 1000
        fig, ax = plt.subplots(figsize=(12, 6));
        # the histogram of the data
        n, bins, patches = ax.hist(doc_lengths, num_bins, density=True)
        ax.set_xlabel('Document Length (tokens)', fontsize=15)
        ax.set_ylabel('Normed Frequency', fontsize=15)
        ax.grid()
        ax.set_xticks(np.logspace(start=np.log10(50), stop=np.log10(2000), num=8, base=10.0))
        plt.xlim(0, 2000)
        ax.plot([np.average(doc_lengths) for i in np.linspace(0.0, 0.0035, 100)], np.linspace(0.0, 0.0035, 100), '-',
                label='average doc length')
        ax.legend()
        ax.grid()
        fig.tight_layout()
        plt.show()

        print("length of list:", len(doc_lengths),
              "\naverage document length", np.average(doc_lengths),
              "\nminimum document length", min(doc_lengths),
              "\nmaximum document length", max(doc_lengths))
        """
        """
        minimum_amount_of_words = 30

        self.df = self.df[self.df['tokenized'].map(len) >= minimum_amount_of_words]
        # make sure all tokenized items are lists
        """
        self.df = self.df[self.df['tokenized'].map(type) == list]
        self.df.reset_index(drop=True, inplace=True)
        print("After cleaning and excluding short aticles, the dataframe now has:", len(self.df), "articles")
        print("df head:")
        print(self.df.head)

        """
        # create a mask of binary values
        msk = np.random.rand(len(self.df)) < 0.999
        print("msk")
        print(msk)

        train_df = self.df[msk]
        print("train_df")
        print(train_df)
        train_df.reset_index(drop=True, inplace=True)

        test_df = self.df[~msk]
        print("Test dataframes:")
        print(test_df['title_x'])
        test_df.reset_index(drop=True, inplace=True)
        print("After reseting the index:")
        print(test_df['title_x'])
        print(len(self.df), len(train_df), len(test_df))
        """
        dictionary, corpus, lda = self.load_lda(self.df)

        # print("Most common topics found:")
        # print(lda.show_topics(num_topics=10, num_words=20))
        # print(lda.show_topic(topicid=4, topn=20))

        print("doc_topic_dist")
        doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])

        # save doc_topic_dist
        # https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        np.save('precalc_vectors/lda_old_doc_topic_dist.npy', doc_topic_dist)
        print("LDA model and documents topic distribution saved")

    def load_stopwords(self):
        filename = "text_classification/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
            return cz_stopwords

class Word2VecOld:

    def __init__(self):
        self.documents = None
        self.df = None
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
        self.get_categories_dataframe()
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
             'all_features_preprocessed']]
        return self.df

    def get_similar_posts_slug(self, most_similar,documents_slugs,number_of_recommended_posts):
        print('\n')

        post_recommendations = pd.DataFrame()
        list_of_article_slugs = []
        list_of_coefficients = []

        most_similar = most_similar[1:number_of_recommended_posts]

        # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD-MOST', 2), ('FOURTH-MOST', 3), ('FIFTH-MOST', 4), ('MEDIAN', len(most_similar) // 2), ('LEAST', len(most_similar) - 1)]:
        for index in range(0,len(most_similar)):
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

    def get_similar_word2vec(self, searched_slug):
        self.get_categories_dataframe()
        self.get_posts_dataframe()
        self.join_posts_ratings_categories()
        self.documents = list(map(' '.join, self.df[["all_features_preprocessed"]].values.tolist()))

        documents_df = pd.DataFrame(self.documents, columns=['documents'])
        del self.documents
        gc.collect()
        documents_df['slug'] = self.df['slug_x']

        filename = "text_classification/czech_stopwords.txt"

        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

        """documents_df['documents_cleaned'] = documents_df.documents.apply(lambda x: ' '.join([item for item in x.split() if item not in cz_stopwords]))
        # removing special characters and stop words from the text
        documents_df['documents_cleaned'] = documents_df.documents.apply(lambda x: " ".join(
            re.sub(r'[^a-zěščřžýáíéĚŠČŘŽÝÁÍÉA-Z]+', ' ', w, flags=re.UNICODE).lower()
            for w in x.split() if re.sub(r'[^a-zA-Z]+', ' ', w, flags=re.UNICODE).lower() not in cz_stopwords))
        """
        documents_df['documents_cleaned'] = documents_df['documents']

        tfidfvectoriser = TfidfVectorizer(dtype=np.float32)
        tfidfvectoriser.fit(documents_df.documents_cleaned)
        tfidf_vectors = tfidfvectoriser.transform(documents_df.documents_cleaned)

        pairwise_similarities = np.dot(tfidf_vectors, tfidf_vectors.T).toarray()
        pairwise_differences = euclidean_distances(tfidf_vectors)

        # self.most_similar(pairwise_similarities, 'Cosine Similarity', documents_df,0)
        # self.most_similar(pairwise_differences, 'Euclidean Distance', documents_df,0)

        # print(tfidf_vectors[0].toarray())
        # print(pairwise_similarities.shape)
        # print(pairwise_similarities[0][:])

         # documents similar to the first document in the corpus
        return self.most_similar(pairwise_similarities, 'Cosine Similarity', documents_df, searched_slug)

    def most_similar(self, similarity_matrix, matrix_type, documents_df, doc_slug, N=21):
        # documents_df.set_index('slug')
        searched_doc_id_list = documents_df.index[documents_df['slug'] == doc_slug].tolist()
        searched_doc_id = searched_doc_id_list[0]
        """
        print(f'Document: {documents_df.iloc[searched_doc_id]["documents"]}')
        print('\n')
        print("similarity_matrix:")
        print(similarity_matrix)
        print('Similar Documents:')
        """
        list_of_article_slugs = []
        list_of_coefficients = []

        if matrix_type == 'Cosine Similarity':
            similar_ix = np.argsort(similarity_matrix[searched_doc_id])[::-1]
        elif matrix_type == 'Euclidean Distance':
            similar_ix = np.argsort(similarity_matrix[searched_doc_id])
        for ix in similar_ix[:N]:
            if ix == searched_doc_id:
                continue
            print('\n')
            print(f'Document: {documents_df.iloc[ix]["documents"]}')
            print(f'{matrix_type} : {similarity_matrix[searched_doc_id][ix]}')
            list_of_article_slugs.append(documents_df.iloc[ix]["slug"])
            list_of_coefficients.append(similarity_matrix[searched_doc_id][ix])


        post_recommendations = pd.DataFrame()
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

    def flatten(self,t):
        return [item for sublist in t for item in sublist]

def main():
    #tfidf = TfIdfOld()
    #print(tfidf.recommend_posts_by_all_features_preprocessed("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy", json=True))

    searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    #doc2vecClass = Doc2VecOld()
    #print(doc2vecClass.get_similar_doc2vec(searched_slug))

    lda = LdaOld()
    print(lda.get_similar_lda(searched_slug))

    #searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    #word2vecOld = Word2VecOld()
    #print(word2vecOld.get_similar_word2vec(searched_slug))


if __name__ == "__main__": main()
