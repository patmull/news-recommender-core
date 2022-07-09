import csv
import gc
import logging
import os
import pickle
import time

import pyLDAvis
import regex
import tqdm
from gensim.corpora import WikiCorpus
from gensim.utils import deaccent
from nltk import FreqDist
from pyLDAvis import gensim_models as gensimvis
import content_based_algorithms.data_queries as data_queries
from content_based_algorithms.data_queries import RecommenderMethods

import gensim
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from scipy.stats import entropy

from content_based_algorithms.helper import Helper
from data_connection import Database
import preprocessing.cz_preprocessing
from preprocessing import cz_preprocessing
from preprocessing.cz_preprocessing import cz_stopwords

class Lda:
    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/lda_all_in_one"

    def __init__(self):
        self.documents = None
        self.df = None
        self.posts_df = None
        self.categories_df = None
        self.database = Database()

    @DeprecationWarning
    def join_posts_ratings_categories(self):
        self.get_categories_dataframe()
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
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

        gc.collect()

        print(self.df.head(10).to_string())
        # if there is no LDA model, training will run anyway due to load_texts method handle
        if train is True:
            self.train_lda(self.df, display_dominant_topics=display_dominant_topics)

        dictionary, corpus, lda = self.load_lda(self.df)

        searched_doc_id_list = self.df.index[self.df['post_slug'] == searched_slug].tolist()
        searched_doc_id = searched_doc_id_list[0]
        print("self.df.iloc[searched_doc_id]")
        selected_by_index = self.df.iloc[searched_doc_id]
        selected_by_column = selected_by_index['all_features_preprocessed']
        new_bow = dictionary.doc2bow([selected_by_column])
        new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])

        doc_topic_dist = np.load('precalc_vectors/lda_doc_topic_dist.npy')

        most_sim_ids, most_sim_coefficients = self.get_most_similar_documents(new_doc_distribution, doc_topic_dist, N)

        most_similar_df = self.df.iloc[most_sim_ids]

        most_similar_df = most_similar_df.iloc[1:, :]

        post_recommendations = pd.DataFrame()
        post_recommendations['slug'] = most_similar_df['slug'].iloc[:N]
        # post_recommendations['coefficient'] = most_sim_coefficients[:N - 1]
        if N == 21:
            post_recommendations['coefficient'] = most_sim_coefficients[:N-1]
        else:
            post_recommendations['coefficient'] = pd.Series(most_sim_coefficients[:N-1])

        """
        try:
            post_recommendations['coefficient'] = most_sim_coefficients[:N - 1]
        except ValueError:
            print("Value error. Going with older version of LDA model (not updated for all articles).")
            self.get_similar_lda(searched_slug, train=False, display_dominant_topics=False, N=number_of_df_rows)
            post_recommendations['coefficient'] = most_sim_coefficients[:N]
        """

        del self.df
        gc.collect()

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

        print("get_similar_lda_full_text dataframe:")
        print(self.df.to_string())

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

        searched_doc_id_list = self.df.index[self.df['slug'] == searched_slug].tolist()
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
        post_recommendations['slug'] = most_similar_df['slug'].iloc[:N]
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

            lda_model = LdaModel.load_texts("models/lda_model")
        """
        try:
            lda_model = LdaModel.load("models/lda_model")
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary.gensim')
            corpus = pickle.load(open('precalc_vectors/corpus.pkl', 'rb'))
        except Exception as e:
            print("Could not load_texts LDA models or precalculated vectors. Reason:")
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
            lda_model = LdaModel.load_texts("models/lda_model_full_text")
        except FileNotFoundError:
            print("Downloading LDA model files...")
            dropbox_access_token = "njfHaiDhqfIAAAAAAAAAAX_9zCacCLdpxxXNThA69dVhAsqAa_EwzDUyH1ZHt5tY"
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text", "/lda_model_full_text")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.expElogeta.npy", "/lda_model_full_text.expElogeta.npy")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.id2word", "/lda_model_full_text.id2word")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.state", "/lda_model_full_text.state")
            dropbox_file_download(dropbox_access_token, "models/lda_model_full_text.state.sstats.npy", "/lda_model_full_text.state.sstats.npy")
            print("LDA Model files downloaded")

            lda_model = LdaModel.load_texts("models/lda_model_full_text")
        """
        try:
            lda_model = LdaModel.load("models/lda_model_full_text")
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_full_text.gensim')
            corpus = pickle.load(open('precalc_vectors/corpus_full_text.pkl', 'rb'))
        except Exception as e:
            print("Could not load_texts LDA models or precalculated vectors. Reason:")
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

    def train_lda(self, data, display_dominant_topics=False):
        data_words_nostops = data_queries.remove_stopwords(data['tokenized'])
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
        # low eta means
        # each topic is only represented by a small number of words, and vice versa

        print("LDA training...")
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, minimum_probability=0.0,
                             chunksize=chunksize, eta='auto', alpha='auto',
                             passes=passes)
        t2 = time.time()
        print("Time to train LDA model on ", len(self.df), "articles: ", (t2 - t1) / 60, "min")

        # native gensim method (abandoned due to not storing to single file like it should with separately=[] option)
        lda_model.save("models/lda_model")
        # pickle.dump(lda_model_local, open("lda_all_in_one", "wb"))
        print("Model Saved")
        # lda_model = LdaModel.load_texts("models/lda_model")
        # lda_model_local = pickle.load_texts(smart_open.smart_open("lda_all_in_one"))
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

        top_k_words, _ = RecommenderMethods.most_common_words(all_words)
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
        passes = 20 # evaluated on 20
        # workers = 7  # change when used LdaMulticore on different computer/server according tu no. of CPU cores
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
        pyLDAvis.save_html(vis_data, 'research\LDA\LDA_Visualization.html')

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
        data_words_nostops = data_queries.remove_stopwords(data['tokenized'])
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
        print("Time to load_texts LDA model on ", len(self.df), "articles: ", (t2 - t1) / 60, "min")

        self.visualise_lda(lda_model, corpus, dictionary, data_words_bigrams)

    # supporting function
    def compute_coherence_values(self, corpus, dictionary, num_topics, alpha, eta, passes, iterations, data_lemmatized=None):

        # Make sure that by the final passes, most of the documents have converged. So you want to choose both passes and iterations to be high enough for this to happen.
        # After choosing the right passes, you can set to None because it evaluates model perplexity and this takes too much time
        eval_every = 1

        lda_model = gensim.models.LdaMulticore(corpus=corpus,
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
                                                             workers=2,
                                                             iterations=iterations)

        print("mm")
        print(corpus)

        """
        lda_model.save("full_models/cswiki/lda/lda_model")
        lda_model_loaded = LdaMulticore.load_texts("full_models/cswiki/lda/lda_model")
        
        print(lda_model.print_topics(20))

        meta_file = open("full_models/cswiki/cswiki_bow.mm.metadata.cpickle", 'rb')
        docno2metadata = pickle.load_texts(meta_file)
        meta_file.close()

        doc_num = 0
        print("Title: {}".format(docno2metadata[doc_num][1]))  # take the first article as an example
        vec = corpus[doc_num]  # get tf-idf vector
        print("lda.get_document_topics(vec)")
        print(lda_model_loaded.get_document_topics(vec))
        """
        #  For ‘u_mass’ corpus should be provided, if texts is provided, it will be converted to corpus using the dictionary. For ‘c_v’, ‘c_uci’ and ‘c_npmi’ texts should be provided
        if data_lemmatized is None:
            coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary,coherence='u_mass')
        else:
            coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary, coherence='c_v')

        return coherence_model_lda.get_coherence()

    def find_optimal_model(self, body_text_model=True):

        # Enabling LDA logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='content_based_algorithms/training_logs/lda/logs.log')
        self.preprocess_wiki_corpus()
        path_to_preprocessed_files = "full_models/cswiki/lda/preprocessed/"

        list_of_preprocessed_files = []
        for i in os.listdir(path_to_preprocessed_files):
            if os.path.isfile(os.path.join(path_to_preprocessed_files, i)) and 'articles_' in i:
                list_of_preprocessed_files.append(i)

        list_of_preprocessed_files = [path_to_preprocessed_files + s for s in list_of_preprocessed_files]
        print("Loading data from files:")
        print(list_of_preprocessed_files)
        print("Loading preprocessed corpus...")
        processed_data = self.load_preprocessed_corpus(list_of_preprocessed_files)
        print("Loaded " + str(len(processed_data)) + " documents.")
        print("Saving corpus into single file...")

        single_file_name = "full_models/cswiki/lda/preprocessed/articles_" + str(len(processed_data))
        with open(single_file_name, 'wb') as f:
            print("Saving list to " + single_file_name)
            pickle.dump(processed_data, f)

        print("Removing stopwords...")
        data_words_nostops = data_queries.remove_stopwords(processed_data)
        print("Building bigrams...")
        processed_data = self.build_bigrams_and_trigrams(data_words_nostops)
        print("Creating dictionary...")
        print("TOP WORDS (after bigrams and stopwords removal):")
        top_k_words, _ = RecommenderMethods.most_common_words(processed_data)
        print(top_k_words)
        preprocessed_dictionary = corpora.Dictionary(processed_data)
        print("Saving dictionary...")
        preprocessed_dictionary.save("full_models/cswiki/lda/preprocessed/dictionary")
        print("Translating words into Doc2Bow vectors")
        preprocessed_corpus = [preprocessed_dictionary.doc2bow(token, allow_update=True) for token in processed_data]
        print("Piece of preprocessed_corpus:")
        print(preprocessed_corpus[:1])

        limit = 1500
        start = 10
        step = 100
        # Topics range
        """
        min_topics = 2
        max_topics = 3
        step_size = 1
        topics_range = range(min_topics, max_topics, step_size)
        """
        # topics_range = [20,40,60,80,100,200,300,400,500,600,700,800,900]
        topics_range = [40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        # alpha = list(np.arange(0.01, 1, 0.5))
        alpha = []
        # alpha_params = ['symmetric','asymmetric','auto']
        # alpha_params = ['symmetric','asymmetric']
        alpha_params = ['asymmetric']
        alpha.extend(alpha_params)
        eta = []
        # eta_params = ['symmetric','asymmetric','auto']
        eta_params = ['symmetric', 'auto']
        eta.extend(eta_params)
        min_passes = 20
        max_passes = 20
        step_size = 1
        # passes_range = range(min_passes, max_passes, step_size)
        passes_range = [20]
        min_iterations = 50
        max_iterations = 50
        step_size = 1
        # iterations_range = range(min_iterations, max_iterations, step_size)
        iterations_range = [50]
        num_of_docs = len(preprocessed_corpus)
        corpus_sets = [
            # gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.05)),
            # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
            # gensim.utils.ClippedCorpus(preprocessed_corpus, int(num_of_docs * 0.75)),
            gensim.utils.ClippedCorpus(preprocessed_corpus, int(num_of_docs * 1.00)),
            preprocessed_corpus]
        corpus_title = ['100% Corpus']
        model_results = {'Validation_Set': [],
                         'Topics': [],
                         'Alpha': [],
                         'Eta': [],
                         'Coherence': [],
                         'Passes': [],
                         'Iterations': []
                         }  # Can take a long time to run

        pbar = tqdm.tqdm(total=540)
        print("----------------------")
        print("Testing model on:")
        print("-----------------------")
        print("Topics:")
        print(topics_range)
        print("Passes range")
        print(passes_range)
        print("Iterations:")
        print(iterations_range)
        print("Alpha:")
        print(alpha)
        print("Eta:")
        print(eta)
        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                for p in passes_range:
                    for iterations in iterations_range:
                        # iterate through alpha values
                        for a in alpha:
                            # iterare through eta values
                            for e in eta:
                                # get the coherence score for the given parameters
                                cv = self.compute_coherence_values(corpus=corpus_sets[i], dictionary=preprocessed_dictionary,
                                                                   # data_lemmatized=data_lemmatized,
                                                                   num_topics=k, alpha=a, eta=e, passes=p, iterations=iterations, data_lemmatized=processed_data)
                                # Save the model results
                                model_results['Validation_Set'].append(corpus_title[i])
                                model_results['Topics'].append(k)
                                model_results['Alpha'].append(a)
                                model_results['Eta'].append(e)
                                model_results['Coherence'].append(cv)
                                model_results['Passes'].append(p)
                                model_results['Iterations'].append(iterations)

                                pbar.update(1)
                                pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False, mode="a")
                                print("Saved training results...")
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
        print(self.df.head(10).to_string())
        df_dominant_topic_merged = df_dominant_topic.merge(self.df, how='outer', left_index=True, right_index=True)
        print("After join")
        df_dominant_topic_filtered_columns = df_dominant_topic_merged[['Document_No', 'slug', 'post_title', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']]
        print(df_dominant_topic_filtered_columns.head(10).to_string())
        # saving dominant topics with corresponding documents
        df_dominant_topic_filtered_columns.to_csv("exports/dominant_topics_and_documents.csv", sep=';', encoding='iso8859_2', errors='replace')

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

    def preprocess_wiki_corpus(self):

        corpus = WikiCorpus('full_models/cswiki-20220301-pages-articles-multistream.xml.bz2', dictionary=False)
        id2word = gensim.corpora.Dictionary.load_from_text('full_models/cswiki/cswiki_wordids.txt.bz2')
        # corpus = gensim.corpora.MmCorpus('full_models/cswiki/cswiki_tfidf.mm')
        # corpus = [dct.doc2bow(line) for line in dataset]

        # preprocessing steps
        czlemma = cz_preprocessing.CzPreprocess()
        helper = Helper()
        processed_data = []
        # takes very long time

        # find files starting with "articles_"
        """
        for article_file in os.listdir("full_models/cswiki/lda/preprocessed"):
            if article_file.startswith("articles_"):
                list_of_preprocessed_files.append(article_file)
        """

        list_of_preprocessed_files = []
        path_to_preprocessed_files = "full_models/cswiki/lda/preprocessed/"
        for article_file in os.listdir("full_models/cswiki/lda/preprocessed"):
            if article_file.startswith("articles_"):
                list_of_preprocessed_files.append(article_file)
        list_of_preprocessed_files = [path_to_preprocessed_files + s for s in list_of_preprocessed_files]
        print("Loading preprocessed corpus...")
        number_of_documents = 0
        if len(list_of_preprocessed_files) > 0:
            processed_data = self.load_preprocessed_corpus(list_of_preprocessed_files)

            number_of_documents = len(processed_data)

            print("Loaded " + str(number_of_documents) + " documents.")
            print("Saving corpus into single file...")
            single_file_name = "full_models/cswiki/lda/preprocessed/articles_" + str(number_of_documents)
            with open(single_file_name, 'wb') as f:
                print("Saving list to " + single_file_name)
                pickle.dump(processed_data, f)

            print("Saving preprocessed articles to csv")
            self.save_list_to_csv(processed_data)

            print("Starting another preprocessing from document where it was halted.")
        else:
            print("No file with preprocessed articles was found. Starting from 0.")
            number_of_documents = 0

        i = 0
        num_of_preprocessed_docs = number_of_documents
        num_of_iterations_until_saving = 100 # Saving file every 100nd document
        path_to_save_list = "full_models/cswiki/lda/preprocessed/articles_newest"
        processed_data = []
        for doc in helper.generate_lines_from_corpus(corpus):
            if number_of_documents > 0:
                number_of_documents -= 1
                print("Skipping doc.")
                print(doc[:10])
                continue
            print("Processing doc. num. " + str(num_of_preprocessed_docs))
            print("Before:")
            print(doc)
            tokens = deaccent(czlemma.preprocess(doc))

            # removing words in greek, azbuka or arabian
            # use only one of the following lines, whichever you prefer
            tokens = [i for i in tokens.split() if regex.sub(r'[^\p{Latin}]',u'',i)]
            processed_data.append(tokens)
            print("After:")
            print(tokens)
            i = i + 1
            num_of_preprocessed_docs = num_of_preprocessed_docs + 1
            # saving list to pickle evey 100th document
            if i > num_of_iterations_until_saving:
                with open(path_to_save_list, 'wb') as f:
                    print("Saving list to " + path_to_save_list)
                    pickle.dump(processed_data, f)
                i = 0
        print("Preprocessing Wikipedia has (finally) ended. All articles were preprocessed.")

    def load_preprocessed_corpus(self, list_of_preprocessed_files):
        preprocessed_data_from_pickles = []
        for file_path in list_of_preprocessed_files:
            if os.path.getsize(file_path) > 0:
                try:
                    with open(file_path, 'rb') as f:
                        preprocessed_data_from_pickles.extend(pickle.load(f))
                    print("Opened file:")
                    print(file_path)
                except EOFError:
                    print("Can't load_texts file")
                    print(file_path)
        print("Example of 100th loaded document:")
        print(preprocessed_data_from_pickles[100:101])
        top_k_words, _ = RecommenderMethods.most_common_words([item for sublist in preprocessed_data_from_pickles for item in sublist])
        print("TOP WORDS:")
        print(top_k_words[:500])
        return preprocessed_data_from_pickles

    def save_list_to_csv(self, list_to_save, pandas=False):
        print("Saving to CSV...")
        if pandas is True:
            my_df = pd.DataFrame(list_to_save)
            my_df.to_csv('full_models/cswiki/lda/preprocessed/preprocessed_articles.csv', index=False, header=False)
        else:
            with open("full_models/cswiki/lda/preprocessed/preprocessed_articles.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(list_to_save)

    def save_list_to_txt(self, list_to_save):
        with open('full_models/cswiki/lda/preprocessed/preprocessed_articles.txt', 'w') as f:
            for item in list_to_save:
                f.write("%s\n" % item)
