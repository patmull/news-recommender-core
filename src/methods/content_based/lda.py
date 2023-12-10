import csv
import gc
import logging
import os
import pickle
import time
from pathlib import Path

import pyLDAvis
import regex
from gensim.corpora import WikiCorpus
from gensim.utils import deaccent, flatten
from nltk import FreqDist
from pyLDAvis import gensim_models as gensimvis

from src.prefillers.preprocessing.czech_preprocessing import preprocess
from src.recommender_core.data_handling.data_queries import RecommenderMethods

import gensim
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from scipy.stats import entropy

from src.recommender_core.recommender_algorithms.content_based_algorithms.helper import generate_lines_from_corpus
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from src.prefillers.preprocessing.stopwords_loading import remove_stopwords

import config.trials_counter

CORPUS_FULL_TEXT_PICKLE = 'precalc_vectors/topics/corpus_full_text.pkl'
DICTIONARY_FULL_TEXT_GENSIM = 'precalc_vectors/topics/dictionary_full_text.gensim'
LDA_FULL_TEXT_PATH = "models/lda_model_full_text"


def apply_tokenize(text):
    print("text")
    return text.split(" ")


def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire train_corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the train_corpus
    """
    # lets keep with the p,q notation above
    p = query[None, :].T  # take transpose
    q = matrix.T  # transpose matrix
    m = 0.5 * (p + q)
    return np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))


def load_lda():
    try:
        lda_model = LdaModel.load("models/lda_model")
        dictionary = gensim.corpora.Dictionary.load('precalc_vectors/topics/dictionary_idnes.gensim')
        corpus = pickle.load(open('precalc_vectors/topics/corpus_idnes.pkl', 'rb'))
    except FileNotFoundError as e:
        raise FileNotFoundError("Could not load_texts LDA models or precalculated vectors. Reason: ", e)

    return dictionary, corpus, lda_model


def save_corpus_dict(corpus, dictionary):
    # Saving train_corpus and dictionary...
    pickle.dump(corpus, open('precalc_vectors/topics/corpus_idnes.pkl', 'wb'))
    dictionary.save('precalc_vectors/topics/dictionary_idnes.gensim')


def save_corpus_dict_full_text(corpus, dictionary):
    # Saving train_corpus and dictionary...
    pickle.dump(corpus, open(CORPUS_FULL_TEXT_PICKLE, "wb"))
    dictionary.save(DICTIONARY_FULL_TEXT_GENSIM)


def make_bigrams(bigram_mod, texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(trigram_mod, bigram_mod, texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def visualise_lda(lda_model, corpus, dictionary):
    # Compute Perplexity
    # a measure of how good the model is. lower the better.

    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.display(vis_data)
    # noinspection PyPep8
    pyLDAvis.save_html(vis_data, 'stats/LDA/LDA_Visualization.html')


def compute_coherence_values(corpus, dictionary, num_topics, alpha, eta, passes, iterations, data_lemmatized=None):
    # Make sure that by the final passes, most of the documents have converged. So you want to choose both passes and
    # iterations to be high enough for this to happen. After choosing the right passes, you can set to None because
    # it evaluates model perplexity and this takes too much time
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
                                           workers=2,
                                           iterations=iterations)

    print("mm")
    print(corpus)

    # For ‘u_mass’ train_corpus should be provided, if text is provided,
    # it will be converted to train_corpus using
    # the dictionary. For ‘c_v’, ‘c_uci’ and ‘c_npmi’ texts should be provided
    if data_lemmatized is None:
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    else:
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary,
                                             coherence='c_v')

    return coherence_model_lda.get_coherence()


def format_topics_sentences(lda_model, corpus, texts):
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
    return sent_topics_df


def load_preprocessed_corpus(list_of_preprocessed_files):
    preprocessed_data_from_pickles = []
    for file_path in list_of_preprocessed_files:
        if os.path.getsize(file_path) > 0:
            try:
                with open(file_path, 'rb') as f:
                    preprocessed_data_from_pickles.extend(pickle.load(f))
            except EOFError:
                raise EOFError("Can'multi_dimensional_list load_texts file")

    return preprocessed_data_from_pickles


def save_list_to_csv(list_to_save, pandas=False):
    print("Saving to CSV...")
    if pandas is True:
        my_df = pd.DataFrame(list_to_save)
        # noinspection PyTypeChecker
        my_df.to_csv('full_models/cswiki/topics/preprocessed/preprocessed_articles.csv', index=False, header=False)
    else:
        with open("full_models/cswiki/topics/preprocessed/preprocessed_articles.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(list_to_save)


def save_list_to_txt(list_to_save):
    with open('full_models/cswiki/topics/preprocessed/preprocessed_articles.txt', 'w') as f:
        for item in list_to_save:
            f.write("%s\n" % item)


def get_most_similar_documents(query, matrix, k=20):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query, matrix)  # list of jensen shannon distances

    sorted_k_result = sims.argsort()[:k]
    sims = sorted(sims, reverse=True)

    return sorted_k_result, sims  # the top k positional index of the smallest Jensen Shannon distances


def build_ngrams(data_words):
    logging.debug('data_words:')
    logging.debug(data_words)
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    logging.debug('bigram')
    logging.debug(bigram)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # use gensim.models.phrases.Phraser(trigram) for trigrams

    # Form Bigrams
    data_words_bigrams = make_bigrams(bigram_mod, data_words)

    return data_words_bigrams


def preprocess_wiki_corpus():
    corpus = WikiCorpus('full_models/cswiki-20220301-pages-articles-multistream.xml.bz2', dictionary=False)

    # preprocessing steps
    list_of_preprocessed_files = []
    path_to_preprocessed_files = "full_models/cswiki/topics/preprocessed/"
    for article_file in os.listdir("full_models/cswiki/topics/preprocessed"):
        if article_file.startswith("articles_"):
            list_of_preprocessed_files.append(article_file)
    list_of_preprocessed_files = [path_to_preprocessed_files + s for s in list_of_preprocessed_files]
    if len(list_of_preprocessed_files) > 0:
        processed_data = load_preprocessed_corpus(list_of_preprocessed_files)
        number_of_documents = len(processed_data)

        # Saving train_corpus into single file...
        single_file_name = "full_models/cswiki/topics/preprocessed/articles_" + str(number_of_documents)
        with open(single_file_name, 'wb') as f:
            # "Saving list."
            pickle.dump(processed_data, f)

        # Saving preprocessed articles to csv.
        save_list_to_csv(processed_data)

        # Starting another preprocessing from document where it was halted.
    else:
        # No file with preprocessed articles was found. Starting from 0.
        number_of_documents = 0

    i = 0
    num_of_preprocessed_docs = number_of_documents
    num_of_iterations_until_saving = 100  # Saving file every 100nd document
    path_to_save_list = "full_models/cswiki/topics/preprocessed/articles_newest"
    processed_data = []
    for doc in generate_lines_from_corpus(corpus):
        if number_of_documents > 0:
            number_of_documents -= 1
            continue
        tokens = deaccent(preprocess(doc))

        # removing words in greek, azbuka or arabian
        # use only one of the following lines, whichever you prefer
        tokens = [i for i in tokens.split() if regex.sub(r'[^\p{Latin}]', u'', i)]
        processed_data.append(tokens)

        i = i + 1
        num_of_preprocessed_docs = num_of_preprocessed_docs + 1
        # saving list to pickle evey 100th document
        if i > num_of_iterations_until_saving:
            with open(path_to_save_list, 'wb') as f:
                pickle.dump(processed_data, f)
            i = 0


def prepare_post_categories_df(recommender_methods, posts_from_cache, searched_slug):
    recommender_methods.get_posts_dataframe(from_cache=posts_from_cache)
    df = recommender_methods.get_posts_categories_dataframe(from_cache=posts_from_cache)

    if searched_slug not in recommender_methods.df['slug'].to_list():
        print('Slug does not appear in dataframe.')
        recommender_methods = RecommenderMethods()
        recommender_methods.get_posts_dataframe(force_update=True)
        df = recommender_methods.get_posts_categories_dataframe(from_cache=True)

    return df


class Lda:
    # *** HERE WAS a Amazon bucket connection string. ABANDONED DUE TO: not being needed anymore

    def __init__(self):
        self.most_similar_df = None
        self.top_k_words = None
        self.documents = None
        self.df = pd.DataFrame()
        self.posts_df = pd.DataFrame()
        self.categories_df = pd.DataFrame()
        self.database = DatabaseMethods()

    def get_searched_doc_id(self, recommender_methods, searched_slug):
        logging.info("Finding id for post with slug:")
        logging.info(searched_slug)
        recommender_methods.df = recommender_methods.df.rename(columns={'post_slug': 'slug'})
        searched_doc_id_list = recommender_methods.df.index[recommender_methods.df['slug'] == searched_slug].tolist()
        logging.debug("searched_doc_id_list:")
        logging.debug(searched_doc_id_list)
        searched_doc_id = searched_doc_id_list[0]
        return searched_doc_id

    def get_similar_lda_full_text(self, searched_slug: str, n=21, train=False, display_dominant_topics=True,
                                  posts_from_cache=True):
        if type(searched_slug) is not str:
            raise TypeError("searched_slug needs to be a string!")
        if searched_slug == "":
            raise ValueError("Empty string inserted instead of slug string.")

        recommender_methods = RecommenderMethods()
        recommender_methods.df = prepare_post_categories_df(recommender_methods, posts_from_cache, searched_slug)

        logging.debug('recommender_methods.df')
        logging.debug(recommender_methods.df)

        recommender_methods.df['tokenized'] = recommender_methods.tokenize_text()
        gc.collect()

        if searched_slug not in recommender_methods.df['slug'].to_list():
            # ** HERE WAS A HANDLING OF THIS ERROR BY UPDATING POSTS_CATEGORIES DF. ABANDONED DUE TO MASKING OF ERROR
            # FOR BAD INPUT **
            raise ValueError("searched_slug not in dataframe")

        if train is True:
            self.train_lda_full_text(recommender_methods.df, display_dominant_topics=display_dominant_topics)

        _dictionary, _, lda = self.load_lda_full_text(recommender_methods.df,
                                                      display_dominant_topics=display_dominant_topics)

        searched_doc_id = self.get_searched_doc_id(recommender_methods, searched_slug)

        new_sentences = recommender_methods.df.iloc[searched_doc_id, :]
        new_sentences = new_sentences[['tokenized']]
        print("new_sentences[['tokenized']][0]:")
        print(new_sentences[['tokenized']][0])
        # .replace(' ', '_')
        new_sentences_split = [gensim.utils.deaccent(sentence).replace(' ', '_') for sentence in
                               new_sentences[['tokenized']][0]]
        print("new_sentences_split:")
        print(new_sentences_split)
        new_bow = _dictionary.doc2bow(new_sentences_split)
        new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])

        doc_topic_dist = np.load('precalc_vectors/topics/lda_doc_topic_dist_full_text.npy')

        most_sim_ids, most_sim_coefficients = get_most_similar_documents(new_doc_distribution, doc_topic_dist, n)
        try:
            self.most_similar_df = recommender_methods.df.iloc[most_sim_ids]
        except IndexError as e:
            if config.trials_counter.NUM_OF_TRIALS < 1:
                logging.warning('Index error occurred when trying to get Doc2Vec model for posts')
                logging.warning(e)
                logging.info('Trying to deal with this by retraining LDA...')
                logging.debug('Preparing test features')

                logging.debug('recommender_methods.df')
                logging.debug(recommender_methods.df)
                self.train_lda_full_text(recommender_methods.df, display_dominant_topics=display_dominant_topics)
                config.trials_counter.NUM_OF_TRIALS += 1
                self.get_similar_lda_full_text(searched_slug, n, train, display_dominant_topics, posts_from_cache)
            else:
                logging.warning(
                    "Tried to train Doc2Vec again but it didn't helped and IndexError got raised again. "
                    "Need to shutdown.")
                raise e
        del recommender_methods.df
        gc.collect()
        most_similar_df = self.most_similar_df.iloc[1:, :]
        post_recommendations = pd.DataFrame()
        post_recommendations['slug'] = most_similar_df['slug'].iloc[:n]
        post_recommendations['coefficient'] = most_sim_coefficients[:n - 1]

        posts_dict = post_recommendations.to_dict('records')
        list_of_articles = [posts_dict.copy()]

        return flatten(list_of_articles)

    def keep_top_k_words(self, text):
        return [word for word in text if word in self.top_k_words]

    def load_lda_full_text(self, data, display_dominant_topics):

        try:
            lda_model = LdaModel.load(LDA_FULL_TEXT_PATH)
            dictionary = gensim.corpora.Dictionary.load(DICTIONARY_FULL_TEXT_GENSIM)
            corpus = pickle.load(open(CORPUS_FULL_TEXT_PICKLE, 'rb'))
        except Exception:
            self.train_lda_full_text(data, display_dominant_topics)
            lda_model = LdaModel.load(LDA_FULL_TEXT_PATH)
            dictionary = gensim.corpora.Dictionary.load(DICTIONARY_FULL_TEXT_GENSIM)
            corpus = pickle.load(open(CORPUS_FULL_TEXT_PICKLE, 'rb'))

        return dictionary, corpus, lda_model

    def train_lda_full_text(self, data, display_dominant_topics=True):

        logging.debug(data['tokenized'])
        data_words_nostops = remove_stopwords(data['tokenized'].tolist())
        data_words_bigrams = build_ngrams(data_words_nostops)

        self.df = self.df.assign(tokenized=data_words_bigrams)

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

        save_corpus_dict(corpus, dictionary)

        num_topics = 20  # set according visualise_lda() model_variant (Coherence value) = 20
        chunksize = 1000
        passes = 20  # evaluated on 20
        # set workers when used LdaMulticore on different computer/server according tu no. of CPU cores
        eta = 'auto'
        iterations = 200
        logging.info("LDA training...")
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                             minimum_probability=0.0, chunksize=chunksize,
                             eta=eta, alpha='auto',
                             passes=passes, iterations=iterations)

        if "PYTEST_CURRENT_TEST" in os.environ:
            path_to_save = Path('tests/models/lda_testing.model')
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            lda_model.save(path_to_save.as_posix())
        else:
            lda_model.save(LDA_FULL_TEXT_PATH)

        if display_dominant_topics is True:
            self.display_dominant_topics(optimal_model=lda_model, corpus=corpus, texts=data_words_bigrams)

        if "PYTEST_CURRENT_TEST" in os.environ:
            path_to_save = Path('tests/models/lda_testing.model')
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            lda = lda_model.load(path_to_save.as_posix())
        else:
            lda = lda_model.load(LDA_FULL_TEXT_PATH)

        doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
        print("np.save")
        # save doc_topic_dist
        # https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        np.save('precalc_vectors/topics/lda_doc_topic_dist_full_text.npy',
                doc_topic_dist)  # IndexError: index 14969 is out of bounds for axis 1 with size 14969

    def display_lda_stats(self):
        recommender_methods = RecommenderMethods()
        recommender_methods.get_posts_dataframe()
        self.df = recommender_methods.get_posts_categories_dataframe()

        self.df['tokenized_keywords'] = recommender_methods.tokenize_text()
        # noinspection PyPep8
        self.df['tokenized'] = self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed'] \
                               + self.df['tokenized_full_text']
        data = self.df

        data_words_nostops = remove_stopwords(data['tokenized'])
        data_words_bigrams = build_ngrams(data_words_nostops)
        # Term Document Frequency

        dictionary = corpora.Dictionary(data_words_bigrams)
        corpus = [dictionary.doc2bow(doc) for doc in data_words_bigrams]

        t1 = time.time()

        # low alpha means each document is only represented by a small number of topics, and vice versa
        # low eta means each topic is only represented by a small number of words, and vice versa

        # LDA loading...
        lda_model = LdaModel.load(LDA_FULL_TEXT_PATH)
        t2 = time.time()
        logging.info(f"Time to load_texts LDA model on: {len(self.df)}, articles: {(t2 - t1) / 60} min")

        visualise_lda(lda_model, corpus, dictionary)

    # supporting function

    def display_dominant_topics(self, optimal_model, corpus, texts):
        """
        # See more at:
        https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/17howtofindtheoptimalnumberoftopicsforlda
        """

        df_topic_sents_keywords = format_topics_sentences(lda_model=optimal_model, corpus=corpus, texts=texts)
        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        # Show dominant topics
        pd.set_option('display.max_rows', 1000)
        # Dominant topics.
        df_dominant_topic_merged = df_dominant_topic.merge(self.df, how='outer', left_index=True, right_index=True)
        # After join
        logging.debug("df_dominant_topic_merged.columns:")
        logging.debug(df_dominant_topic_merged.columns)
        df_dominant_topic_filtered_columns = df_dominant_topic_merged[
            ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']]
        # saving dominant topics with corresponding documents
        path_to_csv = Path("stats/topics/dominant_topics_and_documents.csv")
        df_dominant_topic_filtered_columns.to_csv(path_to_csv.as_posix(), sep=';',
                                                  encoding='iso8859_2', errors='replace')

        # Group top 5 sentences under each topic
        sent_topics_sorteddf = pd.DataFrame()
        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, grp.sort_values(['Perc_Contribution'],
                                                                                    ascending=[0]).head(1)], axis=0)

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
        # noinspection PyTypeChecker
        df_dominant_topics.to_csv(path_to_csv.as_posix(), sep=';', encoding='iso8859_2', errors='replace')
        print("Results saved to csv")

    @staticmethod
    def get_prefilled_full_text(slug, variant):
        recommender_methods = RecommenderMethods()
        recommender_methods.get_posts_categories_dataframe()  # joining posts and categories into one table

        found_post = recommender_methods.find_post_by_slug(slug)

        if variant == "idnes_short_text":
            column_name = 'recommended_lda'
        elif variant == "idnes_full_text":
            column_name = 'recommended_lda'
        elif variant == "wiki_eval_1":
            column_name = 'recommended_lda_wiki_eval_1'
        else:
            raise ValueError("No variant is selected from options.")
        returned_posts = found_post[column_name].iloc[0]

        return returned_posts
