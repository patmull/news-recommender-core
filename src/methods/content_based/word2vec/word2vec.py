import csv
import gc
import itertools
import json
import logging
import os
import pickle
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors, Word2Vec, TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix
from gensim.similarities.annoy import AnnoyIndexer
from pymongo import MongoClient

from src.data_handling.data_queries import RecommenderMethods
from src.data_handling.dataframe_methods.data_selects import combine_features_from_single_df_row
from src.data_handling.evaluation.evaluation_data_handling import save_wordsim
from src.data_handling.evaluation.evaluation_results import append_training_results
from src.data_handling.hyperparam_tuning import random_hyperparameter_choice, prepare_hyperparameters_grid
from src.data_handling.reader import get_preprocessed_dict_idnes
from src.methods.content_based.doc_sim import calculate_similarity_idnes_model_gensim, load_docsim_index
from src.methods.content_based.helper import NumpyEncoder
from src.prefillers.preprocessing.czech_preprocessing import preprocess
from src.prefillers.preprocessing.stopwords_loading import load_cz_stopwords

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from Word2vec.")
FORMAT_W2V_LOG = "%(asctime)s:%(levelname)s:%(message)s"

W2V_IDNES_MODEL_PATH = "models/w2v_idnes.model"
DEFAULT_MODEL = 'idnes_3'

# defining globals
source = None
dictionary = None


def save_to_mongo(data, number_of_processed_files, supplied_mongo_collection):
    dict_to_insert = dict({"number": number_of_processed_files, "text": data})
    supplied_mongo_collection.insert_one(dict_to_insert)


# *** HERE was also a Mongo corpus iterator.ABANDONED DUE TO no longer being needed.
# *** HERE was also a datasets cropping.ABANDONED DUE TO no longer being needed.
# *** HERE were also stat calculations for simple evaluation, e.g., word similarities for both idnes and word2vec.
# ABANDONED DUE TO no longer being needed.


def save_tuple_to_csv(path, data):
    with open(path, 'w+') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['word', 'sim'])
        for row in data:
            csv_out.writerow(row)


def get_client():
    for _handler in logging.root.handlers[:]:
        logging.root.removeHandler(_handler)

    # Enabling Word2Vec logging
    logging.basicConfig(format=FORMAT_W2V_LOG,
                        level=logging.NOTSET)
    logger = logging.getLogger()  # get the root logger
    logger.info("Testing file write")

    client = MongoClient("localhost", 27017, maxPoolSize=50)
    return client


def create_dictionary_from_dataframe(force_update=False):
    global dictionary
    path_to_dict = "full_models/idnes/unprocessed/idnes.dict"
    path_to_corpus = "full_models/idnes/unprocessed/idnes.mm"
    if os.path.exists(path_to_dict) is False or os.path.exists(path_to_corpus) is False or force_update is True:
        recommender_methods = RecommenderMethods()
        post_df = recommender_methods.get_posts_dataframe()
        post_df['full_text'] = post_df['full_text'].replace([None], '')

        post_df['all_features_preprocessed'] = post_df['all_features_preprocessed'] + ' ' + post_df['full_text']

        gc.collect()

        # Preprocessing to small to calling
        post_df['all_features_preprocessed'] = post_df.all_features_preprocessed.apply(lambda x: x.split(' '))
        post_df['all_features_preprocessed'] = post_df[['all_features_preprocessed']]

        all_features_preprocessed_list = post_df['all_features_preprocessed'].to_numpy()

        path_to_pickle = 'full_models/idnes/unprocessed/idnes.pkl'
        pickle.dump(all_features_preprocessed_list, open(path_to_pickle, 'wb'))
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
        time.sleep(50)
        dictionary = corpora.Dictionary(line for line in tokenzized_texts)
        path_to_dict = 'full_models/idnes/unprocessed/idnes.dict'
        path_to_dict_folder = 'full_models/idnes/unprocessed/'
        if not os.path.isfile(path_to_dict):
            os.makedirs(path_to_dict_folder)
        dictionary.save(path_to_dict)
        corpus = [dictionary.doc2bow(text, allow_update=True) for text in tokenzized_texts]
        # Serializing and saving...
        corpora.MmCorpus.serialize(path_to_corpus, corpus)  # store to disk, for later use

    return dictionary


def get_preprocessed_dictionary(filter_extremes, path_to_dict):
    return get_preprocessed_dict_idnes(filter_extremes=filter_extremes,
                                       path_to_dict=path_to_dict)


def create_dictionary_from_mongo_idnes(force_update=False, filter_extremes=False):
    # a memory-friendly iterator
    path_to_dict = 'precalc_vectors/word2vec/dictionary_idnes.gensim'
    if os.path.isfile(path_to_dict) is False or force_update is True:
        preprocessed_dictionary = get_preprocessed_dictionary(path_to_dict=path_to_dict,
                                                              filter_extremes=filter_extremes)
        return preprocessed_dictionary
    else:
        # Dictionary already exists. Loading...
        loaded_dict = corpora.Dictionary.load("full_models/idnes/preprocessed/_dictionary")
        return loaded_dict


def grid_like_search(_source, model_results, corpus_title, *hyperparameters_search_field):
    pbar = tqdm.tqdm(total=540)
    # loop over the cartesian product of the hyperparameters
    for _ in itertools.product(*hyperparameters_search_field):

        if _source == "idnes":
            set_title = "idnes"
        elif _source == "cswiki":
            set_title = "cswiki"
        else:
            raise ValueError("Bad ource specified")
        model_results['Validation_Set'].append(set_title + " " + corpus_title[0])

        append_training_results(model_results)

        pbar.update(1)
        pbar.close()


class ModelTrainer:

    def __init__(self):
        self.w2v_model = None
        self.logger = None
        self.client = MongoClient("localhost", 27017, maxPoolSize=50)
        self.init_logging()

    def init_logging(self):
        for _handler in logging.root.handlers[:]:
            logging.root.removeHandler(_handler)
        logging.basicConfig(format=FORMAT_W2V_LOG, level=logging.NOTSET)
        self.logger = logging.getLogger()

    def get_sentences_from_db(self, _source):
        db = self.client[_source]
        collection = db.preprocessed_articles_trigrams
        cursor = collection.find({})
        sentences = [document['text'] for document in cursor]
        return sentences

    def random_like_search(self, sentences, _source, corpus_title, number_of_trials=None, model_variants=None,
                           vector_size_range=None, sample_range=None, epochs_range=None, window_range=None,
                           min_count_range=None, negative_sampling_variants=None, hs_softmax_variants=None,
                           word2vec_hyperparameters=None):
        pbar = tqdm.tqdm(total=540)

        for _ in range(0, number_of_trials):
            hs_softmax = random.choice(hs_softmax_variants)
            model_variant, vector_size, window, min_count, epochs, sample, negative_sampling_variant \
                = random_hyperparameter_choice(model_variants=model_variants,
                                               negative_sampling_variants=negative_sampling_variants,
                                               vector_size_range=vector_size_range,
                                               sample_range=sample_range,
                                               epochs_range=epochs_range,
                                               window_range=window_range,
                                               min_count_range=min_count_range)
            negative_sampling_variant = random.choice(negative_sampling_variants)

            if hs_softmax == 1:
                word_pairs_eval, analogies_eval = self.evaluate_model(sentences=sentences, _source=_source,
                                                                      force_update_model=True,
                                                                      )
            else:
                word_pairs_eval, analogies_eval = self.evaluate_model(sentences=sentences, _source=_source,
                                                                      hyperparameters=word2vec_hyperparameters,
                                                                      force_update_model=True)

            # List of variable names
            model_results = {
                "model_variant": model_variant,
                "vector_size": vector_size,
                "window": window,
                "min_count": min_count,
                "epochs": epochs,
                "sample": sample,
                "negative_sampling_variant": negative_sampling_variant,
                "word_pairs_eval": word_pairs_eval,
                "analogies_eval": analogies_eval
            }

            # Create _dictionary using _dictionary comprehension
            model_results = {var: eval(var) for var in model_results}
            model_results['Validation_Set'].append("cs.wikipedia.org " + corpus_title[0])

            # noinspection DuplicatedCode
            append_training_results(model_results)

            if _source == "idnes":
                # noinspection PyTypeChecker
                pd.DataFrame(model_results).to_csv('word2vec_tuning_results_random_search_idnes.csv',
                                                   index=False,
                                                   mode="w")
            elif _source == "cswiki":
                # noinspection PyTypeChecker
                pd.DataFrame(model_results).to_csv('word2vec_tuning_results_random_search_cswiki.csv',
                                                   index=False,
                                                   mode="w")
            else:
                raise ValueError("No from selected models is in options.")

            pbar.update(1)
            pbar.close()

    def evaluate_model(self, sentences, _source, hyperparameters=None, force_update_model=True,
                       use_default_model=False, save_model=True):
        if _source == "idnes":
            model_path = Path(W2V_IDNES_MODEL_PATH)
        elif _source == "cswiki":
            model_path = Path("full_models/cswiki/word2vec/w2v_cswiki.model")
        else:
            raise ValueError("Wrong _source of the model was chosen.")

        if os.path.isfile(model_path) is False or force_update_model is True:
            if hyperparameters is None:
                raise ValueError("You must provide hyperparameters to evaluate the model.")

            if _source == "idnes":
                logging.info("Started training on iDNES.cz dataset...")
            elif _source == "cswiki":
                logging.info("Started training on cs.Wikipedia.cz dataset...")

            word2vec = Word2VecClass()
            word2vec.save_w2v_model(sentences, hyperparameters, use_default_model, save_model, _source)
        else:
            # Loading Word2Vec model from saved model file"
            self.w2v_model = Word2Vec.load(model_path)

        word2vec = Word2VecClass()
        overall_score, word_pairs_eval = word2vec.prepare_and_run_evaluation()

        if _source == "idnes":
            logging.info("Analogies tuning of iDnes.cz model:")
        elif _source == "cswiki":
            logging.info("Analogies tuning of cs.wikipedia.org model:")

        return overall_score, word_pairs_eval


def path_to_idnes_folder(model_name):
    if model_name.startswith("idnes_1"):
        path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_1/"
    elif model_name.startswith("idnes_2"):
        path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_2_default_parameters/"
    elif model_name.startswith("idnes_3"):
        path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_3/"
    elif model_name.startswith("idnes_4"):
        path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_4/"
    elif model_name.startswith("idnes"):
        path_to_folder = "w2v_idnes.model"
    else:
        raise ValueError("Wrong model name chosen.")

    return path_to_folder


def init_word2vec(hyperparams):
    w2v_model = Word2Vec(sentences=hyperparams['sentences'], sg=hyperparams['model_variant'],
                         negative=hyperparams['negative_sampling_bias'],
                         vector_size=hyperparams['vector_size'], window=hyperparams['window'],
                         min_count=hyperparams['min_count'], epochs=hyperparams['epochs'],
                         sample=hyperparams['sample'], workers=7)
    return w2v_model


def find_optimal_model(_source, random_search=False):
    """
    number_of_trials: default value according to random_order search study:
    https://dl.acm.org/doi/10.5555/2188385.2188395
    """
    for _handler in logging.root.handlers[:]:
        logging.root.removeHandler(_handler)
    # Enabling Word2Vec logging
    logging.basicConfig(format=FORMAT_W2V_LOG,
                        level=logging.NOTSET)
    logger = logging.getLogger()  # get the root logger
    logger.info("Testing file write")

    sentences = []

    client = MongoClient("localhost", 27017, maxPoolSize=50)
    if _source == "idnes":
        db = client.idnes
    elif _source == "cswiki":
        db = client.cswiki
    else:
        raise ValueError("No from selected sources are in options.")

    collection = db.preprocessed_articles_trigrams
    cursor = collection.find({})
    for document in cursor:
        sentences.append(document['text'])

    model_variants = [0, 1]  # sg parameter: 0 = CBOW; 1 = Skip-Gram
    hs_softmax_variants = [0]  # 1 = Hierarchical SoftMax
    negative_sampling_variants, _, vector_size_range, window_range, min_count_range, \
        epochs_range, sample_range, corpus_title, model_results = prepare_hyperparameters_grid()

    # CUSTOM:
    if random_search is False:
        # list of lists of hyperparameters
        hyperparameters_search_field = [model_variants, negative_sampling_variants, vector_size_range, window_range,
                                        min_count_range, epochs_range, sample_range, hs_softmax_variants]

        grid_like_search(sentences, _source, model_results, corpus_title,
                         hyperparameters_search_field)

        if _source == "idnes":
            csv_file_name = 'word2vec_tuning_results_cswiki.csv'
        elif _source == "cswiki":
            csv_file_name = 'word2vec_tuning_results_idnes.csv'
        else:
            raise ValueError("Bad _source specified")
        # noinspection PyTypeChecker
        pd.DataFrame(model_results).to_csv(csv_file_name, index=False,
                                           mode="w")
    else:
        model_trainer = ModelTrainer()
        model_trainer.random_like_search(sentences, _source, corpus_title)


class Word2VecClass:

    def __init__(self):
        self.w2v_model = None
        self.df = None
        self.posts_df = None
        self.categories_df = None
        self.w2v_model: Word2Vec

    def get_similar_word2vec(self, searched_slug, model_name, model=None, docsim_index=None, _dictionary=None,
                             force_update_data=False, posts_from_cache=True):

        global source
        logging.debug("Testing logging from Word2vec.")
        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a input_string.")
        elif searched_slug == "":
            raise ValueError("Entered input_string is empty.")

        recommender_methods = RecommenderMethods()

        self.posts_df = recommender_methods.get_posts_dataframe(force_update=force_update_data,
                                                                from_cache=posts_from_cache)
        self.categories_df = recommender_methods.get_categories_dataframe()
        self.df = recommender_methods.get_posts_categories_dataframe()

        logging.debug("self.posts_df:")
        logging.debug(self.posts_df)

        if searched_slug not in self.df['slug'].to_list():
            # TODO: Deal with the potential problems
            #  by counting the number of trials in config file with special variable for this purpose.
            #  If num_of_trials > 1, then throw ValueError (Same as in Doc2vec)
            raise ValueError("Slug does not appear in dataframe.")

        self.categories_df = self.categories_df.rename(columns={'title': 'category_title'})
        self.categories_df = self.categories_df.rename(columns={'slug': 'category_slug'})

        found_post_dataframe = recommender_methods.find_post_by_slug(searched_slug)
        found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        found_post_dataframe[['trigrams_full_text']] = found_post_dataframe[['trigrams_full_text']].fillna('')
        found_post_dataframe[['keywords']] = found_post_dataframe[['keywords']].fillna('')
        # noinspection PyPep8
        found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                  found_post_dataframe.iloc[0]['trigrams_full_text']

        del self.posts_df
        del self.categories_df

        documents_df = pd.DataFrame()
        documents_df["features_to_use"] = self.df["category_title"] + " " + self.df["keywords"] \
                                          + ' ' + self.df["all_features_preprocessed"] \
                                          + " " + self.df["body_preprocessed"]

        if 'slug_x' in self.df.columns:
            self.df = self.df.rename(columns={'slug_x': 'slug'})
        elif 'post_slug' in self.df.columns:
            self.df = self.df.rename(columns={'post_slug': 'slug'})
        documents_df["searched_slug"] = self.df["slug"]
        found_post = found_post_dataframe['features_to_use'].iloc[0]

        del self.df
        del found_post_dataframe

        documents_df['features_to_use'] = documents_df['features_to_use'] + "; " + documents_df['searched_slug']
        list_of_document_features = documents_df["features_to_use"].tolist()

        del documents_df
        # https://github.com/v1shwa/document-similarity with my edits
        if model is not None:
            self.w2v_model = model_name
            if model_name.startswith("idnes"):
                source = "idnes"
            elif model_name.startswith("cswiki"):
                source = "cswiki"
            else:
                raise ValueError("model_name needs to be set")
        else:
            self.load_w2v_from_keyed_vectors(model_name)

        logging.info("Calculating similarities on iDNES.cz model.")
        logging.debug("found_post:")
        logging.debug(found_post)
        if docsim_index is None and _dictionary is None:
            logging.debug("Docsim or _dictionary is not passed into method_name. Loading.")

            docsim_index = load_docsim_index(source=source, model_name=model_name)
        most_similar_articles_with_scores \
            = calculate_similarity_idnes_model_gensim(found_post,
                                                      docsim_index,
                                                      _dictionary,
                                                      list_of_document_features)[:21]

        # removing post itself
        if len(most_similar_articles_with_scores) > 0:
            logging.debug("most_similar_articles_with_scores:")
            logging.debug(most_similar_articles_with_scores)
            del most_similar_articles_with_scores[0]  # removing post itself
            logging.debug("most_similar_articles_with_scores after del:")
            logging.debug(most_similar_articles_with_scores)

            # workaround due to float32 error in while converting to JSON
            return json.loads(json.dumps(most_similar_articles_with_scores, cls=NumpyEncoder))
        else:
            raise ValueError("Unexpected length of 'most_similar_articles_with_scores' JSON. "
                             "Most similar articles is "
                             "smaller than 0.")

    # @profile
    def get_similar_word2vec_full_text(self, searched_slug):
        """
        Differs from src.prefillers module method_name

        :param searched_slug:
        :return:
        """
        recommender_methods = RecommenderMethods()

        self.posts_df = recommender_methods.get_posts_dataframe()
        self.categories_df = recommender_methods.get_categories_dataframe()
        self.df = recommender_methods.get_posts_categories_full_text()

        found_post_dataframe = recommender_methods.find_post_by_slug(searched_slug)

        if found_post_dataframe is None:
            return []
        else:
            found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id',
                                                              right_on='searched_id')
            found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                      found_post_dataframe.iloc[0]['all_features_preprocessed'] \
                                                      + " " + found_post_dataframe.iloc[0]['body_preprocessed']

            del self.posts_df
            del self.categories_df

            documents_df = pd.DataFrame()

            documents_df["features_to_use"] = self.df["keywords"] + '||' + self.df["category_title"] + ' ' + self.df[
                "all_features_preprocessed"] + ' ' + self.df["body_preprocessed"]
            documents_df["slug"] = self.df["post_slug"]

            found_post = found_post_dataframe['features_to_use'].iloc[0]

            del self.df
            del found_post_dataframe

            documents_df['features_to_use'] = documents_df['features_to_use'].str.replace(';', ' ')
            documents_df['features_to_use'] = documents_df['features_to_use'].str.replace(redis_instance'\redis_instance\n', '', regex=True)
            documents_df['features_to_use'] = documents_df['features_to_use'] + "; " + documents_df['slug']
            list_of_document_features = documents_df["features_to_use"].tolist()
            del documents_df
            # https://github.com/v1shwa/document-similarity with my edits
            _dictionary = create_dictionary_from_dataframe()

            docsim_index = load_docsim_index(source=source, model_name=DEFAULT_MODEL)
            calculatd_similarities_for_posts = calculate_similarity_idnes_model_gensim(found_post,
                                                                                       docsim_index,
                                                                                       _dictionary,
                                                                                       list_of_document_features)[:21]
            # removing post itself
            del calculatd_similarities_for_posts[0]  # removing post itself

            # workaround due to float32 error in while converting to JSON
            return json.loads(json.dumps(calculatd_similarities_for_posts, cls=NumpyEncoder))

    def save_w2v_model(self, sentences, hyperparams, use_default_model=False, save_model=True, _source="idnes"):
        if use_default_model is True:
            # DEFAULT:
            self.w2v_model = Word2Vec(sentences=sentences)
            if save_model is True:
                if _source == "idnes":
                    self.w2v_model.save(W2V_IDNES_MODEL_PATH)
                elif _source == "cswiki":
                    self.w2v_model.save("models/w2v_cswiki.model")
        else:
            # CUSTOM:
            self.w2v_model = init_word2vec(hyperparams)

            if _source == "idnes":
                self.w2v_model.save(W2V_IDNES_MODEL_PATH)
            elif _source == "cswiki":
                self.w2v_model.save("models/w2v_cswiki.model")

    def prepare_and_run_evaluation(self):
        path_to_cropped_wordsim_file = 'stats/word2vec/similarities/WordSim353-cs-cropped.tsv'
        if os.path.exists(path_to_cropped_wordsim_file):
            word_pairs_eval = self.w2v_model.wv.evaluate_word_pairs(
                path_to_cropped_wordsim_file)
        else:
            save_wordsim(path_to_cropped_wordsim_file)
            word_pairs_eval = self.w2v_model.wv.evaluate_word_pairs(path_to_cropped_wordsim_file)

        overall_score, _ = self.w2v_model.wv.evaluate_word_analogies(
            'stats/word2vec/analogies/questions-words-cs.txt')
        return overall_score, word_pairs_eval

    @staticmethod
    def get_prefilled_full_text(slug, variant):
        recommender_methods = RecommenderMethods()
        found_post = recommender_methods.find_post_by_slug(slug)
        if variant == "idnes_short_text":
            column_name = 'recommended_word2vec'
        elif variant == "idnes_full_text":
            column_name = 'recommended_word2vec_full_text'
        elif variant == "idnes_eval_1":
            column_name = 'recommended_word2vec_eval_1'
        elif variant == "idnes_eval_2":
            column_name = 'recommended_word2vec_eval_2'
        elif variant == "idnes_eval_3":
            column_name = 'recommended_word2vec_eval_3'
        elif variant == "idnes_eval_4":
            column_name = 'recommended_word2vec_eval_4'
        elif variant == 'fasttext_limited':
            column_name = 'recommended_word2vec_limited_fasttext'
        elif variant == "fasttext_limited_full_text":
            column_name = 'recommended_word2vec_limited_fasttext_full_text'
        elif variant == 'wiki_eval_1':
            column_name = 'recommended_word2vec_wiki_eval_1'
        else:
            raise ValueError("No variant selected matches available options.")

        returned_post = found_post[column_name].iloc[0]
        return returned_post

    @staticmethod
    def preprocess(sentence):
        stop_words = load_cz_stopwords()
        return [w for w in sentence.lower().split() if w not in stop_words]

    @staticmethod
    def get_vector(s, models):
        return np.sum(
            np.array([models[i] for i in preprocess(s)]), axis=0)

    def get_pair_similarity_word2vec(self, slug_1, slug_2, w2v_model=None):

        logging.debug('Calculating Word2Vec pair similarity for posts:')
        logging.debug(slug_1)
        logging.debug(slug_2)

        recommend_methods = RecommenderMethods()
        post_1 = recommend_methods.find_post_by_slug(slug_1)
        post_2 = recommend_methods.find_post_by_slug(slug_2)

        feature_1 = 'all_features_preprocessed'
        feature_2 = 'title'

        list_of_features = [feature_1, feature_2]

        first_text = combine_features_from_single_df_row(post_1, list_of_features)
        second_text = combine_features_from_single_df_row(post_2, list_of_features)

        first_text = preprocess(first_text).split()
        second_text = preprocess(second_text).split()

        documents = [first_text, second_text]

        _dictionary = Dictionary(documents)

        first_text = _dictionary.doc2bow(first_text)
        second_text = _dictionary.doc2bow(second_text)

        if w2v_model is None:
            w2v_model = KeyedVectors.load("full_models/idnes/evaluated_models/word2vec_model_3/w2v_idnes.model")

        documents = [first_text, second_text]
        termsim_matrix = self.prepare_termsim_and_dictionary_for_pair(_dictionary, first_text,
                                                                      second_text, w2v_model)

        from gensim.models import TfidfModel
        tfidf = TfidfModel(documents)

        first_text = tfidf[first_text]
        second_text = tfidf[second_text]

        # compute word similarities # for docsim_index creation
        similarity = termsim_matrix.inner_product(first_text, second_text, normalized=(True, True))

        return similarity

    @staticmethod
    def prepare_termsim_and_dictionary_for_pair(_dictionary, first_text, second_text, w2v_model):

        documents = [first_text, second_text]
        tfidf = TfidfModel(documents)

        words = [word for word, count in _dictionary.most_common()]

        try:
            word_vectors = w2v_model.wv.vectors_for_all(words, allow_inference=False)
            # produce vectors for words in train_corpus
        except AttributeError:
            try:
                word_vectors = w2v_model.vectors_for_all(words, allow_inference=False)
            except AttributeError as e:
                logging.error(e)
                logging.error(traceback.format_exc())
                raise AttributeError

        indexer = AnnoyIndexer(word_vectors, num_trees=2)  # use Annoy for faster word similarity lookups
        # for similarity index
        termsim_index = WordEmbeddingSimilarityIndex(word_vectors, kwargs={'indexer': indexer})
        termsim_matrix = SparseTermSimilarityMatrix(termsim_index, _dictionary, tfidf)

        return termsim_matrix

    def load_w2v_from_keyed_vectors(self, model_name):
        if model_name == "cswiki":
            self.w2v_model = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
        elif model_name.startswith("idnes"):
            path_to_folder = path_to_idnes_folder(model_name)
            file_name = "w2v_idnes.model"
            path_to_model = path_to_folder + file_name
            self.w2v_model = KeyedVectors.load(path_to_model)
        else:
            raise ValueError("No from option is available.")


if __name__ == '__main__':
    logging.info("Word2Vec module")
