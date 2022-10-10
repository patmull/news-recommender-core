import csv
import gc
import json
import logging
import os
import pickle
import random
import time
import traceback
from collections import defaultdict

import gensim
import numpy as np
import pymongo as pymongo
import regex
import tqdm
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors, Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix
from gensim.similarities.annoy import AnnoyIndexer
from gensim.utils import deaccent
from pymongo import MongoClient

from src.recommender_core.data_handling.data_handlers import flatten
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc_sim import DocSim, calculate_similarity, \
    calculate_similarity_idnes_model_gensim
from src.recommender_core.recommender_algorithms.content_based_algorithms.helper import NumpyEncoder, \
    generate_lines_from_mmcorpus
import pandas as pd

from src.recommender_core.data_handling.data_queries import RecommenderMethods, append_training_results, save_wordsim, \
    get_eval_results_header, prepare_hyperparameters_grid, random_hyperparameter_choice, \
    combine_features_from_single_df_row
from src.prefillers.preprocessing.cz_preprocessing import preprocess
from src.prefillers.preprocessing.stopwords_loading import remove_stopwords, load_cz_stopwords
from src.recommender_core.data_handling.reader import MongoReader, get_preprocessed_dict_idnes


def save_to_mongo(data, number_of_processed_files, supplied_mongo_collection):
    dict_to_insert = dict({"number": number_of_processed_files, "text": data})
    supplied_mongo_collection.insert_one(dict_to_insert)


# TODO: Repair iterator. Doesn'multi_dimensional_list work (still lops). Check Gensim Word2Vec article for guidance
class MyCorpus(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __iter__(self):
        print("Loading bigrams from preprocessed articles...")
        reader = MongoReader(db_name='idnes', col_name='preprocessed_articles_bigrams')
        print("Creating Doc2Bow...")
        i = 1
        for doc in reader.iterate():
            # assume there's one document per line, tokens separated by whitespace
            print("\rDoc. num. " + str(i), end='')
            yield self.dictionary.doc2bow(doc.get('text'))
            i = i + 1


def save_full_model_to_smaller(model="wiki"):
    print("Saving full doc2vec_model to limited doc2vec_model...")
    if model == "wiki":
        word2vec_embedding = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full",
                                                               limit=87000)  #
    elif model == "idnes":
        word2vec_embedding = KeyedVectors.load_word2vec_format("full_models/idnes/word2vec/w2v_model_full",
                                                               limit=87000)  #
    else:
        word2vec_embedding = None
    print("word2vec_embedding:")
    print(word2vec_embedding)
    word2vec_embedding.save("models/w2v_model_limited_test")  # write separately=[] for all_in_one model


def save_fast_text_to_w2v():
    print("Loading and saving FastText pretrained model to Word2Vec model")
    word2vec_model = gensim.models.fasttext.load_facebook_vectors("full_models/cswiki/word2vec/cc.cs.300.bin.gz",
                                                                  encoding="utf-8")
    print("FastText loaded...")
    word2vec_model.fill_norms()
    word2vec_model.save_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
    print("Fast text saved...")


def refresh_model():
    save_fast_text_to_w2v()
    print("Loading word2vec doc2vec_model...")
    save_full_model_to_smaller()


def save_tuple_to_csv(path, data):
    with open(path, 'w+') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['word', 'sim'])
        for row in data:
            csv_out.writerow(row)


@DeprecationWarning
def save_corpus_dict(corpus, dictionary):
    print("Saving train_corpus and dictionary...")
    pickle.dump(corpus, open('precalc_vectors/corpus_idnes.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    dictionary.save('precalc_vectors/dictionary_idnes.gensim')


def eval_idnes_basic():
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
        print("Doesn'multi_dimensional_list exists in vocab")

    save_tuple_to_csv("idnes_top_" + str(topn) + "_similar_words_to_hokej_limit_" + str(limit) + ".csv", sims)

    print("Word pairs evaluation FastText on iDNES.cz model:")
    print(idnes_model.evaluate_word_pairs('research/word2vec/similarities/WordSim353-cs-cropped.tsv'))

    overall_analogies_score, _ = idnes_model.evaluate_word_analogies(
        "research/word2vec/analogies/questions-words-cs.txt")
    print("Analogies evaluation of FastText on iDNES.cz model:")
    print(overall_analogies_score)


def eval_wiki():
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
        print("Doesn'multi_dimensional_list exists in vocab")

    # self.save_tuple_to_csv("research/word2vec/most_similar_words/cswiki_top_10_similar_words_to_hokej.csv", sims)
    save_tuple_to_csv("cswiki_top_" + str(topn) + "_similar_words_to_hokej_limit_" + str(limit) + ".csv", sims)

    print("Word pairs evaluation FastText on Wikipedia.cz model:")
    print(wiki_full_model.evaluate_word_pairs('research/word2vec/similarities/WordSim353-cs-cropped.tsv'))

    overall_analogies_score, _ = wiki_full_model.evaluate_word_analogies(
        "research/word2vec/analogies/questions-words-cs.txt")
    print("Analogies evaluation of FastText on Wikipedia.cz model:")
    print(overall_analogies_score)



def get_client():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        # Enabling Word2Vec logging
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.NOTSET)
    logger = logging.getLogger()  # get the root logger
    logger.info("Testing file write")
    # TODO: Replace with iterator when is fixed: sentences = MyCorpus(dictionary)

    client = MongoClient("localhost", 27017, maxPoolSize=50)
    return client


def create_dictionary_from_dataframe(force_update=False):
    path_to_dict = "full_models/idnes/unprocessed/idnes.dict"
    path_to_corpus = "full_models/idnes/unprocessed/idnes.mm"
    if os.path.exists(path_to_dict) is False or os.path.exists(path_to_corpus) is False or force_update is True:
        recommender_methods = RecommenderMethods()
        post_df = recommender_methods.get_posts_dataframe()
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
        word_counts = [[(dictionary[doc_id], count) for doc_id, count in line] for line in corpus]
        print(word_counts[:100])
        print("Serializing...")
        corpora.MmCorpus.serialize(path_to_corpus, corpus)  # store to disk, for later use
        print("Dictionary and  Corpus successfully saved on disk")

    else:
        print("Dictionary and train_corpus already exists")


def create_corpus_from_mongo_idnes(dictionary, force_update=False):
    path_part_1 = "precalc_vectors"
    path_part_2 = "/corpus_idnes.mm"
    path_to_corpus = path_part_1 + path_part_2

    if os.path.isfile(path_to_corpus) is False or force_update is True:
        corpus = MyCorpus(dictionary)
        gc.collect()
        print("Saving preprocessed train_corpus...")
        corpora.MmCorpus.serialize(path_to_corpus, corpus)
    else:
        print("Corpus already exists. Loading...")
        corpus = corpora.MmCorpus(path_to_corpus)
    return corpus


def get_preprocessed_dictionary(filter_extremes, path_to_dict):
    return get_preprocessed_dict_idnes(filter_extremes=filter_extremes,
                                       path_to_dict=path_to_dict)


def create_dictionary_from_mongo_idnes(force_update=False, filter_extremes=False):
    # a memory-friendly iterator
    path_to_dict = 'precalc_vectors/dictionary_idnes.gensim'
    if os.path.isfile(path_to_dict) is False or force_update is True:
        preprocessed_dictionary = get_preprocessed_dictionary(path_to_dict=path_to_dict,
                                                              filter_extremes=filter_extremes)
        return preprocessed_dictionary
    else:
        print("Dictionary already exists. Loading...")
        loaded_dict = corpora.Dictionary.load("full_models/idnes/preprocessed/dictionary")
        return loaded_dict


class Word2VecClass:
    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY +
    # "@moje-clanky/w2v_embedding_all_in_one"

    def __init__(self):
        self.documents = None
        self.df = None
        self.posts_df = None
        self.categories_df = None
        self.w2v_model = None

    # @profile
    def get_similar_word2vec(self, searched_slug, model_name=None, docsim_index=None, dictionary=None,
                             force_update_data=False, posts_from_cache=True):

        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a input_string.")
        else:
            if searched_slug == "":
                raise ValueError("Entered input_string is empty.")
            else:
                pass

        recommender_methods = RecommenderMethods()

        self.posts_df = recommender_methods.get_posts_dataframe(force_update=force_update_data,
                                                                from_cache=posts_from_cache)
        self.categories_df = recommender_methods.get_categories_dataframe()
        self.df = recommender_methods.get_posts_categories_dataframe()

        if searched_slug not in self.df['slug'].to_list():
            raise ValueError('Slug does not appear in dataframe.')

        self.categories_df = self.categories_df.rename(columns={'title': 'category_title'})
        self.categories_df = self.categories_df.rename(columns={'slug': 'category_slug'})

        found_post_dataframe = recommender_methods.find_post_by_slug(searched_slug)
        print("found_post_dataframe:")
        print(found_post_dataframe)
        print(found_post_dataframe.columns)
        print("self.categories_df")
        print(self.categories_df)
        print(self.categories_df.columns)

        found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        print("found_post_dataframe")
        print(found_post_dataframe)

        found_post_dataframe[['trigrams_full_text']] = found_post_dataframe[['trigrams_full_text']].fillna('')
        found_post_dataframe[['keywords']] = found_post_dataframe[['keywords']].fillna('')
        # noinspection PyPep8
        found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                  found_post_dataframe.iloc[0]['trigrams_full_text']

        del self.posts_df
        del self.categories_df

        documents_df = pd.DataFrame()
        """
        documents_df["features_to_use"] = self.df["category_title"] + " " + self.df["keywords"] + ' ' + self.df[
            "all_features_preprocessed"] + " " + self.df["body_preprocessed"]
        """
        # documents_df["features_to_use"] = self.df["trigrams_full_text"]
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

        if model_name == "wiki":
            w2v_model = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
            print("Similarities on Wikipedia.cz model:")
            ds = DocSim(w2v_model)
            most_similar_articles_with_scores = ds.calculate_similarity_wiki_model_gensim(found_post,
                                                                                          list_of_document_features)[:21]
        elif model_name.startswith("idnes_"):
            source = "idnes"
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
                path_to_folder = None
                ValueError("Wrong model name chosen.")
            file_name = "w2v_idnes.model"
            path_to_model = path_to_folder + file_name
            self.w2v_model = KeyedVectors.load(path_to_model)
            print("Similarities on iDNES.cz model:")
            ds = DocSim(self.w2v_model)
            print("found_post")
            print(found_post)
            if docsim_index is None and dictionary is None:
                print("Docsim or dictionary is not passed into method. Loading.")
                docsim_index = ds.load_docsim_index(source=source, model_name=model_name)
            most_similar_articles_with_scores \
                = calculate_similarity_idnes_model_gensim(found_post,
                                                          docsim_index,
                                                          dictionary,
                                                          list_of_document_features)[:21]
        else:
            raise ValueError("No from option is available.")
        # removing post itself
        if len(most_similar_articles_with_scores) > 0:
            print("most_similar_articles_with_scores:")
            print(most_similar_articles_with_scores)
            del most_similar_articles_with_scores[0]  # removing post itself
            print("most_similar_articles_with_scores after del:")
            print(most_similar_articles_with_scores)

            # workaround due to float32 error in while converting to JSON
            return json.loads(json.dumps(most_similar_articles_with_scores, cls=NumpyEncoder))
        else:
            raise ValueError("Unexpected length of 'most_similar_articles_with_scores' JSON. "
                             "Most similar articles is "
                             "smaller than 0.")

    # @profile
    def get_similar_word2vec_full_text(self, searched_slug):
        """
        Differs from src.prefillers module method

        :param searched_slug:
        :return:
        """
        recommender_methods = RecommenderMethods()

        self.posts_df = recommender_methods.get_posts_dataframe()
        self.categories_df = recommender_methods.get_categories_dataframe()
        self.df = recommender_methods.get_posts_categories_full_text()

        found_post_dataframe = recommender_methods.find_post_by_slug(searched_slug)

        print("found_post_dataframe")
        print(found_post_dataframe)

        # TODO: If this works well on production, add also to short text version
        if found_post_dataframe is None:
            return []
        else:
            print("found_post_dataframe.iloc.columns")
            print(found_post_dataframe.columns)
            print("found_post_dataframe.iloc[0]")
            print(found_post_dataframe.iloc[0])
            print("Keywords:")
            print(found_post_dataframe.iloc[0]['keywords'])
            print("all_features_preprocessed:")
            print(found_post_dataframe.iloc[0]['all_features_preprocessed'])
            print("body_preprocessed:")
            print(found_post_dataframe.iloc[0]['body_preprocessed'])
            found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id',
                                                              right_on='searched_id')
            found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
                                                      found_post_dataframe.iloc[0]['all_features_preprocessed'] \
                                                      + " " + found_post_dataframe.iloc[0]['body_preprocessed']

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

            documents_df['features_to_use'] = documents_df['features_to_use'].str.replace(';', ' ')
            documents_df['features_to_use'] = documents_df['features_to_use'].str.replace(r'\r\n', '', regex=True)
            documents_df['features_to_use'] = documents_df['features_to_use'] + "; " + documents_df['slug']
            list_of_document_features = documents_df["features_to_use"].tolist()
            del documents_df
            # https://github.com/v1shwa/document-similarity with my edits

            calculatd_similarities_for_posts = calculate_similarity(found_post,
                                                                    list_of_document_features)[:21]
            # removing post itself
            del calculatd_similarities_for_posts[0]  # removing post itself

            # workaround due to float32 error in while converting to JSON
            return json.loads(json.dumps(calculatd_similarities_for_posts, cls=NumpyEncoder))

    def evaluate_model(self, source, sentences=None, model_variant=None, negative_sampling_variant=None,
                       vector_size=None, window=None, min_count=None,
                       epochs=None, sample=None, force_update_model=True,
                       use_default_model=False, save_model=True):
        model_path = None
        if source == "idnes":
            model_path = "models/w2v_idnes.model"
        elif source == "cswiki":
            model_path = "models/w2v_cswiki.model"
        else:
            ValueError("Wrong source of the model was chosen.")

        if os.path.isfile(model_path) is False or force_update_model is True:
            if source == "idnes":
                print("Started training on iDNES.cz dataset...")
            elif source == "cswiki":
                print("Started training on cs.Wikipedia.cz dataset...")

            if use_default_model is True:
                # DEFAULT:
                self.w2v_model = Word2Vec(sentences=sentences)
                if save_model is True:
                    if source == "idnes":
                        self.w2v_model.save("models/w2v_idnes.model")
                    elif source == "cswiki":
                        self.w2v_model.save("models/w2v_cswiki.model")
            else:
                # CUSTOM:
                self.w2v_model = Word2Vec(sentences=sentences, sg=model_variant, negative=negative_sampling_variant,
                                          vector_size=vector_size, window=window, min_count=min_count, epochs=epochs,
                                          sample=sample, workers=7)
                if source == "idnes":
                    self.w2v_model.save("models/w2v_idnes.model")
                elif source == "cswiki":
                    self.w2v_model.save("models/w2v_cswiki.model")
        else:
            print("Loading Word2Vec model from saved model file")
            self.w2v_model = Word2Vec.load(model_path)

        overall_score, word_pairs_eval = self.prepare_and_run_evaluation()

        if source == "idnes":
            print("Analogies evaluation of iDnes.cz model:")
        elif source == "cswiki":
            print("Analogies evaluation of cs.wikipedia.org model:")

        print(overall_score)

        return overall_score, word_pairs_eval

    def prepare_and_run_evaluation(self):
        path_to_cropped_wordsim_file = 'research/word2vec/similarities/WordSim353-cs-cropped.tsv'
        if os.path.exists(path_to_cropped_wordsim_file):
            word_pairs_eval = self.w2v_model.wv.evaluate_word_pairs(
                path_to_cropped_wordsim_file)
        else:
            save_wordsim(path_to_cropped_wordsim_file)
            word_pairs_eval = self.w2v_model.wv.evaluate_word_pairs(path_to_cropped_wordsim_file)

        overall_score, _ = self.w2v_model.wv.evaluate_word_analogies(
            'research/word2vec/analogies/questions-words-cs.txt')
        return overall_score, word_pairs_eval

    def find_optimal_model(self, source, random_search=False, number_of_trials=512):
        """
        number_of_trials: default value according to random_order search study:
        https://dl.acm.org/doi/10.5555/2188385.2188395
        """
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Enabling Word2Vec logging
        logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                            level=logging.NOTSET)
        logger = logging.getLogger()  # get the root logger
        logger.info("Testing file write")

        sentences = []

        client = MongoClient("localhost", 27017, maxPoolSize=50)
        global db
        if source == "idnes":
            db = client.idnes
        elif source == "cswiki":
            db = client.cswiki
        else:
            ValueError("No from selected sources are in options.")
        collection = db.preprocessed_articles_trigrams
        cursor = collection.find({})
        for document in cursor:
            # joined_string = ' '.join(document['text'])
            # sentences.append([joined_string])
            sentences.append(document['text'])
        print("Sentences build into type of:")
        print(type(sentences))
        print(sentences[0:10])

        model_variants = [0, 1]  # sg parameter: 0 = CBOW; 1 = Skip-Gram
        hs_softmax_variants = [0]  # 1 = Hierarchical SoftMax
        negative_sampling_variants, no_negative_sampling, vector_size_range, window_range, min_count_range, \
        epochs_range, sample_range, corpus_title, model_results = prepare_hyperparameters_grid()

        pbar = tqdm.tqdm(total=540)
        set_title, csv_file_name = None, None
        if random_search is False:
            for model_variant in model_variants:
                for negative_sampling_variant in negative_sampling_variants:
                    for vector_size in vector_size_range:
                        for window in window_range:
                            for min_count in min_count_range:
                                for epochs in epochs_range:
                                    for sample in sample_range:
                                        for hs_softmax in hs_softmax_variants:
                                            if hs_softmax == 1:
                                                word_pairs_eval, analogies_eval = self.evaluate_model(
                                                    sentences=sentences, source=source,
                                                    model_variant=model_variant,
                                                    negative_sampling_variant=no_negative_sampling,
                                                    vector_size=vector_size,
                                                    window=window,
                                                    min_count=min_count,
                                                    epochs=epochs,
                                                    sample=sample,
                                                    force_update_model=True)
                                            else:
                                                word_pairs_eval, analogies_eval = self.evaluate_model(
                                                    sentences=sentences, source=source,
                                                    model_variant=model_variant,
                                                    negative_sampling_variant=negative_sampling_variant,
                                                    vector_size=vector_size,
                                                    window=window,
                                                    min_count=min_count,
                                                    epochs=epochs,
                                                    sample=sample)

                                            print(word_pairs_eval[0][0])
                                            if source == "idnes":
                                                set_title = "idnes"
                                            elif source == "cswiki":
                                                set_title = "cswiki"
                                            else:
                                                ValueError("Bad ource specified")
                                            model_results['Validation_Set'].append(set_title + " " + corpus_title[0])
                                            # noinspection DuplicatedCode
                                            append_training_results(source=source,
                                                                    corpus_title=corpus_title[0],
                                                                    model_variant=model_variant,
                                                                    negative_sampling_variant=negative_sampling_variant,
                                                                    vector_size=vector_size,
                                                                    window=window,
                                                                    min_count=min_count,
                                                                    epochs=epochs, sample=sample,
                                                                    hs_softmax=hs_softmax,
                                                                    pearson_coeff_word_pairs_eval=word_pairs_eval[0][0],
                                                                    pearson_p_val_word_pairs_eval=word_pairs_eval[0][1],
                                                                    spearman_p_val_word_pairs_eval=word_pairs_eval[1][
                                                                        1],
                                                                    spearman_coeff_word_pairs_eval=word_pairs_eval[1][
                                                                        0],
                                                                    out_of_vocab_ratio=word_pairs_eval[2],
                                                                    analogies_eval=analogies_eval,
                                                                    model_results=model_results)

                                            pbar.update(1)
                                            if source == "idnes":
                                                csv_file_name = 'word2vec_tuning_results_cswiki.csv'
                                            elif source == "cswiki":
                                                csv_file_name = 'word2vec_tuning_results_idnes.csv'
                                            else:
                                                ValueError("Bad source specified")
                                            # noinspection PyTypeChecker
                                            pd.DataFrame(model_results).to_csv(csv_file_name, index=False,
                                                                               mode="w")
                                            print("Saved training results...")
        else:
            for i in range(0, number_of_trials):
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
                    word_pairs_eval, analogies_eval = self.evaluate_model(sentences=sentences, source=source,
                                                                          model_variant=model_variant,
                                                                          negative_sampling_variant=no_negative_sampling,
                                                                          vector_size=vector_size,
                                                                          window=window,
                                                                          min_count=min_count,
                                                                          epochs=epochs,
                                                                          sample=sample,
                                                                          force_update_model=True)
                else:
                    word_pairs_eval, analogies_eval = self.evaluate_model(sentences=sentences, source=source,
                                                                          model_variant=model_variant,
                                                                          negative_sampling_variant=negative_sampling_variant,
                                                                          vector_size=vector_size,
                                                                          window=window,
                                                                          min_count=min_count,
                                                                          epochs=epochs,
                                                                          sample=sample,
                                                                          force_update_model=True)

                print(word_pairs_eval[0][0])
                model_results['Validation_Set'].append("cs.wikipedia.org " + corpus_title[0])
                # noinspection DuplicatedCode
                append_training_results(source=source, corpus_title=corpus_title[0],
                                        model_variant=model_variant,
                                        negative_sampling_variant=negative_sampling_variant,
                                        vector_size=vector_size,
                                        window=window, min_count=min_count, epochs=epochs,
                                        sample=sample,
                                        hs_softmax=hs_softmax,
                                        pearson_coeff_word_pairs_eval=word_pairs_eval[0][0],
                                        pearson_p_val_word_pairs_eval=word_pairs_eval[0][1],
                                        spearman_p_val_word_pairs_eval=word_pairs_eval[1][1],
                                        spearman_coeff_word_pairs_eval=word_pairs_eval[1][0],
                                        out_of_vocab_ratio=word_pairs_eval[2],
                                        analogies_eval=analogies_eval,
                                        model_results=model_results)
                pbar.update(1)
                if source == "idnes":
                    # noinspection PyTypeChecker
                    pd.DataFrame(model_results).to_csv('word2vec_tuning_results_random_search_idnes.csv', index=False,
                                                       mode="w")
                    print("Saved training results...")
                elif source == "cswiki":
                    # noinspection PyTypeChecker
                    pd.DataFrame(model_results).to_csv('word2vec_tuning_results_random_search_cswiki.csv', index=False,
                                                       mode="w")
                    print("Saved training results...")
                else:
                    ValueError("No from selected models is in options.")
        pbar.close()

    def eval_model(self, source):
        print(self.evaluate_model(source=source, sentences=None, force_update_model=False))

    def final_training_model(self, source):
        client = get_client()
        sentences = []
        """
        Method for running final evaluation based on selected parameters (i.e. through random_order search).
        """
        global db
        if source == "idnes":
            db = client.idnes
        elif source == "cswiki":
            db = client.cswiki
        else:
            ValueError("No source is available.")
        collection = db.preprocessed_articles_trigrams
        cursor = collection.find({})
        for document in cursor:
            sentences.append(document['text'])
        print("Sentences build into type of:")
        print(type(sentences))
        print(sentences[0:100])

        model_variant = 1  # sg parameter: 0 = CBOW; 1 = Skip-Gram
        # noinspection DuplicatedCode
        negative_sampling_variant = 10  # 0 = no negative sampling
        no_negative_sampling = 0  # use with hs_soft_max
        vector_size = 200
        window = 16
        min_count = 3
        epochs = 25
        sample = 0.0
        hs_softmax = 0
        # 1 = Hierarchical SoftMax

        # useful range is (0, 1e-5) acording to : https://radimrehurek.com/gensim/models/word2vec.html
        corpus_title, model_results = get_eval_results_header()

        pbar = tqdm.tqdm(total=540)

        if hs_softmax == 1:
            word_pairs_eval, analogies_eval = self.compute_eval_values_idnes(sentences=sentences,
                                                                             model_variant=model_variant,
                                                                             negative_sampling_variant=no_negative_sampling,
                                                                             vector_size=vector_size,
                                                                             window=window,
                                                                             min_count=min_count,
                                                                             epochs=epochs,
                                                                             sample=sample,
                                                                             force_update_model=True,
                                                                             )
        else:
            word_pairs_eval, analogies_eval = self.compute_eval_values_idnes(sentences=sentences,
                                                                             model_variant=model_variant,
                                                                             negative_sampling_variant=negative_sampling_variant,
                                                                             vector_size=vector_size,
                                                                             window=window,
                                                                             min_count=min_count,
                                                                             epochs=epochs,
                                                                             sample=sample,
                                                                             force_update_model=True,
                                                                             )

        print(word_pairs_eval[0][0])
        # noinspection DuplicatedCode
        append_training_results(source=source, corpus_title=corpus_title[0],
                                model_variant=model_variant,
                                negative_sampling_variant=negative_sampling_variant,
                                vector_size=vector_size,
                                window=window, min_count=min_count, epochs=epochs, sample=sample,
                                hs_softmax=hs_softmax,
                                pearson_coeff_word_pairs_eval=word_pairs_eval[0][0],
                                pearson_p_val_word_pairs_eval=word_pairs_eval[0][1],
                                spearman_p_val_word_pairs_eval=word_pairs_eval[1][1],
                                spearman_coeff_word_pairs_eval=word_pairs_eval[1][0],
                                out_of_vocab_ratio=word_pairs_eval[2],
                                analogies_eval=analogies_eval,
                                model_results=model_results)

        pbar.update(1)
        if source == "idnes":
            # noinspection PyTypeChecker
            pd.DataFrame(model_results).to_csv('word2vec_final_evaluation_results_idnes.csv', index=False,
                                               mode="a")
        elif source == "cswiki":
            # noinspection PyTypeChecker
            pd.DataFrame(model_results).to_csv('word2vec_final_evaluation_results_cswiki.csv', index=False,
                                               mode="a")
        else:
            ValueError("No source matches available options.")

        print("Saved training results...")
        pbar.close()

    def compute_eval_values_idnes(self, sentences=None, model_variant=None, negative_sampling_variant=None,
                                  vector_size=None, window=None, min_count=None,
                                  epochs=None, sample=None, force_update_model=True,
                                  w2v_model_path="models/w2v_idnes.model",
                                  use_defaul_model=False, save_model=True):
        if os.path.isfile("models/w2v_idnes.model") is False or force_update_model is True:
            print("Started training on iDNES.cz dataset...")

            if use_defaul_model is True:
                # DEFAULT:
                self.w2v_model = Word2Vec(sentences=sentences)
                if save_model is True:
                    self.w2v_model.save("models/w2v_idnes.model")
            else:
                # CUSTOM:
                self.w2v_model = Word2Vec(sentences=sentences, sg=model_variant, negative=negative_sampling_variant,
                                          vector_size=vector_size, window=window, min_count=min_count, epochs=epochs,
                                          sample=sample, workers=7)
                if save_model is True:
                    self.w2v_model.save("models/w2v_idnes.model")
        else:
            print("Loading Word2Vec iDNES.cz model from saved model file")
            self.w2v_model = Word2Vec.load(w2v_model_path)

        overall_score, word_pairs_eval = self.evaluate_model(source="idnes", model_variant=model_variant,
                                                             force_update_model=True, use_default_model=False)

        print("Analogies evaluation of iDnes.cz model:")
        print(overall_score)

        return word_pairs_eval, overall_score

    @staticmethod
    def get_prefilled_full_text(slug, variant):
        recommender_methods = RecommenderMethods()
        recommender_methods.get_posts_dataframe(force_update=False)  # load posts to dataframe
        recommender_methods.get_categories_dataframe()  # load categories to dataframe
        recommender_methods.join_posts_ratings_categories()  # joining posts and categories into one table

        found_post = recommender_methods.find_post_by_slug(slug)
        column_name = None
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
            ValueError("No variant selected matches available options.")

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

        # TODO: Deliver model to method. Does not make a sense to load every time!
        recommend_methods = RecommenderMethods()
        post_1 = recommend_methods.find_post_by_slug(slug_1)
        post_2 = recommend_methods.find_post_by_slug(slug_2)

        feature_1 = 'all_features_preprocessed'
        feature_2 = 'title'

        list_of_features = [feature_1, feature_2]

        first_text = combine_features_from_single_df_row(post_1, list_of_features)
        second_text = combine_features_from_single_df_row(post_2, list_of_features)

        print(first_text)
        print(second_text)

        first_text = preprocess(first_text).split()
        second_text = preprocess(second_text).split()

        documents = [first_text, second_text]

        dictionary = Dictionary(documents)

        first_text = dictionary.doc2bow(first_text)
        second_text = dictionary.doc2bow(second_text)

        if w2v_model is None:
            w2v_model = KeyedVectors.load("full_models/idnes/evaluated_models/word2vec_model_3/w2v_idnes.model")

        documents = [first_text, second_text]
        termsim_matrix = self.prepare_termsim_and_dictionary_for_pair(documents, dictionary, first_text,
                                                                      second_text, w2v_model)

        from gensim.models import TfidfModel
        tfidf = TfidfModel(documents)

        first_text = tfidf[first_text]
        second_text = tfidf[second_text]

        # compute word similarities # for docsim_index creation
        similarity = termsim_matrix.inner_product(first_text, second_text, normalized=(True, True))

        return similarity

    @staticmethod
    def prepare_termsim_and_dictionary_for_pair(documents, dictionary, first_text, second_text, w2v_model):
        print("documents:")
        print(documents)

        from gensim.models import TfidfModel
        documents = [first_text, second_text]
        tfidf = TfidfModel(documents)

        words = [word for word, count in dictionary.most_common()]

        try:
            word_vectors = w2v_model.wv.vectors_for_all(words, allow_inference=False)
            # produce vectors for words in train_corpus
        except AttributeError:
            # TODO: This is None Type, found out why!
            try:
                word_vectors = w2v_model.vectors_for_all(words, allow_inference=False)
            except AttributeError as e:
                print(e)
                print(traceback.format_exc())
                raise AttributeError

        indexer = AnnoyIndexer(word_vectors, num_trees=2)  # use Annoy for faster word similarity lookups
        # for similarity index
        termsim_index = WordEmbeddingSimilarityIndex(word_vectors, kwargs={'indexer': indexer})
        termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)

        return termsim_matrix

