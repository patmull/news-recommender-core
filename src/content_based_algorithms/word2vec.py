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
import psycopg2
import pymongo as pymongo
import regex
import tqdm
from gensim import corpora
from gensim.models import KeyedVectors, Word2Vec
from gensim.utils import deaccent
from pymongo import MongoClient

from src.content_based_algorithms.doc_sim import DocSim, calculate_similarity
from src.content_based_algorithms.gensim_native_models import GensimMethods
from src.content_based_algorithms.helper import NumpyEncoder, Helper
import pandas as pd
import time as t

from src.data_manipulation import Database
from src.data_handling.data_queries import RecommenderMethods
from src.preprocessing.cz_preprocessing import CzPreprocess
from src.preprocessing.stopwords_loading import remove_stopwords
from src.data_handling.reader import MongoReader

PATH_TO_UNPROCESSED_QUESTIONS_WORDS = 'research/word2vec/analogies/questions-words-cs-unprocessed.txt'
PATH_TO_PREPROCESSED_QUESTIONS_WORDS = 'research/word2vec/analogies/questions-words-cs.txt'

myclient = pymongo.MongoClient('localhost', 27017)
db = myclient.test
mongo_db = myclient["idnes"]
mongo_collection = mongo_db["preprocessed_articles"]
mongo_collection_stopwords_free = mongo_db["preprocessed_articles_stopwords_free"]
mongo_collection_bigrams = mongo_db["preprocessed_articles_bigrams"]


def save_to_mongo(data, number_of_processed_files, supplied_mongo_collection):
    dict_to_insert = dict({"number": number_of_processed_files, "text": data})
    supplied_mongo_collection.insert_one(dict_to_insert)


# TODO: Repair iterator. Doesn'multi_dimensional_list work (still lops). Check Gensim Word2Vec article for guidance
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


@DeprecationWarning
def build_bigrams_and_trigrams(force_update=False):
    cursor_any_record = mongo_collection_bigrams.find_one()
    if cursor_any_record is not None and force_update is False:
        print("There are already records in MongoDB. Skipping bigrams building.")
        pass
    else:
        print("Building bigrams...")
        # mongo_collection_bigrams.delete_many({})
        print("Loading stopwords free documents...")
        # using 80% training set

        reader = MongoReader(dbName='idnes', collName='preprocessed_articles_stopwords_free')

        print("Building sentences...")
        sentences = [doc.get('text') for doc in reader.iterate()]

        first_sentence = next(iter(sentences))
        print("first_sentence[:10]")
        print(first_sentence[:10])

        print("Sentences sample:")
        print(sentences[1500:1600])
        time.sleep(40)
        phrase_model = gensim.models.Phrases(sentences, min_count=1, threshold=1)  # higher threshold fewer phrases.

        cursor = mongo_collection_stopwords_free.find({})
        i = 1
        for doc in cursor:
            print("Building bigrams for document number " + str(i))
            bigram_text = phrase_model[doc['text']]
            print("bigram_text:")
            print(bigram_text)
            save_to_mongo(number_of_processed_files=i, data=bigram_text,
                          supplied_mongo_collection=mongo_collection_bigrams)
            i = i + 1


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


def remove_stopwords_mongodb(force_update=False):
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
            removed_stopwords = remove_stopwords(doc['text'])
            helper = Helper()
            removed_stopwords = helper.flatten(removed_stopwords)
            print("After removal:")
            print(removed_stopwords)
            save_to_mongo(number_of_processed_files=number, data=removed_stopwords,
                          supplied_mongo_collection=mongo_collection_stopwords_free)
            number = number + 1


def preprocess_idnes_corpus(force_update=False):
    print("Corpus lines are above")
    cursor_any_record = mongo_collection.find_one()
    if cursor_any_record is not None and force_update is False:
        print("There are already records in MongoDB. Skipping Idnes preprocessing (1st phase)")
        pass
    else:
        path_to_pickle = 'full_models/idnes/unprocessed/idnes.pkl'
        corpus = pickle.load(open(path_to_pickle, 'rb'))
        print("Corpus length:")
        print(len(corpus))
        time.sleep(120)
        # preprocessing steps
        czpreprocessing = CzPreprocess()
        helper = Helper()

        last_record = mongo_db.mongo_collection.find()
        print("last_record")
        print(last_record)
        print("Fetching records for DB...")
        cursor_any_record = mongo_collection.find_one()
        # Checking the cursor is empty or not
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


def get_preprocessed_dictionary(sentences, filter_extremes, path_to_dict):
    mongo_reader = MongoReader()
    return mongo_reader.get_preprocessed_dict_idnes(sentences=sentences, filter_extremes=filter_extremes,
                                                    path_to_dict=path_to_dict)


def create_dictionary_from_mongo_idnes(sentences=None, force_update=False, filter_extremes=False):
    # a memory-friendly iterator
    path_to_dict = 'precalc_vectors/dictionary_idnes.gensim'
    if os.path.isfile(path_to_dict) is False or force_update is True:
        preprocessed_dictionary = get_preprocessed_dictionary(sentences=sentences, path_to_dict=path_to_dict,
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
    def get_similar_word2vec(self, searched_slug, model=None, docsim_index=None, dictionary=None,
                             force_update_data=False):

        recommender_methods = RecommenderMethods()

        self.posts_df = recommender_methods.get_posts_dataframe(force_update=force_update_data)
        self.categories_df = recommender_methods.get_categories_dataframe()

        self.df = recommender_methods.get_posts_categories_dataframe()

        self.categories_df = self.categories_df.rename(columns={'title': 'category_title'})
        self.categories_df = self.categories_df.rename(columns={'slug': 'category_slug'})

        found_post_dataframe = recommender_methods.find_post_by_slug(searched_slug)
        found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        print("found_post_dataframe:")
        print(found_post_dataframe)
        print(found_post_dataframe.columns)

        found_post_dataframe[['trigrams_full_text']] = found_post_dataframe[['trigrams_full_text']].fillna('')
        found_post_dataframe[['keywords']] = found_post_dataframe[['keywords']].fillna('')
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
        documents_df["features_to_use"] = self.df["category_title"] + " " + self.df["keywords"] + ' ' + self.df[
            "all_features_preprocessed"] + " " + self.df["body_preprocessed"]

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

        global most_similar_articles_with_scores
        if model == "wiki":
            source = "cswiki"
            w2v_model = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
            print("Similarities on Wikipedia.cz model:")
            ds = DocSim(w2v_model)
            most_similar_articles_with_scores = ds.calculate_similarity_wiki_model_gensim(found_post,
                                                                                          list_of_document_features)[
                                                :21]
        elif model.startswith("idnes_"):
            source = "idnes"
            if model.startswith("idnes_1"):
                path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_1/"
            elif model.startswith("idnes_2"):
                path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_2_default_parameters/"
            elif model.startswith("idnes_3"):
                path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_3/"
            elif model.startswith("idnes_4"):
                path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_4/"
            elif model.startswith("idnes"):
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
                docsim_index, dictionary = ds.load_docsim_index_and_dictionary(source=source, model=model)
            most_similar_articles_with_scores = ds.calculate_similarity_idnes_model_gensim(found_post, docsim_index,
                                                                                           dictionary,
                                                                                           list_of_document_features)[
                                                :21]
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
        Differs from Prefillers module method

        :param searched_slug:
        :return:
        """
        recommender_methods = RecommenderMethods()

        self.posts_df = recommender_methods.get_posts_dataframe()
        self.categories_df = recommender_methods.get_categories_dataframe()
        self.df = recommender_methods.get_posts_categories_full_text()

        found_post_dataframe = recommender_methods.find_post_by_slug(searched_slug, force_update=True)

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
            found_post_dataframe = found_post_dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
            found_post_dataframe['features_to_use'] = found_post_dataframe.iloc[0]['keywords'] + "||" + \
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

    def fill_recommended_for_all_posts(self, skip_already_filled, full_text=True, random_order=False,
                                       reversed_order=False):
        recommender_methods = RecommenderMethods()
        if skip_already_filled is False:
            posts = recommender_methods.get_all_posts()
        else:
            database.connect()
            posts = database.get_not_prefilled_posts(full_text, method="tfidf")
            database.disconnect()

        number_of_inserted_rows = 0

        if reversed_order is True:
            print("Reversing list of posts...")
            posts.reverse()

        if random_order is True:
            print("Starting random_order iteration...")
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
                        except psycopg2.Error as e:
                            print("Error in DB insert. Skipping.")
                            print(e)
                            print(traceback.format_exc())
                            pass
                    else:
                        try:
                            database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                             article_id=post_id, full_text=True, db="pgsql",
                                                             method="word2vec")
                        except psycopg2.Error as e:
                            print("Error in DB insert. Skipping.")
                            print(e)
                            print(traceback.format_exc())
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

    def prefilling_job(self, full_text, reverse, random_order=False):
        if full_text is False:
            for i in range(100):
                while True:
                    try:
                        self.fill_recommended_for_all_posts(skip_already_filled=True, full_text=False,
                                                            reversed_order=reverse,
                                                            random_order=random_order)
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

    def evaluate_model(self, source, sentences=None, model_variant=None, negative_sampling_variant=None,
                       vector_size=None, window=None, min_count=None,
                       epochs=None, sample=None, force_update_model=True,
                       use_default_model=False, save_model=True):
        global model_path
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
            df = pd.read_csv('research/word2vec/similarities/WordSim353-cs.csv',
                             usecols=['cs_word_1', 'cs_word_2', 'cs mean'])
            cz_preprocess = CzPreprocess()
            df['cs_word_1'] = df['cs_word_1'].apply(lambda x: gensim.utils.deaccent(cz_preprocess.preprocess(x)))
            df['cs_word_2'] = df['cs_word_2'].apply(lambda x: gensim.utils.deaccent(cz_preprocess.preprocess(x)))

            df.to_csv(path_to_cropped_wordsim_file, sep='\t', encoding='utf-8', index=False)
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
        negative_sampling_variants = range(5, 20, 5)  # 0 = no negative sampling
        no_negative_sampling = 0  # use with hs_soft_max
        # vector_size_range = [50, 100, 158, 200, 250, 300, 450]
        vector_size_range = [50, 100, 158, 200, 250, 300, 450]
        # window_range = [1, 2, 4, 5, 8, 12, 16, 20]
        window_range = [1, 2, 4, 5, 8, 12, 16, 20]
        min_count_range = [0, 1, 2, 3, 5, 8, 12]
        epochs_range = [20, 25, 30]
        sample_range = [0.0, 1.0 * (10.0 ** -1.0), 1.0 * (10.0 ** -2.0), 1.0 * (10.0 ** -3.0), 1.0 * (10.0 ** -4.0),
                        1.0 * (10.0 ** -5.0)]
        # useful range is (0, 1e-5) acording to : https://radimrehurek.com/gensim/models/word2vec.html

        corpus_title = ['100% Corpus']
        model_results = {'Validation_Set': [],
                         'Model_Variant': [],
                         'Negative': [],
                         'Vector_size': [],
                         'Window': [],
                         'Min_count': [],
                         'Epochs': [],
                         'Sample': [],
                         'Softmax': [],
                         'Word_pairs_test_Pearson_coeff': [],
                         'Word_pairs_test_Pearson_p-val': [],
                         'Word_pairs_test_Spearman_coeff': [],
                         'Word_pairs_test_Spearman_p-val': [],
                         'Word_pairs_test_Out-of-vocab_ratio': [],
                         'Analogies_test': []
                         }  # Can take a long time to run
        pbar = tqdm.tqdm(total=540)
        global set_title, csv_file_name
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
                                            model_results['Model_Variant'].append(model_variant)
                                            model_results['Negative'].append(negative_sampling_variant)
                                            model_results['Vector_size'].append(vector_size)
                                            model_results['Window'].append(window)
                                            model_results['Min_count'].append(min_count)
                                            model_results['Epochs'].append(epochs)
                                            model_results['Sample'].append(sample)
                                            model_results['Softmax'].append(hs_softmax)
                                            model_results['Word_pairs_test_Pearson_coeff'].append(word_pairs_eval[0][0])
                                            model_results['Word_pairs_test_Pearson_p-val'].append(word_pairs_eval[0][1])
                                            model_results['Word_pairs_test_Spearman_coeff'].append(
                                                word_pairs_eval[1][0])
                                            model_results['Word_pairs_test_Spearman_p-val'].append(
                                                word_pairs_eval[1][1])
                                            model_results['Word_pairs_test_Out-of-vocab_ratio'].append(
                                                word_pairs_eval[2])
                                            model_results['Analogies_test'].append(analogies_eval)

                                            pbar.update(1)
                                            if source == "idnes":
                                                csv_file_name = 'word2vec_tuning_results_cswiki.csv'
                                            elif source == "cswiki":
                                                csv_file_name = 'word2vec_tuning_results_idnes.csv'
                                            else:
                                                ValueError("Bad source specified")
                                            pd.DataFrame(model_results).to_csv(csv_file_name, index=False,
                                                                               mode="w")
                                            print("Saved training results...")
        else:
            for i in range(0, number_of_trials):
                hs_softmax = random.choice(hs_softmax_variants)
                model_variant = random.choice(model_variants)
                vector_size = random.choice(vector_size_range)
                window = random.choice(window_range)
                min_count = random.choice(min_count_range)
                epochs = random.choice(epochs_range)
                sample = random.choice(sample_range)
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
                model_results['Model_Variant'].append(model_variant)
                model_results['Negative'].append(negative_sampling_variant)
                model_results['Vector_size'].append(vector_size)
                model_results['Window'].append(window)
                model_results['Min_count'].append(min_count)
                model_results['Epochs'].append(epochs)
                model_results['Sample'].append(sample)
                model_results['Softmax'].append(hs_softmax)
                model_results['Word_pairs_test_Pearson_coeff'].append(word_pairs_eval[0][0])
                model_results['Word_pairs_test_Pearson_p-val'].append(word_pairs_eval[0][1])
                model_results['Word_pairs_test_Spearman_coeff'].append(word_pairs_eval[1][0])
                model_results['Word_pairs_test_Spearman_p-val'].append(word_pairs_eval[1][1])
                model_results['Word_pairs_test_Out-of-vocab_ratio'].append(word_pairs_eval[2])
                model_results['Analogies_test'].append(analogies_eval)

                pbar.update(1)
                if source == "idnes":
                    pd.DataFrame(model_results).to_csv('word2vec_tuning_results_random_search_idnes.csv', index=False,
                                                       mode="w")
                    print("Saved training results...")
                elif source == "cswiki":
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

        corpus_title = ['100% Corpus']
        model_results = {'Validation_Set': [],
                         'Model_Variant': [],
                         'Negative': [],
                         'Vector_size': [],
                         'Window': [],
                         'Min_count': [],
                         'Epochs': [],
                         'Sample': [],
                         'Softmax': [],
                         'Word_pairs_test_Pearson_coeff': [],
                         'Word_pairs_test_Pearson_p-val': [],
                         'Word_pairs_test_Spearman_coeff': [],
                         'Word_pairs_test_Spearman_p-val': [],
                         'Word_pairs_test_Out-of-vocab_ratio': [],
                         'Analogies_test': []
                         }  # Can take a long time to run
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

        model_results['Validation_Set'].append(source + " " + corpus_title[0])
        model_results['Model_Variant'].append(model_variant)
        model_results['Negative'].append(negative_sampling_variant)
        model_results['Vector_size'].append(vector_size)
        model_results['Window'].append(window)
        model_results['Min_count'].append(min_count)
        model_results['Epochs'].append(epochs)
        model_results['Sample'].append(sample)
        model_results['Softmax'].append(hs_softmax)
        model_results['Word_pairs_test_Pearson_coeff'].append(word_pairs_eval[0][0])
        model_results['Word_pairs_test_Pearson_p-val'].append(word_pairs_eval[0][1])
        model_results['Word_pairs_test_Spearman_coeff'].append(word_pairs_eval[1][0])
        model_results['Word_pairs_test_Spearman_p-val'].append(word_pairs_eval[1][1])
        model_results['Word_pairs_test_Out-of-vocab_ratio'].append(word_pairs_eval[2])
        model_results['Analogies_test'].append(analogies_eval)

        pbar.update(1)
        if source == "idnes":
            pd.DataFrame(model_results).to_csv('word2vec_final_evaluation_results_idnes.csv', index=False,
                                               mode="a")
        elif source == "cswiki":
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


def preprocess_question_words_file():
    # open file1 in reading mode
    file1 = open(PATH_TO_UNPROCESSED_QUESTIONS_WORDS, 'r', encoding="utf-8")

    # open file2 in writing mode
    file2 = open(PATH_TO_PREPROCESSED_QUESTIONS_WORDS, 'w', encoding="utf-8")

    # read from file1 and write to file2
    for line in file1:
        if len(line.split()) == 4 or line.startswith(":"):
            if not line.startswith(":"):
                cz_preprocess = CzPreprocess()
                file2.write(gensim.utils.deaccent(cz_preprocess.preprocess(line)) + "\n")
            else:
                file2.write(line)
        else:
            continue

    # close file1 and file2
    file1.close()
    file2.close()

    # open file2 in reading mode
    file2 = open(PATH_TO_PREPROCESSED_QUESTIONS_WORDS, 'r')

    # print the file2 content
    print(file2.read())

    # close the file2
    file2.close()
