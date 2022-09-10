import gc
import os

import pymongo
import logging
import re

from gensim import corpora
from pymongo import MongoClient
from src.prefillers.preprocessing.cz_preprocessing import CzPreprocess


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO
_logger = logging.getLogger(__name__)


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
        # noinspection
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

    def __init__(self, db_name=None, col_name=None, mongo_uri="mongodb://localhost:27017", limit=None):
        ''' init
            :param mongo_uri: mongoDB URI. default: localhost:27017
            :param db_name: MongoDB database name.
            :param col_name: MongoDB Collection name.
            :param limit: query limit
        '''
        Reader.__init__(self)
        super().__init__()
        self.conn = None
        self.mongoURI = mongo_uri
        self.dbName = db_name
        self.collName = col_name
        self.limit = limit
        self.fields = ['text']
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
            # tags = [multi_dimensional_list.strip() for multi_dimensional_list in tags]
            doc = { "text": texts }
            yield doc

    def get_preprocessed_dict_idnes(self, sentences, filter_extremes, path_to_dict):
        sentences = self.build_sentences()
        print("Creating dictionary...")
        preprocessed_dictionary = corpora.Dictionary(line for line in sentences)
        del sentences
        gc.collect()
        if filter_extremes is True:
            preprocessed_dictionary.filter_extremes()
        print("Saving dictionary...")
        preprocessed_dictionary.save(path_to_dict)
        print("Dictionary saved to: " + path_to_dict)
        return preprocessed_dictionary

    def create_dictionary_from_mongo_idnes(self, sentences=None, force_update=False, filter_extremes=False):
        # a memory-friendly iterator
        path_to_dict = 'precalc_vectors/dictionary_idnes.gensim'
        if os.path.isfile(path_to_dict) is False or force_update is True:
            return self.get_preprocessed_dict_idnes(sentences, filter_extremes, path_to_dict)
        else:
            print("Dictionary already exists. Loading...")
            loaded_dict = corpora.Dictionary.load(path_to_dict)
            return loaded_dict

    def build_sentences(self):
        print("Building sentences...")
        sentences = []
        client = MongoClient("localhost", 27017, maxPoolSize=50)
        db = client.idnes
        collection = db.preprocessed_articles_bigrams
        cursor = collection.find({})
        for document in cursor:
            # joined_string = ' '.join(document['text'])
            # sentences.append([joined_string])
            sentences.append(document['text'])
        return sentences


if __name__ == "__main__":
    pass