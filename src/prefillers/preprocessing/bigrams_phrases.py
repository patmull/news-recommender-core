import logging
import os
import time

import gensim
import pymongo

from src.recommender_core.data_handling.reader import MongoReader

for handler in logging.root.handlers[:]:
   logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging.")

myclient = pymongo.MongoClient('localhost', 27017)
db = myclient.test
mongo_db = myclient["idnes"]
mongo_collection = mongo_db["preprocessed_articles"]
mongo_collection_stopwords_free = mongo_db["preprocessed_articles_stopwords_free"]
mongo_collection_bigrams = mongo_db["preprocessed_articles_bigrams"]

PATH_TO_FROZEN_BIGRAM_MODEL = "full_models/idnes/ngrams/bigrams_phrase_model_frozen.pkl"
PATH_TO_FROZEN_TRIGRAM_MODEL = "full_models/idnes/ngrams/trigrams_phrase_model_frozen.pkl"


def train_phrases_from_mongo_idnes():
    """
    Training and generating the phrases model file based on corpus textual records stored in MongoDB.
    :rtype: object
    """
    logging.debug("Building bigrams...")
    # mongo_collection_bigrams.delete_many({})
    logging.debug("Loading stopwords free documents...")
    # using 80% training set

    reader = MongoReader(db_name='idnes', coll_name='preprocessed_articles_stopwords_free')

    logging.debug("Building sentences...")
    sentences = [doc.get('text') for doc in reader.iterate()]

    first_sentence = next(iter(sentences))
    logging.debug("first_sentence[:10]")
    logging.debug(first_sentence[:10])

    logging.debug("Sentences sample:")
    logging.debug(sentences[1500:1600])
    time.sleep(40)
    logging.debug("Training Phrases doc2vec_model...")
    phrase_model = gensim.models.Phrases(sentences, min_count=1, threshold=1)  # higher threshold fewer phrases.
    folder = "full_models/idnes/ngrams"
    filename = "bigrams.phrases"
    path = folder + filename
    if not os.path.exists(folder):
        os.makedirs(folder)
    logging.debug("Saving phrases doc2vec_model to " + str(path))
    phrase_model.save(path)
    # less RAM, no updates allowed
    frozen_model = phrase_model.freeze()
    filename_frozen = "bigrams_phrase_model_frozen.pkl"
    path_to_frozen = folder + filename_frozen
    frozen_model.save(path_to_frozen)


def freeze_existing_phrase_model(path_to_existing_phrases_model=None, path_to_frozen=None):
    """
    Converting the existing bigram file to (smaller) freezed variant of the file for better space efficiency.
    :param path_to_existing_phrases_model: defaults to None, if not set, it uses the default bigrams model location
    :param path_to_frozen:  defaults to None, if not set, it uses the default frozen model location
    :return:
    """
    if path_to_existing_phrases_model is None:
        folder = "full_models/idnes/"
        filename = "bigrams.phrases"
        path_to_existing_phrases_model = folder + filename
        logging.debug("None path to existing phrases doc2vec_model supplied, using defualt location in:")
        logging.debug(str(path_to_existing_phrases_model))
        logging.debug("Loading existing Phrases doc2vec_model")
        phrases_model = gensim.models.Phrases.load(path_to_existing_phrases_model)
        logging.debug("Freezing doc2vec_model...")
        frozen_model = phrases_model.freeze()
        if path_to_frozen is None:
            folder = "full_models/idnes/ngrams"
            filename_frozen = "bigrams_phrase_model_frozen.pkl"
            path_to_frozen = folder + filename_frozen
            logging.debug("None path to frozen doc2vec_model supplied, saving frozen doc2vec_model to default location in:")
            logging.debug(str(path_to_frozen))
            frozen_model.save(path_to_frozen)


class PhrasesCreator:
    """
    Class for handling the creation of the phrases n-grams models from the corpus.
    """
    def __init__(self):
        logging.debug("Loading phrases doc2vec_model...")
        self.bigram_phrases_model = gensim.models.Phrases.load(PATH_TO_FROZEN_BIGRAM_MODEL)
        self.trigram_phrases_model = gensim.models.Phrases.load(PATH_TO_FROZEN_TRIGRAM_MODEL)

    def create_bigrams(self, splitted_tokens):
        """
        Method for bigrams processing and creation.
        :param splitted_tokens: from provided list of strings, create bigrams
        :return: splitted text of bigrams
        """
        bigrams = self.bigram_phrases_model[splitted_tokens]
        bigram_text = ' '.join(bigrams)
        return bigram_text

    def create_trigrams(self, splitted_tokens):
        """
        Method for processing and creation of trigrams.
        :param splitted_tokens: provided list of strings for the creation of bigrams and trigrams
        :return: splitted text of trigrams
        """
        trigrams = self.trigram_phrases_model[self.bigram_phrases_model[splitted_tokens]]
        bigram_text = ' '.join(trigrams)
        return bigram_text
