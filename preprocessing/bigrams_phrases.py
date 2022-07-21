import os
import time

import gensim
import pymongo

from reader import MongoReader

myclient = pymongo.MongoClient('localhost', 27017)
db = myclient.test
mongo_db = myclient["idnes"]
mongo_collection = mongo_db["preprocessed_articles"]
mongo_collection_stopwords_free = mongo_db["preprocessed_articles_stopwords_free"]
mongo_collection_bigrams = mongo_db["preprocessed_articles_bigrams"]


class BigramPhrases():

    def train_phrases_from_mongo_idnes(self):

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
        print("Training Phrases model...")
        phrase_model = gensim.models.Phrases(sentences, min_count=1, threshold=1)  # higher threshold fewer phrases.
        folder = "full_model/idnes/"
        filename = "bigrams.phrases"
        path = folder + filename
        if not os.path.exists(folder):
            os.makedirs(folder)
        print("Saving phrases model to " + str(path))
        phrase_model.save(path)
        # less RAM, no updates allowed
        frozen_model = phrase_model.freeze()
        frozen_model.save("/full_model/idnes/bigrams_phrase_model_frozen.pkl")
