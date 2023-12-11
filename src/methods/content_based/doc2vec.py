import csv
import gc
import logging
import os
import random
from pathlib import Path

import gensim
import pandas as pd
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pymongo import MongoClient
from scipy import spatial
from sklearn.model_selection import train_test_split


import config.trials_counter
from src.checks.data_types import check_empty_string, accepts_first_argument
from src.data_handling.data_manipulation import DatabaseMethods
from src.data_handling.data_queries import RecommenderMethods
from src.data_handling.data_tools import flatten
from src.data_handling.evaluation.evaluation_data_handling import save_wordsim
from src.data_handling.reader import build_sentences
from src.methods.content_based.helper import verify_searched_slug_sanity
from src.methods.content_based.models_manipulation.models_loaders import load_doc2vec_model
from src.prefillers.preprocessing.czech_preprocessing import preprocess, preprocess_columns

DEFAULT_MODEL_LOCATION = "models/d2v_limited.model"
TESTING_MODEL_LOCATION = "tests/models/d2v_testing.model"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from Doc2Vec.")


def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


def print_eval_values(source, train_corpus=None, test_corpus=None, model_variant=None,
                      negative_sampling_variant=None,
                      vector_size=None, window=None, min_count=None,
                      epochs=None, sample=None, force_update_model=True,
                      default_parameters=False):
    if source == "idnes":
        model_path = "models/d2v_idnes.model"
    elif source == "cswiki":
        model_path = "models/d2v_cswiki.model"
    else:
        raise ValueError("No _source matches available options.")

    if os.path.isfile(model_path) is False or force_update_model is True:
        # Started training on iDNES.cz dataset...

        if default_parameters is True:
            # DEFAULT:
            d2v_model = Doc2Vec()
        else:
            # CUSTOM:
            d2v_model = Doc2Vec(dm=model_variant, negative=negative_sampling_variant,
                                vector_size=vector_size, window=window, min_count=min_count, epochs=epochs,
                                sample=sample, workers=7)

        # Sample of train_enabled corpus:
        d2v_model.build_vocab(train_corpus)
        d2v_model.train(train_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
        d2v_model.save(model_path)

    else:
        # Loading Doc2Vec iDNES.cz doc2vec_model from saved doc2vec_model file
        d2v_model = Doc2Vec.load(model_path)

    path_to_cropped_wordsim_file = 'stats/word2vec/similarities/WordSim353-cs-cropped.tsv'
    if os.path.exists(path_to_cropped_wordsim_file):
        word_pairs_eval = d2v_model.wv.evaluate_word_pairs(
            path_to_cropped_wordsim_file)
    else:
        save_wordsim(path_to_cropped_wordsim_file)
        word_pairs_eval = d2v_model.wv.evaluate_word_pairs(path_to_cropped_wordsim_file)

    overall_score, _ = d2v_model.wv.evaluate_word_analogies('stats/word2vec/analogies/questions-words-cs.txt')
    # Analogies tuning of doc2vec_model:

    doc_id = random.randint(0, len(test_corpus) - 1)
    logging.debug("print(test_corpus[:2])")
    logging.debug(train_corpus[:2])
    logging.debug("print(test_corpus[:2])")
    logging.debug(test_corpus[:2])
    inferred_vector = d2v_model.infer_vector(test_corpus[doc_id])
    sims = d2v_model.dv.most_similar([inferred_vector], topn=len(d2v_model.dv))
    # Compare and print the most/median/least similar documents from the train_enabled train_corpus
    # 'SIMILAR/DISSIMILAR DOCS PER MODEL
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

    return word_pairs_eval, overall_score


def create_dictionary_from_mongo_idnes(sentences=None, force_update=False, filter_extremes=False):
    # a memory-friendly iterator
    path_to_train_dict = 'precalc_vectors/dictionary_train_idnes.gensim'
    if os.path.isfile(path_to_train_dict) is False or force_update is True:
        if sentences is None:
            sentences = build_sentences()

        sentences_train, _ = train_test_split(sentences, train_size=0.2, shuffle=True)
        # Creating _dictionary...
        preprocessed_dictionary_train = gensim.corpora.Dictionary(line for line in sentences_train)
        del sentences
        gc.collect()
        if filter_extremes is True:
            preprocessed_dictionary_train.filter_extremes()
        # Saving _dictionary...
        preprocessed_dictionary_train.save(path_to_train_dict)
        # Dictionary save
        return preprocessed_dictionary_train
    else:
        loaded_dict = corpora.Dictionary.load(path_to_train_dict)
        return loaded_dict


def init_and_start_training(model, tagged_data, max_epochs):
    model.build_vocab(tagged_data)

    for _ in range(max_epochs):
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
    return model


def most_similar(model, search_term):
    inferred_vector = model.infer_vector(search_term)
    sims = model.docvecs.most_similar([inferred_vector], topn=20)

    res = []
    for elem in sims:
        inner = {'index': elem[0], 'distance': elem[1]}
        res.append(inner)

    return res[:20]


def prepare_train_test_corpus(source='idnes'):
    client = MongoClient("localhost", 27017, maxPoolSize=50)
    if source == "idnes":
        db = client.idnes
    elif source == "cswiki":
        db = client.cswiki
    else:
        raise ValueError("No from selected sources are in options.")

    sentences = build_sentences()
    collection = db.preprocessed_articles_trigrams

    cursor = collection.find({})
    for document in cursor:
        sentences.append(document['text'])
    train_corpus, test_corpus = train_test_split(sentences, test_size=0.2, shuffle=True)
    train_corpus = list(create_tagged_document(sentences))

    return train_corpus, test_corpus


def train(tagged_data):
    # E.g., hardcoded stats.
    max_epochs = 20
    vec_size = 8
    alpha = 0.025
    """
    E.g., Other possible stats:
    minimum_alpha = 0.0025
    reduce_alpha = 0.0002
    """

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_count=0,
                    dm=0)

    model = init_and_start_training(model, tagged_data, max_epochs)

    if "PYTEST_CURRENT_TEST" in os.environ:
        path_to_save = Path(TESTING_MODEL_LOCATION)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        model.save(path_to_save.as_posix())
    else:
        model.save("models/d2v_mini_vectors.model")
        print("Doc2Vec model Saved")

# HERE WAS a full text training. ABANDONED DUE TO: no longer needed
# HERE WAS a Doc2Vec tuning and final training. ABANDONED DUE TO: unknown bug in test calculations giving a null result


def train_doc2vec(documents_all_features_preprocessed, create_csv=False):
    print("documents_all_features_preprocessed")
    print(documents_all_features_preprocessed)

    tagged_data = []
    for i, doc in enumerate(documents_all_features_preprocessed):
        selected_list = []
        for word in doc.split(", "):
            # if not word in all_stopwords:
            words_preprocessed = gensim.utils.simple_preprocess(preprocess(word))
            for sublist in words_preprocessed:
                if len(sublist) > 0:
                    selected_list.append(sublist)

        # Preprocessing doc.
        tagged_data.append(TaggedDocument(words=selected_list, tags=[str(i)]))
        if create_csv is True:
            # Will append to exising file! CSV needs to be removed first if needs to be up updated as a whole
            with open("testing_datasets/idnes_preprocessed.txt", "a+", encoding="utf-8") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(selected_list)

    train(tagged_data)


def get_similar_by_posts_slug(most_similar_items, documents_slugs, number_of_recommended_posts):

    post_recommendations = pd.DataFrame()
    list_of_article_slugs = []
    list_of_coefficients = []

    most_similar_items = most_similar_items[1:number_of_recommended_posts]

    logging.debug("most_similar_items:")
    logging.debug(most_similar_items)
    logging.debug("len(most_similar_items):")
    logging.debug(len(most_similar_items))
    logging.debug("len(documents_slugs)")
    logging.debug(len(documents_slugs))

    for i in range(0, len(most_similar_items)):
        logging.debug("most_similar_items[index][1]]")
        logging.debug(most_similar_items[i][1])
        logging.debug("documents_slugs[int(most_similar_items[index][0])]")
        logging.debug(documents_slugs[int(most_similar_items[i][0])])
        list_of_article_slugs.append(documents_slugs[int(most_similar_items[i][0])])
        list_of_coefficients.append(most_similar_items[i][1])

    post_recommendations['slug'] = list_of_article_slugs
    post_recommendations['coefficient'] = list_of_coefficients

    posts_dict = post_recommendations.to_dict('records')

    list_of_articles = [posts_dict.copy()]
    return flatten(list_of_articles)


class Doc2VecClass:

    def __init__(self):
        self.documents = None
        self.df = None
        self.posts_df = None
        self.categories_df = None
        self.database = DatabaseMethods()
        self.doc2vec_model = None

    def prepare_posts_df(self):
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        self.posts_df = self.posts_df.rename({'title': 'post_title'})
        return self.posts_df

    def prepare_categories_df(self):
        self.categories_df = self.database.get_categories_dataframe()
        self.posts_df = self.posts_df.rename({'title': 'category_title'})
        return self.categories_df

    def rename_slug(self, posts_from_cache, recommender_methods):
        self.df = (recommender_methods
                   .get_posts_categories_dataframe(from_cache=posts_from_cache))
        if 'post_slug' in self.df.columns:
            self.df = self.df.rename(columns={'post_slug': 'slug'})
        if 'slug_x' in self.df.columns:
            self.df = self.df.rename(columns={'slug_x': 'slug'})

    def modify_posts_df(self, posts_from_cache, searched_slug, train_enabled, limited,
                        full_text, number_of_recommended_posts):

        recommender_methods = RecommenderMethods()

        self.rename_slug(posts_from_cache, recommender_methods)
        if searched_slug not in self.df['slug'].to_list():
            if config.trials_counter.NUM_OF_TRIALS < 1:
                print('Slug does not appear in dataframe. Refreshing datafreme of posts.')
                recommender_methods.get_posts_dataframe(force_update=True)
                self.df = recommender_methods.get_posts_categories_dataframe(from_cache=True)
                config.trials_counter.NUM_OF_TRIALS += 1
                self.get_similar_doc2vec(searched_slug, train_enabled, limited, number_of_recommended_posts,
                                         full_text, posts_from_cache)
            else:
                config.trials_counter.NUM_OF_TRIALS = 0
                raise ValueError("searched_slug not in dataframe. Tried to deal with this by updating posts_categories "
                                 "df but didn't helped")

    @check_empty_string
    def get_similar_doc2vec(self, searched_slug, train_enabled=False, limited=True, number_of_recommended_posts=21,
                            full_text=False, posts_from_cache=True):

        if type(searched_slug) is not str:
            raise ValueError("Bad searched_slug parameter inserted.")

        verify_searched_slug_sanity(searched_slug)

        self.modify_posts_df(posts_from_cache, searched_slug, train_enabled,
                             limited, full_text, number_of_recommended_posts)

        config.trials_counter.NUM_OF_TRIALS = 0

        if full_text is False:
            cols = ['keywords', 'all_features_preprocessed']
        else:
            cols = ['keywords', 'all_features_preprocessed', 'body_preprocessed']

        documents_all_features_preprocessed = preprocess_columns(self.df, cols)
        gc.collect()

        if 'post_slug' in self.df:
            self.df = self.df.rename(columns={'post_slug': 'slug'})
        documents_slugs = self.df['slug'].tolist()

        if train_enabled is True:
            train_doc2vec(documents_all_features_preprocessed, create_csv=False)

        del documents_all_features_preprocessed
        gc.collect()

        doc2vec_loaded_model = load_doc2vec_model()

        recommender_methods = RecommenderMethods()
        post_found = recommender_methods.find_post_by_slug(searched_slug)
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")

        if full_text is False:
            tokens = keywords_preprocessed + all_features_preprocessed
        else:
            full_text = post_found.iloc[0]['body_preprocessed'].split(" ")
            tokens = keywords_preprocessed + all_features_preprocessed + full_text

        vector_source = doc2vec_loaded_model.infer_vector(tokens)
        most_similar_posts = doc2vec_loaded_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)

        try:
            recommendations = get_similar_by_posts_slug(most_similar_posts, documents_slugs,
                                                        number_of_recommended_posts)
            config.trials_counter.NUM_OF_TRIALS = 0
        except IndexError as e:
            if config.trials_counter.NUM_OF_TRIALS < 1:
                documents_all_features_preprocessed = preprocess_columns(self.df, cols)
                train_doc2vec(documents_all_features_preprocessed)
                config.trials_counter.NUM_OF_TRIALS += 1
                recommendations = self.get_similar_doc2vec(searched_slug, train_enabled, limited,
                                                           number_of_recommended_posts,
                                                           full_text, posts_from_cache)
            else:
                logging.warning("Tried to train Doc2Vec again but it didn't helped and IndexError got raised again. "
                                "Need to shutdown,")
                raise e

        return recommendations

    @accepts_first_argument(str)
    @check_empty_string
    def get_similar_doc2vec_with_full_text(self, searched_slug, train_enabled=False, number_of_recommended_posts=21,
                                           posts_from_cache=True):

        recommender_methods = RecommenderMethods()
        self.df = recommender_methods.get_posts_categories_dataframe(from_cache=posts_from_cache)

        if 'post_slug' in self.df.columns:
            self.df = self.df.rename(columns={'post_slug': 'slug'})
        if 'slug_x' in self.df.columns:
            self.df = self.df.rename(columns={'slug_x': 'slug'})

        if searched_slug not in self.df['slug'].to_list():
            # ** HERE WAS A HANDLING OF THIS ERROR BY UPDATING POSTS_CATEGORIES DF.
            # ABANDONED DUE TO MASKING OF ERROR FOR BAD INPUT **
            raise ValueError("searched_slug not in dataframe")

        cols = ['keywords', 'all_features_preprocessed', 'body_preprocessed']
        documents_all_features_preprocessed = preprocess_columns(self.df, cols)

        gc.collect()

        documents_slugs = self.df['slug'].tolist()

        if train_enabled is True:
            train_doc2vec(documents_all_features_preprocessed)
        del documents_all_features_preprocessed
        gc.collect()

        if "PYTEST_CURRENT_TEST" in os.environ:
            doc2vec_loaded_model = Doc2Vec.load(TESTING_MODEL_LOCATION)
        else:
            doc2vec_loaded_model = Doc2Vec.load("models/d2v_full_text_limited.model")

        recommend_methods = RecommenderMethods()

        # not necessary
        post_found = recommend_methods.find_post_by_slug(searched_slug)
        # IndexError: single positional indexer is out-of-bounds
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")
        full_text = post_found.iloc[0]['body_preprocessed'].split(" ")
        tokens = keywords_preprocessed + all_features_preprocessed + full_text
        vector_source = doc2vec_loaded_model.infer_vector(tokens)

        most_similar_items = doc2vec_loaded_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)

        try:
            recommendations = get_similar_by_posts_slug(most_similar_items, documents_slugs,
                                                        number_of_recommended_posts)
            config.trials_counter.NUM_OF_TRIALS = 0

        except IndexError as e:
            if config.trials_counter.NUM_OF_TRIALS < 1:
                logging.warning('Index error occurred when trying to get Doc2Vec model for posts')
                logging.warning(e)
                logging.info('Trying to deal with this by retraining Doc2Vec...')
                logging.debug('Preparing test features')
                documents_all_features_preprocessed = preprocess_columns(self.df, cols)
                train_doc2vec(documents_all_features_preprocessed)
                config.trials_counter.NUM_OF_TRIALS += 1
                recommendations = self.get_similar_doc2vec_with_full_text(searched_slug, train_enabled,
                                                                          number_of_recommended_posts,
                                                                          posts_from_cache)
            else:
                logging.warning(
                    "Tried to train Doc2Vec again but it didn't helped and IndexError got raised again. Need to "
                    "shutdown.")
                raise e

        return recommendations

    def get_vector_representation(self, searched_slug):
        """
        For Learn-to-Rank
        """
        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a input_string.")
        else:
            if searched_slug == "":
                raise ValueError("Entered input_string is empty.")

        recommender_methods = RecommenderMethods()
        return self.doc2vec_model.infer_vector(recommender_methods.find_post_by_slug(searched_slug))

    @DeprecationWarning
    def create_or_update_corpus_and_dict_from_mongo_idnes(self):
        dict_idnes = create_dictionary_from_mongo_idnes(force_update=True)
        return dict_idnes

    def get_pair_similarity_doc2vec(self, slug_1, slug_2, d2v_model=None):

        logging.debug('Calculating Doc2Vec pair similarity for posts:')
        logging.debug(slug_1)
        logging.debug(slug_2)

        if d2v_model is None:
            d2v_model = self.load_model()

        recommend_methods = RecommenderMethods()
        post_1 = recommend_methods.find_post_by_slug(slug_1)
        post_2 = recommend_methods.find_post_by_slug(slug_2)

        feature_1 = 'all_features_preprocessed'
        feature_2 = 'title'

        first_text = post_1[feature_2].iloc[0] + ' ' + post_1[feature_1].iloc[0]
        second_text = post_2[feature_2].iloc[0] + ' ' + post_2[feature_1].iloc[0]

        logging.debug(first_text)
        logging.debug(second_text)

        vec1 = d2v_model.infer_vector(first_text.split())
        vec2 = d2v_model.infer_vector(second_text.split())

        cos_distance = spatial.distance.cosine(vec1, vec2)
        logging.debug("cos_distance:")
        logging.debug(cos_distance)

        return cos_distance

    def load_model(self):
        self.doc2vec_model = load_doc2vec_model()
        return self.doc2vec_model
