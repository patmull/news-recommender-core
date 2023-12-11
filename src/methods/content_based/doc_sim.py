import logging
import os.path
import pickle
import re
import traceback
from pathlib import Path

import gensim
import numpy as np
from gensim.models import KeyedVectors
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix, SoftCosineSimilarity
from scipy import spatial
from sklearn.feature_extraction.text import HashingVectorizer
from src.methods.content_based.gensim_methods import GensimMethods
from src.methods.content_based.tfidf import TfIdf

REGEX_STRING_UNTIL_SEMICOLON = redis_instance'^.*?;'


def calculate_similarity(source_doc, target_docs=None, threshold=0.2):
    """Calculates & returns similarity scores between given _source document & all
    the target documents."""
    if not target_docs:
        return []
    if isinstance(target_docs, str):
        target_docs = [target_docs]

    vectorizer = HashingVectorizer(n_features=20)
    source_vec = vectorizer.transform([source_doc])
    results = []
    # Searching for similar articles...
    for doc in target_docs:
        doc_without_slug = doc.split(";", 1)  # removing searched_slug
        target_vec = vectorizer.transform([doc_without_slug[0]])
        sim_score = 1 - spatial.distance.cosine(source_vec[0].toarray(), target_vec[0].toarray())
        results = sort_results(sim_score, threshold, doc, results)
    return results


def sort_results(sim_score, threshold, doc, results):
    if sim_score > threshold:
        slug = re.sub(redis_instance'', ';', doc)  # keeping only searched_slug of the document
        slug = slug.replace('; ', '')
        results.append({"slug": slug, "coefficient": sim_score})
    # Sort results by score in desc order
    return results.sort(key=lambda k: k["coefficient"], reverse=True)


# noinspection PyPep8
def _cosine_sim(vec_a, vec_b):
    """Find the cosine similarity distance between two vectors."""
    csim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if np.isnan(np.sum(csim)):
        return 0
    return csim


def create_docsim_index(source_doc, docsim_index, dictionary):
    source_doc = source_doc.replace(",", "")
    source_doc = source_doc.replace("||", " ")

    source_text = source_doc.split()
    sims = docsim_index[dictionary.doc2bow(source_text)]

    return sims


def create_results(source_doc, docsim_index, dictionary, target_docs):
    sims = create_docsim_index(source_doc=source_doc, docsim_index=docsim_index, dictionary=dictionary)
    results = []
    for sim_tuple in sims:
        doc_found = target_docs[sim_tuple[0]]  # get document by position from sims results
        slug = re.sub(REGEX_STRING_UNTIL_SEMICOLON, ';', doc_found)  # keeping only searched_slug of the document
        slug = slug.replace("; ", "")
        sim_score = sim_tuple[1]
        results.append({"slug": slug, "coefficient": sim_score})

    return results


def calculate_similarity_idnes_model_gensim(source_doc, docsim_index, dictionary, target_docs):
    """Calculates & returns similarity scores between given _source document & all
    the target documents."""
    # TO HERE
    return create_results(source_doc, docsim_index, dictionary, target_docs)


def load_docsim_index(source, model_name, force_update=True):
    gensim_methods = GensimMethods()
    common_texts = gensim_methods.load_texts()

    if source == "idnes":
        path_to_docsim_index = Path("full_models/idnes/docsim_index_idnes")
    elif source == "cswiki":
        path_to_docsim_index = Path("full_models/cswiki/docsim_index_cswiki")
    else:
        raise ValueError("Bad _source name selected")

    if os.path.exists(path_to_docsim_index.as_posix()) and force_update is False:
        docsim_index = SoftCosineSimilarity.load(path_to_docsim_index.as_posix())
    else:
        logging.info("Docsim index not found or forced to update. Will create a new from available articles.")
        docsim_index = update_docsim_index(model=model_name, common_texts=common_texts)
    return docsim_index


class DocSim:
    def __init__(self, w2v_model=None, stopwords=None):
        self.w2v_model = w2v_model
        self.stopwords = stopwords if stopwords is not None else []

    def calculate_similarity_wiki_model_gensim(self, source_doc, target_docs=None):
        """Calculates & returns similarity scores between given _source document & all
        the target documents."""
        termsim_index = WordEmbeddingSimilarityIndex(self.w2v_model)
        # WARNING: cswiki may not be in disk
        dictionary = gensim.corpora.Dictionary.load('precalc_vectors/word2vec/dictionary_cswiki.gensim')
        bow_corpus = pickle.load(open("precalc_vectors/word2vec/corpus_idnes.pkl", "rb"))
        similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix

        docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=21)
        # source_doc:
        # noinspection PyTypeChecker
        return create_results(source_doc, docsim_index, dictionary, target_docs)

    def load_model(self, model):
        if model == "cswiki":
            source = "cswiki"
            return KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full"), source
        elif model.startswith("idnes"):
            idnes_model_paths = {
                "idnes_1": "full_models/idnes/evaluated_models/word2vec_model_1/",
                "idnes_2": "full_models/idnes/evaluated_models/word2vec_model_2_default_parameters/",
                "idnes_3": "full_models/idnes/evaluated_models/word2vec_model_3/",
                "idnes_4": "full_models/idnes/evaluated_models/word2vec_model_4/",
                "idnes": "models/"
            }

            if model.startswith("idnes"):
                path_to_folder = idnes_model_paths.get(model, "models/")
                file_name = "w2v_idnes.model"
                path_to_model = path_to_folder + file_name
                self.w2v_model = KeyedVectors.load(path_to_model)
        else:
            raise ValueError("Wrong model name chosen.")


def load_dictionary(source, supplied_dictionary):
    if source == "idnes":
        if supplied_dictionary is None:
            logging.warning("Dictionary not supplied. Must load. If this is repeated routine, try to supply _dictionary "
                            "to speed up the program.")
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/word2vec/dictionary_idnes.gensim')
        else:
            dictionary = supplied_dictionary
        docsim_index_path = "full_models/idnes/docsim_index_idnes"
    elif source == "cswiki":
        if supplied_dictionary is None:
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/word2vec/dictionary_cswiki.gensim')
        else:
            dictionary = supplied_dictionary
        docsim_index_path = "full_models/cswiki/docsim_index_cswiki"
    else:
        raise ValueError("Bad _source name selected")
    return dictionary, docsim_index_path


# *** HERE was also a simple vectorize() method_name with averaging the vector value


def update_docsim_index(model, supplied_dictionary=None, common_texts=None, tfidf_corpus=None):
    if model == "cswiki":
        source = "cswiki"
        w2v_model = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
    elif model.startswith("idnes"):
        source = "idnes"
        if model.startswith("idnes_1"):
            path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_1/"
        elif model.startswith("idnes_2"):
            path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_2_default_parameters/"
        elif model.startswith("idnes_3"):
            path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_3/"
        elif model.startswith("idnes_4"):
            path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_4/"
        elif model == "idnes":
            path_to_folder = "models/"
        else:
            raise ValueError("Wrong idnes model name chosen.")
        file_name = "w2v_idnes.model"
        path_to_model = path_to_folder + file_name
        w2v_model = KeyedVectors.load(path_to_model)
    else:
        raise ValueError("Wrong model name chosen.")

    if source == "idnes":
        if supplied_dictionary is None:
            print("Dictionary not supplied. Must load. If this is repeated routine, try to supply _dictionary"
                  "to speed up the program.")
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/word2vec/dictionary_idnes.gensim')
        else:
            dictionary = supplied_dictionary
        docsim_index_path = "full_models/idnes/docsim_index_idnes"
    elif source == "cswiki":
        if supplied_dictionary is None:
            print("Dictionary not supplied. Must load. If this is repeated routine, try to supply _dictionary"
                  "to speed up the program.")
            dictionary = gensim.corpora.Dictionary.load('precalc_vectors/word2vec/dictionary_cswiki.gensim')
        else:
            dictionary = supplied_dictionary
        docsim_index_path = "full_models/cswiki/docsim_index_cswiki"
    else:
        raise ValueError("Bad source name selected")

    tfidf = TfIdf()
    words = [word for word, count in dictionary.most_common()]

    try:
        word_vectors = w2v_model.wv.vectors_for_all(words, allow_inference=False)
        # produce vectors for words in train_corpus
    except AttributeError:
        try:
            word_vectors = w2v_model.vectors_for_all(words, allow_inference=False)
        except AttributeError as e:
            print(e)
            print(traceback.format_exc())
            raise AttributeError

    from gensim.similarities.annoy import AnnoyIndexer

    indexer = AnnoyIndexer(word_vectors, num_trees=2)  # use Annoy for faster word similarity lookups
    termsim_index = WordEmbeddingSimilarityIndex(word_vectors, kwargs={'indexer': indexer})  # for similarity index
    similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
    # compute word similarities # for docsim_index creation

    if tfidf_corpus is None:
        tfidf_corpus = tfidf[
            [dictionary.doc2bow(document) for document in common_texts]]  # for docsim_index creation
    # index tfidf_corpus
    docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix, num_best=21)
    print("DocSim index saved.")
    docsim_index.save(docsim_index_path)
    return docsim_index
