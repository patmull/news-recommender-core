import os.path
import pickle
import re
import traceback

import gensim
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, KeyedVectors
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix, SoftCosineSimilarity
from gensim.similarities.annoy import AnnoyIndexer
from scipy import spatial
from sklearn.feature_extraction.text import HashingVectorizer
from src.content_based_algorithms.gensim_native_models import GensimMethods


def calculate_similarity(source_doc, target_docs=None, threshold=0.2):
    """Calculates & returns similarity scores between given source document & all
    the target documents."""
    if not target_docs:
        return []
    if isinstance(target_docs, str):
        target_docs = [target_docs]

    vectorizer = HashingVectorizer(n_features=20)
    # print("source_vec")
    source_vec = vectorizer.transform([source_doc])
    # print(source_vec)
    results = []
    # print("for doc")
    print("Searching for similar articles...")
    for doc in target_docs:
        doc_without_slug = doc.split(";", 1)  # removing searched_slug
        target_vec = vectorizer.transform([doc_without_slug[0]])
        sim_score = 1 - spatial.distance.cosine(source_vec[0].toarray(), target_vec[0].toarray())
        results = sort_results(sim_score, threshold, doc, results)
    return results


def sort_results(sim_score, threshold, doc, results):
    if sim_score > threshold:
        slug = re.sub(r'^.*?;', ';', doc)  # keeping only searched_slug of the document
        # print("searched_slug.replace")
        slug = slug.replace('; ', '')
        # print("results.append")
        results.append({"slug": slug, "coefficient": sim_score})
    # Sort results by score in desc order
    # print("results.sort")

    return results.sort(key=lambda k: k["coefficient"], reverse=True)


class DocSim:
    def __init__(self, w2v_model=None, stopwords=None):
        self.w2v_model = w2v_model
        self.stopwords = stopwords if stopwords is not None else []

    def calculate_similarity_wiki_model(self, source_doc, target_docs=None, threshold=0.2):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if not target_docs:
            return []
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        # print("source_vec")
        source_vec = self.vectorize(source_doc)
        # print(source_vec)
        results = []
        # print("for doc")
        print("Searching for similar articles...")
        for doc in target_docs:
            doc_without_slug = doc.split(";", 1)  # removing searched_slug
            target_vec = self.vectorize(doc_without_slug[0])
            sim_score = self._cosine_sim(source_vec, target_vec)
            results = sort_results(sim_score=sim_score, results=results, doc=doc, threshold=threshold)

        return results

    def calculate_similarity_wiki_model_gensim(self, source_doc, target_docs=None, threshold=0.2):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        termsim_index = WordEmbeddingSimilarityIndex(self.w2v_model)
        dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_idnes.gensim')
        bow_corpus = pickle.load(open("precalc_vectors/corpus_idnes.pkl", "rb"))
        similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix

        docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=21)
        print("source_doc:")
        sims = self.create_docsim_index(source_doc=source_doc, docsim_index=docsim_index, dictionary=dictionary)

        results = []
        for sim_tuple in sims:
            doc_found = target_docs[sim_tuple[0]]  # get document by position from sims results
            slug = re.sub(r'^.*?;', ';', doc_found)  # keeping only searched_slug of the document
            slug = slug.replace("; ", "")
            sim_score = sim_tuple[1]
            results.append({"slug": slug, "coefficient": sim_score})

        print("results:")
        print(results)
        return results

    def calculate_similarity_idnes_model_gensim(self, source_doc, docsim_index, dictionary, target_docs=None):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        # TO HERE

        sims = self.create_docsim_index(source_doc=source_doc, docsim_index=docsim_index, dictionary=dictionary)
        results = []
        for sim_tuple in sims:
            doc_found = target_docs[sim_tuple[0]]  # get document by position from sims results
            slug = re.sub(r'^.*?;', ';', doc_found)  # keeping only searched_slug of the document
            print("searched_slug:")
            print(slug)
            slug = slug.replace("; ", "")
            sim_score = sim_tuple[1]
            results.append({"slug": slug, "coefficient": sim_score})

        print("results")
        print(results)

        return results

    def load_docsim_index_and_dictionary(self, source, model, force_update=True):
        global path_to_docsim_index, dictionary
        gensim_methods = GensimMethods()
        common_texts = gensim_methods.load_texts()
        # bow_corpus = pickle.load(open("precalc_vectors/corpus_idnes.pkl","rb"))

        global dictionary
        if source == "idnes":
            path_to_docsim_index = "full_models/idnes/docsim_index_idnes"
        elif source == "cswiki":
            path_to_docsim_index = "full_models/cswiki/docsim_index_cswiki"
        else:
            raise ValueError("Bad source name selected")

        if os.path.exists(path_to_docsim_index) and force_update is False:
            docsim_index = SoftCosineSimilarity.load(path_to_docsim_index)
        else:
            print("Docsim index not found or forced to update. Will create a new from available articles.")
            # TODO: This can be preloaded
            docsim_index = self.update_docsim_index(model=model, common_texts=common_texts)
        return docsim_index, dictionary

    def update_docsim_index(self, model, supplied_dictionary=None, common_texts=None, tfidf_corpus=None):
        global dictionary, path_to_folder

        if model == "wiki":
            source = "cswiki"
            self.w2v_model = KeyedVectors.load_word2vec_format("full_models/cswiki/word2vec/w2v_model_full")
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
                raise ValueError("Wrong idnes model name chosen.")
            file_name = "w2v_idnes.model"
            path_to_model = path_to_folder + file_name
            self.w2v_model = KeyedVectors.load(path_to_model)
        else:
            path_to_folder = None
            raise ValueError("Wrong model name chosen.")

        if source == "idnes":
            if supplied_dictionary is None:
                print("Dictionary not supplied. Must load. If this is repeated routine, try to supply dictionary"
                      "to speed up the program.")
                dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_idnes.gensim')
            else:
                dictionary = supplied_dictionary
            docsim_index_path = "full_models/idnes/docsim_index_idnes"
        elif source == "cswiki":
            if supplied_dictionary is None:
                print("Dictionary not supplied. Must load. If this is repeated routine, try to supply dictionary"
                      "to speed up the program.")
                dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary_cswiki.gensim')
            else:
                dictionary = supplied_dictionary
            docsim_index_path = "full_models/cswiki/docsim_index_cswiki"
        else:
            raise ValueError("Bad source name selected")
        print("Updating DocSim index...")
        tfidf = TfidfModel(dictionary=dictionary)
        words = [word for word, count in dictionary.most_common()]

        try:
            word_vectors = self.w2v_model.wv.vectors_for_all(words,
                                                             allow_inference=False)
            # produce vectors for words in train_corpus
        except AttributeError:
            # TODO: This is None Type, found out why!
            try:
                word_vectors = self.w2v_model.vectors_for_all(words,
                                                              allow_inference=False)
            except AttributeError as e:
                print(e)
                print(traceback.format_exc())
                raise AttributeError

        indexer = AnnoyIndexer(word_vectors, num_trees=2)  # use Annoy for faster word similarity lookups
        termsim_index = WordEmbeddingSimilarityIndex(word_vectors, kwargs={'indexer': indexer})  # for similarity index
        similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary,
                                                       tfidf)  # compute word similarities # for docsim_index creation

        if tfidf_corpus is None:
            tfidf_corpus = tfidf[
                [dictionary.doc2bow(document) for document in common_texts]]  # for docsim_index creation
        docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix,
                                            num_best=21)  # index tfidf_corpus        print("source_doc:")
        print("DocSim index saved.")
        docsim_index.save(docsim_index_path)
        return docsim_index

    def vectorize(self, doc: str) -> np.ndarray:
        """
        Identify the vector values for each word in the given document
        :param doc:
        :return:
        """
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        # TODO: Save computed vectors for later use
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn'multi_dimensional_list exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        # TODO: Change to better model_variant. This looks bad
        # https://radimrehurek.com/gensim/similarities/docsim.html#gensim.similarities.docsim.SoftCosineSimilarity
        vector = np.mean(word_vecs, axis=0)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def create_docsim_index(self, source_doc, docsim_index, dictionary):
        print(source_doc)
        source_doc = source_doc.replace(",", "")
        source_doc = source_doc.replace("||", " ")

        source_text = source_doc.split()
        sims = docsim_index[dictionary.doc2bow(source_text)]

        print("sims:")
        print(sims)

        return sims
