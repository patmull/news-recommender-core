import pickle
import re

import gensim
import numpy as np
from gensim.corpora import Dictionary
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix, SoftCosineSimilarity
from scipy import spatial
from sklearn.feature_extraction.text import HashingVectorizer


class DocSim:
    def __init__(self, w2v_model, stopwords=None):
        self.w2v_model = w2v_model
        self.stopwords = stopwords if stopwords is not None else []


    def calculate_similarity_idnes_model(self, source_doc, target_docs=None, threshold=0.2):
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
            # if type(doc) is str:
            # print("doc_without_slug")
            doc_without_slug = doc.split(";", 1) # removing slug
            # print("target_vec")
            target_vec = vectorizer.transform([doc_without_slug[0]])
            # print(target_vec)
            # print("source_vec")
            # print(source_vec)
            sim_score = 1 - spatial.distance.cosine(source_vec[0].toarray(), target_vec[0].toarray())
            # print("if sim_score > threshold")
            if sim_score > threshold:
                # print("type(doc)")
                # print(type(doc))
                # print("slug = re.sub(...)")
                slug = re.sub(r'^.*?;', ';', doc) # keeping only slug of the document
                # print("slug.replace")
                slug = slug.replace('; ','')
                # print("results.append")
                results.append({"slug": slug, "coefficient": sim_score})
            # Sort results by score in desc order
            # print("results.sort")

            results.sort(key=lambda k: k["coefficient"], reverse=True)
            # else:
              #  continue

        return results


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
            # if type(doc) is str:
            # print("doc_without_slug")
            doc_without_slug = doc.split(";", 1) # removing slug
            # print("target_vec")
            target_vec = self.vectorize(doc_without_slug)
            # print(target_vec)
            # print("source_vec")
            # print(source_vec)
            sim_score = self._cosine_sim(source_vec, target_vec)
            # sim_score = 1 - spatial.distance.cosine(source_vec[0].toarray(), target_vec[0].toarray())
            # print("if sim_score > threshold")
            if sim_score > threshold:
                # print("type(doc)")
                # print(type(doc))
                # print("slug = re.sub(...)")
                slug = re.sub(r'^.*?;', ';', doc) # keeping only slug of the document
                # print("slug.replace")
                slug = slug.replace('; ','')
                # print("results.append")
                results.append({"slug": slug, "coefficient": sim_score})
            # Sort results by score in desc order
            # print("results.sort")

            results.sort(key=lambda k: k["coefficient"], reverse=True)
            # else:
              #  continue

        return results

    def calculate_similarity_wiki_model_gensim(self, source_doc, target_docs=None, threshold=0.2):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        termsim_index = WordEmbeddingSimilarityIndex(self.w2v_model)
        dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary.gensim')
        bow_corpus = pickle.load(open("precalc_vectors/corpus.pkl","rb"))
        similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix

        docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=21)
        print("source_doc:")
        print(source_doc)
        source_doc = source_doc.replace(",","")
        source_doc = source_doc.replace("||", " ")

        source_text = source_doc.split()
        sims = docsim_index[dictionary.doc2bow(source_text)]

        print("sims:")
        print(sims)

        results = []
        for sim_tuple in sims:
            doc_found = target_docs[sim_tuple[0]] # get document by position from sims results
            slug = re.sub(r'^.*?;', ';', doc_found) # keeping only slug of the document
            slug = slug.replace("; ","")
            sim_score = sim_tuple[1]
            results.append({"slug": slug, "coefficient": sim_score})

        print("results:")
        print(results)
        return results

    def calculate_similarity_idnes_model_gensim(self, source_doc, target_docs=None, threshold=0.2):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        termsim_index = WordEmbeddingSimilarityIndex(self.w2v_model.wv)
        dictionary = gensim.corpora.Dictionary.load('precalc_vectors/dictionary.gensim')
        bow_corpus = pickle.load(open("precalc_vectors/corpus.pkl","rb"))
        similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix

        docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=21)
        print("source_doc:")
        print(source_doc)
        source_doc = source_doc.replace(",","")
        source_doc = source_doc.replace("||", " ")

        source_text = source_doc.split()
        sims = docsim_index[dictionary.doc2bow(source_text)]

        print("sims:")
        print(sims)

        results = []
        for sim_tuple in sims:
            doc_found = target_docs[sim_tuple[0]] # get document by position from sims results
            slug = re.sub(r'^.*?;', ';', doc_found) # keeping only slug of the document
            print("slug:")
            print(slug)
            slug = slug.replace("; ","")
            sim_score = sim_tuple[1]
            results.append({"slug": slug, "coefficient": sim_score})

        print("results")
        print(results)

        return results


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
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        # TODO: Change to better method. This looks bad
        # https://radimrehurek.com/gensim/similarities/docsim.html#gensim.similarities.docsim.SoftCosineSimilarity
        vector = np.mean(word_vecs, axis=0)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim