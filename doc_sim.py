import re
import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import HashingVectorizer

class DocSim:
    def __init__(self, w2v_model, stopwords=None):
        self.w2v_model = w2v_model
        self.stopwords = stopwords if stopwords is not None else []

    def calculate_similarity(self, source_doc, target_docs=None, threshold=0.2):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if not target_docs:
            return []
        print("isinstance")
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        vectorizer = HashingVectorizer(n_features=20)
        print("source_vec")
        source_vec = vectorizer.transform([source_doc])
        print(source_vec)
        results = []
        print("for doc")
        for doc in target_docs:
            # if type(doc) is str:
            print("doc_without_slug")
            doc_without_slug = doc.split(";", 1) # removing slug
            print("target_vec")
            target_vec = vectorizer.transform([doc_without_slug[0]])
            print(target_vec)
            print("source_vec")
            print(source_vec)
            sim_score = 1 - spatial.distance.cosine(source_vec[0].toarray(), target_vec[0].toarray())
            print("if sim_score > threshold")
            if sim_score > threshold:
                # print("type(doc)")
                # print(type(doc))
                print("slug = re.sub(...)")
                slug = re.sub(r'^.*?;', ';', doc) # keeping only slug of the document
                print("slug.replace")
                slug = slug.replace('; ','')
                print("results.append")
                results.append({"slug": slug, "coefficient": sim_score})
            # Sort results by score in desc order
            print("results.sort")
            results.sort(key=lambda k: k["coefficient"], reverse=True)
            # else:
              #  continue

        return results
