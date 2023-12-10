from gensim.models import Doc2Vec

DEFAULT_MODEL_LOCATION = "models/d2v_limited.model"


def load_doc2vec_model(path_to_model=None):
    # Loading Doc2Vec model...
    if path_to_model is None:
        doc2vec_model = Doc2Vec.load(DEFAULT_MODEL_LOCATION)
    else:
        doc2vec_model = Doc2Vec.load(path_to_model)
    return doc2vec_model
