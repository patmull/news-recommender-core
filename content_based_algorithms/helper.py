import json

import numpy as np

import cz_lemmatization
from gensim.utils import deaccent

class Helper:
    # helper functions

    def get_id_from_title(self, title, df):
        return self.df[df.title == title]["row_num"].values[0]

    def get_id_from_slug(self, slug, df):
        return self.df[df.slug == slug]["row_num"].values[0]

    def get_title_from_id(self, id, df):
        data_frame_row = df.loc[df['row_num'] == id]
        return data_frame_row["title"].values[0]

    def get_slug_from_id(self, id, df):
        data_frame_row = df.loc[df['row_num'] == id]
        return data_frame_row["slug"].values[0]

    def generate_lines_from_corpus(self, corpus, max_sentence=-1, preprocess=False):
        for index, text in enumerate(corpus.get_texts()):
            if index < max_sentence or max_sentence == -1:
                if preprocess is False:
                    yield text
                if preprocess is True:
                    czlemma = cz_lemmatization.CzLemma()
                    yield czlemma.preprocess(deaccent(text))
            else:
                break


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)