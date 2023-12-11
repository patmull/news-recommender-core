import json

import numpy as np
from gensim.utils import deaccent

from src.prefillers.preprocessing.czech_preprocessing import preprocess


# https://www.machinelearningplus.com/nlp/gensim-tutorial/
def get_id_from_slug(slug, df):
    return df[df.slug == slug]["row_num"].values[0]


def get_title_from_id(searched_id, df):
    data_frame_row = df.loc[df['row_num'] == searched_id]
    return data_frame_row["title"].values[0]


def get_slug_from_id(searched_id, df):
    data_frame_row = df.loc[df['row_num'] == searched_id]
    return data_frame_row["slug"].values[0]


def generate_lines_from_corpus(corpus, max_sentence=-1, use_preprocessing=False):
    for index, text in enumerate(corpus.get_texts()):
        if index < max_sentence or max_sentence == -1:
            if use_preprocessing is False:
                yield text
            if use_preprocessing is True:
                yield preprocess(deaccent(text))
        else:
            break


def generate_lines_from_mmcorpus(corpus, use_preprocessing=False):
    for text in corpus:
        if use_preprocessing is False:
            yield text
        if use_preprocessing is True:
            yield preprocess(deaccent(text))


def clear_blank_lines_from_txt(file_path):
    new_filename_parts = file_path.split('.')
    new_file_name = new_filename_parts[0] + '_blank_lines_free' + new_filename_parts[1]
    with open(file_path, 'redis_instance', encoding='utf-8') as inFile, \
            open(new_file_name, 'w', encoding='utf-8') as outFile:
        for line in inFile:
            if line.strip():
                outFile.write(line)


def verify_searched_slug_sanity(searched_slug):
    if type(searched_slug) is not str:
        raise ValueError("Entered slug must be a input_string.")
    else:
        if searched_slug == "":
            raise ValueError("Entered input_string is empty.")


class NumpyEncoder(json.JSONEncoder):
    """ Special supplied_json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
