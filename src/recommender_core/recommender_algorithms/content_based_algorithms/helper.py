import json
import numpy as np
import pandas as pd
from gensim.utils import deaccent

from src.prefillers.preprocessing.cz_preprocessing import CzPreprocess


class Helper:
    # helper functions

    def get_id_from_slug(self, slug, df):
        return df[df.slug == slug]["row_num"].values[0]

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
                    czlemma = CzPreprocess()
                    yield czlemma.preprocess(deaccent(text))
            else:
                break

    # https://www.machinelearningplus.com/nlp/gensim-tutorial/
    def generate_lines_from_mmcorpus(self, corpus, max_sentence=-1, preprocess=False):
        for text in corpus:
            if preprocess is False:
                yield text
            if preprocess is True:
                czlemma = CzPreprocess()
                yield czlemma.preprocess(deaccent(text))

    def clear_blank_lines_from_txt(self, file_path):
        new_filename_parts = file_path.split('.')
        new_file_name = new_filename_parts[0] + '_blank_lines_free' + new_filename_parts[1]
        with open(file_path, 'r', encoding='utf-8') as inFile, \
                open(new_file_name, 'w', encoding='utf-8') as outFile:
            for line in inFile:
                if line.strip():
                    outFile.write(line)

    def flatten(self, multi_dimensional_list):
        return [item for sublist in multi_dimensional_list for item in sublist]

    def verify_searched_slug_sanity(self, df, searched_slug):
        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a string.")
        else:
            if searched_slug == "":
                raise ValueError("Entered string is empty.")
            else:
                pass

    def preprocess_columns(self, df, cols):
        documents_df = pd.DataFrame()
        df['all_features_preprocessed'] = df['all_features_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))

        df.fillna("", inplace=True)

        df['body_preprocessed'] = df['body_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))
        documents_df['all_features_preprocessed'] = df[cols].apply(
            lambda row: ' '.join(row.values.astype(str)),
            axis=1)

        documents_df['all_features_preprocessed'] = df['category_title'] + ', ' + documents_df[
            'all_features_preprocessed'] + ", " + df['body_preprocessed']

        documents_all_features_preprocessed = list(
            map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        return documents_all_features_preprocessed


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