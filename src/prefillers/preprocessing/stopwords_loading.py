
import gensim
from pathlib import Path

from src.recommender_core.data_handling.data_handlers import flatten


def get_cz_stopwords_file_path():
    cz_stopwords_file_name = Path("src/prefillers/preprocessing/stopwords/czech_stopwords.txt")
    return cz_stopwords_file_name


def get_general_stopwords_file_path():
    general_stopwords_file_name = Path("src/prefillers/preprocessing/stopwords/general_stopwords.txt")
    return general_stopwords_file_name


def load_cz_stopwords(remove_punct=True):
    with open(get_cz_stopwords_file_path(), encoding="utf-8") as file:
        cz_stopwords = file.readlines()
        if remove_punct is False:
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        else:
            cz_stopwords = [gensim.utils.simple_preprocess(line.rstrip()) for line in cz_stopwords]
        return flatten(cz_stopwords)


def load_general_stopwords():
    with open(get_general_stopwords_file_path(), encoding="utf-8") as file:
        general_stopwords = file.readlines()
        general_stopwords = [line.rstrip() for line in general_stopwords]
        return flatten(general_stopwords)


def remove_stopwords(texts, cz_punct=False):
    if type(texts) is list:
        cleaned_text_list = []
        stopwords_cz = load_cz_stopwords(cz_punct)
        stopwords_general = load_general_stopwords()
        stopwords = stopwords_cz + stopwords_general
        stopwords = flatten(stopwords)
        for word in texts:
            if word not in stopwords:
                cleaned_text_list.append(word)
        return cleaned_text_list
    elif type(texts) is str:
        stopwords_cz = load_cz_stopwords()
        stopwords_general = load_general_stopwords()
        stopwords = stopwords_cz + stopwords_general
        stopwords = flatten(stopwords)
        joined_stopwords = ' '.join(str(x) for x in stopwords)
        stopwords = gensim.utils.deaccent(joined_stopwords)
        stopwords = stopwords.split(' ')
        return [[word for word in gensim.utils.simple_preprocess(doc) if word not in stopwords] for doc in texts]

