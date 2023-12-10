import gensim
from pathlib import Path

from typing import List, Union

from src.data_handling.data_tools import flatten


def get_cz_stopwords_file_path():
    """
    Getter for the Czech stopwords file path.
    @return: string path of the Czech stopwords file.
    """
    cz_stopwords_file_name = Path("src/prefillers/preprocessing/stopwords/czech_stopwords.txt")
    return cz_stopwords_file_name


def get_general_stopwords_file_path():
    """
    Getter for the general stopwords file path.
    @return: string path of the general stopwords file.
    """
    general_stopwords_file_name = Path("src/prefillers/preprocessing/stopwords/general_stopwords.txt")
    return general_stopwords_file_name


def load_cz_stopwords(remove_punct=True):
    """
    Method for loading the Czech stopwords.
    @param remove_punct: boolean value for whether you want to remove punctuation before entering the stopwords removal.
    Depends on how you collect the stopwords. Preferred way for this project is to provide stopwords with removed
    punctuation.
    @return: list of Czech stopwords loaded from file
    """
    with open(get_cz_stopwords_file_path().as_posix(), encoding="utf-8") as file:
        cz_stopwords = file.readlines()
        if remove_punct is False:
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        else:
            cz_stopwords = [gensim.utils.simple_preprocess(line.rstrip()) for line in cz_stopwords]
        return flatten(cz_stopwords)


def load_general_stopwords():
    """
    Method for loading the general stopwords.
    @return: list of general stopwords loaded from file
    """
    with open(get_general_stopwords_file_path().as_posix(), encoding="utf-8") as file:
        general_stopwords = file.readlines()
        general_stopwords = [line.rstrip() for line in general_stopwords]
        return flatten(general_stopwords)


def remove_stopwords(texts: Union[str, List[str]], cz_punct: bool = False) -> List[str]:
    """
    The method that actually handles the stopwords removal.
    @rtype: list of words without stopwords
    """
    stopwords_cz = load_cz_stopwords(cz_punct)
    stopwords_general = load_general_stopwords()
    stopwords = stopwords_cz + stopwords_general
    stopwords = flatten(stopwords)
    cleaned_text_list = []

    if isinstance(texts, list):
        for word in texts:
            if word not in stopwords:
                cleaned_text_list.append(word)

    elif isinstance(texts, str):
        joined_stopwords = ' '.join(str(x) for x in stopwords)
        stopwords = gensim.utils.deaccent(joined_stopwords)
        stopwords = stopwords.split(' ')
        cleaned_text_list = [[word for word in gensim.utils.simple_preprocess(doc)
                              if word not in stopwords] for doc in texts]

    else:
        raise ValueError("'texts' parameter needs to be string or list.")

    return cleaned_text_list
