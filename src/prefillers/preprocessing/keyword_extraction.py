import logging
from itertools import zip_longest
from pathlib import Path

import pandas as pd
from multi_rake import Rake
from summa import keywords as summa_keywords
from yake import KeywordExtractor

pd.set_option('display.max_columns', None)


def smart_truncate(content, length=250):
    if len(content) <= length:
        return content
    else:
        return ' '.join(content[:length + 1].split(' ')[0:-1])


def get_cleaned_text(list_text_clean):
    text_clean = ' '.join(list_text_clean)
    return text_clean


def get_cleaned_list_text(raw_text):
    """
    Raw string representation of text preprocessing and conversion to list.
    :param raw_text: string representation of desired text to extract keywords from
    :return: preprocessed list of words
    """
    cleaned_text = raw_text.replace("\n", " ")
    cleaned_text = cleaned_text.replace("\'", "")
    cleaned_text = cleaned_text.replace(".", "")
    cleaned_text = cleaned_text.replace(":", "")
    cleaned_text = cleaned_text.replace("(", "")
    cleaned_text = cleaned_text.replace(")", "")

    sentence_split = cleaned_text.split(" ")

    stopwords = []
    stopwords_file = Path("src/prefillers/preprocessing/stopwords/czech_stopwords.txt")
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for word in f:
            word_split = word.split('\n')
            stopwords.append(word_split[0])

    lowercase_words = [word.lower() for word in sentence_split]

    list_text_clean = [word for word in lowercase_words if word not in stopwords]

    return list_text_clean


def load_stopwords():
    filename = Path("src/prefillers/preprocessing/stopwords/czech_stopwords.txt")
    with open(filename, 'r', encoding='utf-8') as f:
        stopwords = [word.strip() for word in f]
    return stopwords


class SingleDocKeywordExtractor:
    """
    Keyword extraction class for content-based methods.
    """

    def __init__(self, num_of_keywords=5):
        self.list_text_clean = None
        self.text_clean = None
        self.text_raw = None
        self.sentence_split = None
        self.num_of_keywords = num_of_keywords

    def set_text(self, text_raw):
        """
        Setter method for the desired text.
        :param text_raw: string of text to extract keywords from.
        :return:
        """
        self.text_raw = text_raw

    def clean_text(self):
        """
        Handles the transformation of the raw text and performance of the text cleaning operations.
        :return:
        """
        self.list_text_clean = get_cleaned_list_text(self.text_raw)
        self.text_clean = get_cleaned_text(self.list_text_clean)

    def get_keywords_multi_rake(self, string_for_extraction):
        """
        Multi-Rake keywords extractor.
        :param string_for_extraction: string for keyword extraction
        :return: list of extracted keywords
        """
        rake = Rake(language_code='cs')
        keywords_rake = rake.apply(string_for_extraction)

        return keywords_rake[:self.num_of_keywords]

    def get_keywords_summa(self, text_for_extraction):
        """
        Summa keywords extractor.
        :param text_for_extraction: string for keyword extraction
        :return: list of extracted keywords
        """
        if self.text_clean is not None:
            try:
                keywords = summa_keywords.keywords(text_for_extraction,
                                                   words=self.num_of_keywords).split("\n")
                return keywords[:self.num_of_keywords]
            except IndexError:
                return []

    def get_keywords_yake(self, string_for_extraction):
        """
        Yake keywords extractor.
        :param string_for_extraction: string for keyword extraction
        :return: list of extracted keywords
        """
        keywords = []
        extractor = KeywordExtractor(lan="cs", n=1, top=self.num_of_keywords)
        if string_for_extraction:
            keywords = extractor.extract_keywords(string_for_extraction)[::-1]
        return keywords

    def get_keywords_combine_all_methods(self, string_for_extraction):
        """
        Applying all available methods to a given string of text.
        @param string_for_extraction: string to extract keywords from
        @return: list of keywords combined from supported keyword extractors
        """
        rake_keywords = self.get_keywords_multi_rake(string_for_extraction)
        _summa_keywords = self.get_keywords_summa(string_for_extraction)
        yake_keywords = self.get_keywords_yake(string_for_extraction)

        rake_only_words = [x[0] for x in rake_keywords]
        yake_only_words = [y[0] for y in yake_keywords]

        combined_keywords = rake_only_words + yake_only_words + _summa_keywords
        combined_keywords = combined_keywords[:5]

        combined_keywords_flat = [item for sublist in zip_longest(*combined_keywords) for item in sublist if
                                  item is not None]

        combined_keywords_str = ', '.join(combined_keywords_flat)
        combined_keywords_str = smart_truncate(combined_keywords_str)

        return combined_keywords_str


if __name__ == "__main__":
    logging.debug("Keyword extractor.")
