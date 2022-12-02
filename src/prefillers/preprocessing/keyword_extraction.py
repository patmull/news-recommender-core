# from src.multi_rake.model_variant import Rake
import logging
from pathlib import Path

import pandas as pd
from multi_rake import Rake
from yake import KeywordExtractor
from summa import keywords as summa_keywords
from itertools import chain, zip_longest

pd.set_option('display.max_columns', None)


def smart_truncate(content, length=250):
    if len(content) <= length:
        return content
    else:
        return ' '.join(content[:length + 1].split(' ')[0:-1])


def get_cleaned_text(list_text_clean):
    text_clean = ' '.join(list_text_clean)
    return text_clean


class SingleDocKeywordExtractor:
    """
    Keyword extraction class for content-based methods.

    """

    def __init__(self, num_of_keywords=5):
        self.list_text_clean = None
        self.text_clean = None
        self.text_raw = None
        self.sentence_splitted = None
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
        self.list_text_clean = self.get_cleaned_list_text(self.text_raw)
        self.text_clean = get_cleaned_text(self.list_text_clean)

    def get_cleaned_list_text(self, raw_text):
        """
        Raw string representation of text preprocessing and conversion to list.

        :param raw_text: string representation of desired text to extract keywords from
        :return: preprocessed list of words
        """
        self.text_raw = raw_text
        logging.debug("self.text_raw")
        logging.debug(self.text_raw)

        list_text = []
        text_clean = raw_text.replace("\n", " ")
        text_clean = text_clean.replace("\'", "")
        text_clean = text_clean.replace(".", "")
        text_clean = text_clean.replace(":", "")
        text_clean = text_clean.replace("(", "")
        text_clean = text_clean.replace(")", "")

        list_text.append(text_clean[0:5000])

        logging.debug("list_text")
        logging.debug(list_text)

        self.sentence_splitted = raw_text.split(" ")
        logging.debug("self.sentence_splitted")
        logging.debug(self.sentence_splitted)
        
        # https://github.com/Alir3z4/stop-words
        k = []
        z = []
        filename = Path("src/prefillers/preprocessing/stopwords/czech_stopwords.txt")
        with open(filename, 'r', encoding='utf-8') as f:
            for word in f:
                word_splitted = word.split('\n')
                k.append(word_splitted[0])

        for u in self.sentence_splitted:
            z.append(u.lower())

        logging.debug("z:")
        logging.debug(z)

        list_text_clean = [t for t in z if t not in k]
        logging.debug("Stopwords removal...")
        return list_text_clean

    def get_keywords_multi_rake(self, text_for_extraction):
        """
        Multi-Rake keywords extractor.

        :param text_for_extraction: string for keyword extraction
        :return: list of extracted keywords
        """
        rake = Rake(language_code='cs')

        keywords_rake = rake.apply(text_for_extraction)

        return keywords_rake[:self.num_of_keywords]

    def get_keywords_summa(self, text_for_extraction):
        """
        Summa keywords extractor.

        :param text_for_extraction: string for keyword extraction
        :return: list of extracted keywords
        """

        if self.text_clean is not None:
            try:
                summa_keywords_extraction = summa_keywords.keywords(text_for_extraction, words=self.num_of_keywords)\
                    .split("\n")
                return summa_keywords_extraction[:self.num_of_keywords]
            except IndexError:
                return " "
        else:
            logging.debug("Variable keywords_extracted empty!")

    def get_keywords_yake(self, string_for_extraction):
        """
        Yake keywords extractor.

        :param text_for_extraction: string for keyword extraction
        :return: list of extracted keywords
        """
        keywords_extracted = []
        kw_extractor = KeywordExtractor(lan="cs", n=1, top=self.num_of_keywords)
        if string_for_extraction:
            keywords_extracted = kw_extractor.extract_keywords(string_for_extraction)[::-1]
        else:
            logging.debug("No string given for extraction!")
        return keywords_extracted

    def get_keywords_combine_all_methods(self, string_for_extraction):
        """
        Applying all available methods to a given string of text.

        @param string_for_extraction: string to extract keywords from
        @return: list of keywords combined from supported keyword extractors
        """

        keywords_multi_rake = self.get_keywords_multi_rake(string_for_extraction)
        keywords_summa = self.get_keywords_summa(string_for_extraction)
        keywords_yake = self.get_keywords_yake(string_for_extraction)

        keywords_multi_rake_only_words = [x[0] for x in keywords_multi_rake]
        keywords_yake_only_words = [y[0] for y in keywords_yake]

        keyword_all_types = []
        logging.debug(keywords_multi_rake_only_words)
        logging.debug(keywords_summa)
        keyword_all_types.append([keywords_multi_rake_only_words, keywords_yake_only_words, keywords_summa])
        keyword_all_types = list(chain.from_iterable(keyword_all_types))

        keyword_all_types = keyword_all_types[:5]
        logging.debug("All algorithms of keyword extraction combined")
        logging.debug(keyword_all_types)
        keyword_all_types_combined = [item for sublist in zip_longest(*keyword_all_types) for item in sublist if
                                      item is not None]

        logging.debug("keyword_all_types_combined")
        logging.debug(keyword_all_types_combined)
        keyword_all_types_splitted = ', '.join(keyword_all_types_combined)
        keyword_all_types_splitted = smart_truncate(keyword_all_types_splitted)
        logging.debug("keyword_all_types_splitted")
        logging.debug(keyword_all_types_splitted)
        return str(keyword_all_types_splitted)


if __name__ == "__main__":
    logging.debug("Keyword extractor.")
