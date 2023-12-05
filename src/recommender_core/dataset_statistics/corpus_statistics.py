import string
from collections import Counter

import pandas as pd
from nltk import FreqDist

from src.recommender_core.data_handling.data_queries import RecommenderMethods


def most_common_words():
    texts = []

    recommender_methods = RecommenderMethods()
    all_posts_df = recommender_methods.get_posts_dataframe()
    pd.set_option('display.max_columns', None)

    all_posts_df['whole_article'] = all_posts_df['title'] + all_posts_df['excerpt'] + all_posts_df['full_text']

    for line in all_posts_df['whole_article']:
        if type(line) is str:
            try:
                texts.append(line)
            except Exception as e:
                print("Exception occurredf:")
                print(e)
                pass

    texts_joined = ' '.join(texts)

    print("Removing punctuation...")
    texts_joined = texts_joined.translate(str.maketrans('', '', string.punctuation))
    texts_joined = texts_joined.replace(',', '')
    texts_joined = texts_joined.replace('„', '')
    texts_joined = texts_joined.replace('“', '')
    # split() returns list of all the words in the input_string
    split_it = texts_joined.split()

    # Pass the split_it list to instance of Counter class.
    counters_found = Counter(split_it)
    # print(Counters)

    # most_common() produces k frequently encountered
    # input values and their respective counts.
    most_occur = counters_found.most_common(100)
    print("TOP 100 WORDS:")
    for word in most_occur:
        print(word[0])

# ***HERE WAS also idnes statistics. ABANDONED DUE TO: no longer needer

class CorpusStatistics:

    @staticmethod
    def most_common_words_from_supplied_words(all_words):
        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)
        k = 150
        return zip(*fdist.most_common(k))
