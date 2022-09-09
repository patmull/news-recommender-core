import string
from collections import Counter

import pandas as pd
from nltk import FreqDist

from src.recommender_core.data_handling.data_queries import RecommenderMethods


class CorpusStatistics:

    @staticmethod
    def most_common_words_from_supplied_words(all_words):
        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)
        k = 150
        return zip(*fdist.most_common(k))

    def most_common_words(self):
        texts = []

        recommender_methods = RecommenderMethods()
        all_posts_df = recommender_methods.get_posts_dataframe()
        pd.set_option('display.max_columns', None)
        print("all_posts_df")
        print(all_posts_df)


        all_posts_df['whole_article'] = all_posts_df['title'] + all_posts_df['excerpt'] + all_posts_df['full_text']

        print("selected_features_dataframe['whole_article']")
        print(all_posts_df['whole_article'])

        for line in all_posts_df['whole_article']:
            print("line:")
            print(line)
            if type(line) is str:
                try:
                    texts.append(line)
                except Exception:
                    pass


        print("texts[10]:")
        print(texts[10])

        texts_joined = ' '.join(texts)

        print("Removing punctuation...")
        texts_joined = texts_joined.translate(str.maketrans('', '', string.punctuation))
        texts_joined = texts_joined.replace(',','')
        texts_joined = texts_joined.replace('„','')
        texts_joined = texts_joined.replace('“','')
        print("texts_joined[1:10000]:")
        print(texts_joined[0:10000])


        # split() returns list of all the words in the string
        split_it = texts_joined.split()

        # Pass the split_it list to instance of Counter class.
        Counters_found = Counter(split_it)
        #print(Counters)

        # most_common() produces k frequently encountered
        # input values and their respective counts.
        most_occur = Counters_found.most_common(100)
        print("TOP 100 WORDS:")
        for word in most_occur:
            print(word[0])


    def categories_count(self):
        recommender_methods = RecommenderMethods()
        # recommender_methods.get_posts_dataframe()
        # recommender_methods.get_categories_dataframe()
        posts_categories_df = recommender_methods.join_posts_ratings_categories()

        df_freq_stats = posts_categories_df['category_title'].value_counts().reset_index()
        df_freq_stats.columns = ['category_title', 'count']
        print(df_freq_stats)
        df_freq_stats.to_csv("research/corpus_stats/idnes_corpus_freq.csv")


    def overall_stats_idnes(self):
        recommender_methods = RecommenderMethods()
        # number of docs
        # mean document length
        # number of token
        # preprocessing (yes or not)
        # stopwords removal(yes or not)
        posts_categories_df = recommender_methods.join_posts_ratings_categories()
        number_of_docs = len(posts_categories_df.index)
        mean_document_length = posts_categories_df['all_features_preprocessed'].apply(len).mean()
        # number_of_tokens = # too big time complexity, not that relevant anyway
        preprocessing = "YES"
        stopwords_removal = "YES"

        result_df = pd.DataFrame([[number_of_docs, mean_document_length, preprocessing, stopwords_removal]], columns=['number_of_docs', 'mean_document_length', 'preprocessing', 'stopwords_removal'])
        """
        result_df['number_of_docs'] = number_of_docs
        result_df['mean_document_length'] = mean_document_length
        result_df['preprocessing'] = preprocessing
        result_df['stopwords_removal'] = stopwords_removal
        """
        print(result_df)
        # noinspection PyTypeChecker
        result_df.to_csv("research/corpus_stats/corpus_overall_stats_idnes.csv")


    def print_overall_stats_idnes(self):
        self.overall_stats_idnes()
