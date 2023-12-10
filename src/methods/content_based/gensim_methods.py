from collections import defaultdict

import pandas as pd
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


class GensimMethods:

    def __init__(self):
        self.posts_df = None
        self.categories_df = None
        self.df = pd.DataFrame()
        self.database = DatabaseMethods()
        self.documents = None

    def get_posts_dataframe(self):
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    def join_posts_ratings_categories(self):
        """
        :rtype: object

        """
        if self.posts_df is not None:
            self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='searched_id')
            # clean up from unnecessary columns
            self.df = self.df[
                ['post_id', 'post_title', 'slug', 'excerpt', 'body', 'views', 'keywords', 'category_title',
                 'description']]
        else:
            raise ValueError("Datafame is set to None. Cannot continue with next operation.")

    def get_categories_dataframe(self) -> object:
        """

        :return: 
        """
        self.categories_df = self.database.get_categories_dataframe()
        return self.categories_df

    def load_texts(self):
        recommender_methods = RecommenderMethods()
        self.posts_df = recommender_methods.get_posts_dataframe()
        self.categories_df = recommender_methods.get_categories_dataframe()
        self.df = recommender_methods.get_posts_categories_dataframe()

        # converting pandas columns to list of lists and through map to list of input_string joined by space ' '
        self.df[['trigrams_full_text']] = self.df[['trigrams_full_text']].fillna('')
        self.documents = list(map(' '.join, self.df[["trigrams_full_text"]].values.tolist()))

        cz_stopwords_filepath = "src/prefillers/preprocessing/stopwords/czech_stopwords.txt"
        with open(cz_stopwords_filepath, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)
        texts = [
            [word for word in document.lower().split() if word not in cz_stopwords and len(word) > 1]
            for document in self.documents
        ]

        # remove words that appear only once
        frequency = defaultdict(int)  # type: defaultdict
        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [
            [token for token in text if frequency[token] > 1]
            for text in texts
        ]

        return texts
