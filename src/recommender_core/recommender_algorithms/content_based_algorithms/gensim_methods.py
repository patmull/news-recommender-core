import re
import string
from collections import defaultdict

import pandas as pd
from nltk import RegexpTokenizer

from src.recommender_core.data_handling.data_queries import RecommenderMethods, TfIdfDataHandlers
from src.prefillers.preprocessing.cz_preprocessing import cz_lemma
from src.prefillers.preprocessing.czech_stemmer import cz_stem

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


@DeprecationWarning
def preprocess(sentence, stemming=False, lemma=True):
    # print(sentence)
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    cleantext.translate(str.maketrans('', '', string.punctuation))  # removing punctuation
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    # print("rem_num")
    # print(rem_num)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    # print("tokens")
    # print(tokens)
    if stemming is True:
        edited_words = [cz_stem(w, True) for w in tokens if len(w) > 1]  # aggresive
        edited_words = list(filter(None, edited_words))  # empty strings removal
        tokens = " ".join(edited_words)

    elif lemma is True:
        edited_words = [cz_lemma(w) for w in tokens if len(w) > 1]

        #  (hint: "edited_words_list: List[<type>] = ...")  [var-annotated]
        edited_words_list = list(filter(None, edited_words))  # empty strings removal
        tokens = " ".join(edited_words_list)
        # TODO: error: Need type annotation for "edited_words_list". Priority: LOW
    else:
        return tokens

    return tokens


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

    def find_post_by_slug(self, slug):
        post_dataframe = self.df.loc[self.df['post_slug'] == slug]
        # noinspection PyPep8
        doc = post_dataframe["category_title"] + " " + post_dataframe["keywords"] \
              + " " + post_dataframe["post_title"] + " " + post_dataframe["excerpt"]
        return str(doc.tolist())

    def get_categories_dataframe(self) -> object:
        """

        :return: 
        """
        self.categories_df = self.database.get_categories_dataframe()
        return self.categories_df

    @PendingDeprecationWarning
    def get_fit_by_feature(self, feature_name, second_feature=None):
        tf_idf_data_handlers = TfIdfDataHandlers()
        fit_by_feature = tf_idf_data_handlers.get_tfidf_vectorizer(feature_name, second_feature)
        return fit_by_feature

    def load_texts(self):
        recommender_methods = RecommenderMethods()
        self.posts_df = recommender_methods.get_posts_dataframe()
        self.categories_df = recommender_methods.get_categories_dataframe()
        self.df = recommender_methods.get_posts_categories_dataframe()
        # preprocessing
        # self.df["post_title"] = self.df["post_title"].map(lambda s: self.preprocess(s, stemming=False, lemma=False))
        # self.df["excerpt"] = self.df["excerpt"].map(lambda s: self.preprocess(s, stemming=False, lemma=False))

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

        # print(texts)

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
