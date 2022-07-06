import re, string
import gensim
from nltk import RegexpTokenizer
import majka
from html2text import html2text
import content_based_algorithms.data_queries as data_queries
from content_based_algorithms.data_queries import RecommenderMethods
from data_connection import Database
from gensim.utils import deaccent

cz_stopwords = data_queries.load_cz_stopwords()
general_stopwords = data_queries.load_general_stopwords()


class CzPreprocess:

    def __init__(self):
        self.df = None
        self.categories_df = None

    # pre-worked
    def preprocess(self, sentence, stemming=False, lemma=True):
        # print(sentence)
        sentence = sentence
        sentence = str(sentence)
        sentence = sentence.lower()
        sentence = sentence.replace('\r\n', ' ')
        print(sentence)
        sentence = sentence.replace('{html}', "")
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        cleantext.translate(str.maketrans('', '', string.punctuation))  # removing punctuation

        a_string = cleantext.split('=References=')[0]  # remove references and everything afterwards
        a_string = html2text(a_string).lower()  # remove HTML tags, convert to lowercase
        a_string = re.sub(r'https?:\/\/.*?[\s]', '', a_string)  # remove URLs

        # 'ToktokTokenizer' does divide by '|' and '\n', but retaining this
        #   statement seems to improve its speed a little
        a_string = a_string.replace('|', ' ').replace('\n', ' ')

        rem_url = re.sub(r'http\S+', '', cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        # print("rem_num")
        # print(rem_num)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)
        # print("tokens")
        # print(tokens)

        tokens = [w for w in tokens if '=' not in w]  # remove remaining tags and the like
        string_punctuation = list(string.punctuation)
        tokens = [w for w in tokens if not
        all(x.isdigit() or x in string_punctuation for x in w)] # remove tokens that are all punctuation
        tokens = [w.strip(string.punctuation) for w in tokens]  # remove stray punctuation attached to words
        tokens = [w for w in tokens if len(w) > 1]  # remove single characters
        tokens = [w for w in tokens if not any(x.isdigit() for x in w)]  # remove everything with a digit in it

        edited_words = [self.cz_lemma(w) for w in tokens]
        edited_words = list(filter(None, edited_words))  # empty strings removal

        # removing stopwords
        edited_words = [word for word in edited_words if word not in cz_stopwords]
        edited_words = [word for word in edited_words if word not in general_stopwords]

        return " ".join(edited_words)

    def preprocess_single_post_find_by_slug(self, slug, json=False, stemming=False):
        recommenderMethods = RecommenderMethods()
        post_dataframe = recommenderMethods.find_post_by_slug(slug)
        post_dataframe["title"] = post_dataframe["title"].map(lambda s: self.preprocess(s, stemming))
        post_dataframe["excerpt"] = post_dataframe["excerpt"].map(lambda s: self.preprocess(s, stemming))
        if json is False:
            return post_dataframe
        else:
            return recommenderMethods.convert_df_to_json(post_dataframe)

    def preprocess_feature(self, feature_text, stemming=False):
        post_excerpt_preprocessed = self.preprocess(feature_text, stemming)
        return post_excerpt_preprocessed

    def cz_lemma(self, string, json=False):
        morph = majka.Majka('morphological_database/majka.w-lt')

        morph.flags |= majka.ADD_DIACRITICS  # find word forms with diacritics
        morph.flags |= majka.DISALLOW_LOWERCASE  # do not enable to find lowercase variants
        morph.flags |= majka.IGNORE_CASE  # ignore the word case whatsoever
        morph.flags = 0  # unset all flags

        morph.tags = False  # return just the lemma, do not process the tags
        morph.tags = True  # turn tag processing back on (default)

        morph.compact_tag = True  # return tag in compact form (as returned by Majka)
        morph.compact_tag = False  # do not return compact tag (default)

        morph.first_only = True  # return only the first entry
        morph.first_only = False  # return all entries (default)

        morph.tags = False
        morph.first_only = True
        morph.negative = "ne"

        ls = morph.find(string)

        if json is not True:
            if not ls:
                return string
            else:
                # # print(ls[0]['lemma'])
                return str(ls[0]['lemma'])
        else:
            return ls


def main():
    cz_lemma = CzPreprocess()


if __name__ == "__main__": main()
