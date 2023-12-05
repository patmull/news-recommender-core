import re
import string

import majka
from html2text import html2text
from nltk import RegexpTokenizer

from src.prefillers.preprocessing.stopwords_loading import load_cz_stopwords, load_general_stopwords

cz_stopwords = load_cz_stopwords()
general_stopwords = load_general_stopwords()


def cz_lemma(input_string, json=False):
    """
    Czech lemmatization of the words. Uses the Majka morphological dictionary to create the accurate unified
    representation of the similar Czech words in different inflections and grammatical forms.
    :param input_string: string to lemmatize
    :param json: specify whether should return JSON or nor since some of the methods may need it. Defaults to False
    :return: list of lemmatized words or JSON of lemmatozed words
    """
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

    ls = morph.find(input_string)

    if json is not True:
        if not ls:
            return input_string
        else:
            return str(ls[0]['lemma'])
    else:
        return ls


def preprocess(sentence):
    """
        Global way of text preprocessing used for content-based or hybrid methods.
        All the operations it does currently:
        1. Convert sentence input to string (for making sure it is really string before it starts to use methods
        that needs strings).
        2. Convert the input to lower case.
        3. Replace 'new line' characters and 'return' characters.
        4. Replace HTML markup.
        5. Remove leftover punctuation
        6. Removing references tags, http(s) links.
        7. Removing numbers (since they usually don't add any value to content-based methods).
        8. Removing single characters
        9. Removing any leftovers digits or unwanted characters.
        10. CZ lemmatization according to Majka morphological dictionary
        11. Removing the empty string characters that can occur after lemmatization, i.e. if word is not found in dictionary.
        12. Removing the Czech stopwords
        13. Remving the general stopwords
        14. Joining the final text to string
        :param sentence: input string for preprocessing
        :return: preprocessed string
        """
    sentence = sentence
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('\r\n', ' ')
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    cleantext.translate(str.maketrans('', '', string.punctuation))  # removing punctuation

    cleantext = cleantext.split('=References=')[0]  # remove references and everything afterwards
    cleantext = html2text(cleantext).lower()  # remove HTML tags, convert to lowercase

    # 'ToktokTokenizer' does divide by '|' and '\n', but retaining this
    #   statement seems to improve its speed a little
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)

    tokens = [w for w in tokens if '=' not in w]  # remove remaining tags and the like
    string_punctuation = list(string.punctuation)
    # remove tokens that are all punctuation
    tokens = [w for w in tokens if not all(x.isdigit() or x in string_punctuation for x in w)]
    tokens = [w.strip(string.punctuation) for w in tokens]  # remove stray punctuation attached to words
    tokens = [w for w in tokens if len(w) > 1]  # remove single characters
    tokens = [w for w in tokens if not any(x.isdigit() for x in w)]  # remove everything with a digit in it

    edited_words = [cz_lemma(w) for w in tokens]
    edited_words = list(filter(None, edited_words))  # empty strings removal

    # removing stopwords
    edited_words = [word for word in edited_words if word not in cz_stopwords]
    edited_words = [word for word in edited_words if word not in general_stopwords]

    return " ".join(edited_words)


def preprocess_feature(feature_text):
    post_excerpt_preprocessed = preprocess(feature_text)
    return post_excerpt_preprocessed
