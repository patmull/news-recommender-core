import string
from collections import Counter

import pandas as pd

from content_based_algorithms.data_queries import RecommenderMethods

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
most_occur = Counters_found.most_common(10)
print("TOP 10 WORDS:")
print(most_occur)
