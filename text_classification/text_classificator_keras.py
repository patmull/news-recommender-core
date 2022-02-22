import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt

# plt.style.use('ggplot')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

from tensorflow.python import keras
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
import keras.preprocessing.text
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.text import Tokenizer



"""
filepath_dict = {
    'Auto moto': 'data_sets/Auto_moto.csv',
    'Celebrity': 'data_sets/Celebrity.csv',
    'Ekonomika': 'data_sets/Ekonomika.csv',
    'Finance': 'data_sets/Finance.csv',
    'Hry':'data_sets/Hry.csv',
    'Móda':'data_sets/Moda.csv',
    'Ostatní':'data_sets/Ostatni.csv',
    'Regionalní':'data_sets/Regionalni.csv',
    'Sport':'data_sets/Sport.csv',
    'Technologie':'data_sets/Technologie.csv',
    'Věda':'data_sets/Veda.csv',
    'Vztahy':'data_sets/Vztahy.csv',
    'Zdraví':'data_sets/Zdravi.csv',
    'Zprávy domáci':'data_sets/Zpravy_domaci.csv',
    'Zprávy z kultury':'data_sets/Zpravy_z_kultury.csv'
}
"""

filepath_dict = {
    'iDNES': 'data_sets/idnes.csv',
}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['title', 'category'], sep=';')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)

sentences = ("„Létaly střepy a padal strop. Zalehl jsem děti a modlil se“")

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
print("vectorizer.vocabulary_")
print(vectorizer.vocabulary_)
print(vectorizer.transform(sentences).toarray())

print("df")
print(df)

df_economy = df[df['source'] == 'iDNES']

print(df_economy)
print("df_zpravy_domaci['title'].values")
print(df_economy['title'].values)

sentences = df_economy['title'].values
y = df_economy['category'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.1, random_state=2000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
print("X_train")
print(X_train)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score)

prediction = classifier.predict(np.array([sentences]))
