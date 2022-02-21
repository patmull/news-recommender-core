import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin")

import csv
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
"""
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
"""
#STOPWORDS = set(stopwords.words('english'))


STOPWORDS = set(open("czech_stopwords.txt", encoding="utf-8").read().splitlines())
vocab_size = 5000 # make the top list of words (common words)
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' # OOV = Out of Vocabulary
training_portion = .8
articles = []
labels = []

with open("data_sets/idnes_all_excerpt_category_all.csv", 'r', encoding="utf-8") as csvfile:
#with open("data_sets/bbc-text.csv", 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader) # skipping header
    for row in reader:
        # print(row)
        labels.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)

print(len(labels))
print(len(articles))

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

print("train_size", train_size)
print("train_articles", len(train_articles))
print("train_labels", len(train_labels))
print("validation_articles", len(validation_articles))
print("validation_labels", len(validation_labels))

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print("train_padded")
print(train_padded)

print(set(labels))

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(label_tokenizer.word_index)

model = Sequential()

model.add(Embedding(vocab_size,embedding_dim))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dense(32,activation='softmax'))

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)


num_epochs = 10

print(train_padded)
print(training_label_seq)
print(validation_padded)
print(validation_label_seq)
history = model.fit(train_padded, training_label_seq, validation_data=(validation_padded, validation_label_seq), epochs=num_epochs, verbose=2)


#txt = ["Vrtulník letecké záchranné služby v pátek odpoledne musel vzlétnout do Bohumína na Karvinsku. Dvouleté dítě tam vypadlo z okna v prvním patře a vážně se zranilo. Podle informací ČTK později dítě v nemocnici zemřelo, policejní mluvčí to potvrdila."]
txt = ["Velice povedenými výkony se v základní skupině evropského šampionátu prezentovali fotbalisté Itálie. V osmifinále narazí ve Wembley na Rakousko, které může pouze překvapit. Utkání startuje ve 21 hodin, sledovat jej můžete prostřednictvím podrobné online reportáže minutu po minutě."]

seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
labels = ['zdravi', 'ragionalni', 'domaci', 'vztahy', 'automoto', 'ostatni', 'sport', 'ekonomika', 'finance', 'moda', 'celebrity', 'technologie', 'kultura', 'hry', 'veda', 'zahranici']

print(pred)
print(np.argmax(pred))
print(labels[np.argmax(pred)-1])


