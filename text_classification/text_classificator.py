import csv
import datetime
import itertools
import time
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from text_classification.czech_stemmer import cz_stem

words = []
classes = []
documents = []
ignore_words = ['?',',','.',':','!']

# english stemmer
stemmer = LancasterStemmer()

training_data = []

# open file in read mode
with open('articles_categories_recommender_full.csv', 'r', encoding='utf-8') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
    header = next(read_obj).strip().split(';')
    csv_dict_reader = csv.DictReader((l.lower() for l in read_obj), delimiter=';', fieldnames=header)
    # iterate over each line as a ordered dictionary

    #for row in itertools.islice(csv_dict_reader,1000):
    for row in csv_dict_reader:
        if not row['category_title'].isnumeric() or not row['post_title'].isnumeric():
            training_data.append({"class": row['category_title'],
                              "sentence": row['post_title']})

# manual training data
"""
training_data.append(
    {"class": "hry", "sentence": "Milujete kávu? V Espresso Tycoon se budete starat o vlastní kavárnu"})
training_data.append(
    {"class": "hry", "sentence": "Vrací se Želvy Ninja, v nové videohře vyzvou samotného Trhače"})
training_data.append(
    {"class": "hry", "sentence": "Do prodeje míří plyšák podle nejdivnějšího hrdiny z Mass Effectu"})
training_data.append(
    {"class": "hry", "sentence": "Prodeje hororu Amnesia: Rebirth přesáhly sto tisíc, na zisk to nestačí"})

training_data.append({"class": "věda a technologie",
                      "sentence": "Brzy uvidíme nový supersmartphone pro geeky. Jací byli jeho předchůdci?"})
training_data.append({"class": "věda a technologie",
                      "sentence": "iPhony čeká designová změna. Apple se má inspirovat androidí konkurencí"})
training_data.append(
    {"class": "věda a technologie", "sentence": "Pomůžou na postcovidový syndrom vakcíny? Zdá se to možné"})
training_data.append({"class": "věda a technologie",
                      "sentence": "Život v budovách nám nesvědčí, říká vědkyně, která vymyslela rovnici stresu"})

training_data.append(
    {"class": "sport",
     "sentence": "ONLINE: Pardubice zachraňují sérii, doma vedou 3:1. Vzepře se Hradec Tygrům?"})
training_data.append(
    {"class": "sport", "sentence": "FAČR o Slavii a Rangers: Věříme v objektivní vyšetření, odmítáme rasismus"})
training_data.append(
    {"class": "sport", "sentence": "První časovka v Quick-Stepu. Černý v Katalánsku vedl, nakonec byl osmý"})
training_data.append(
    {"class": "sport", "sentence": "Malý krok pro člověka, ale velký skok pro golf. Skončilo to před 50 lety"})
training_data.append(
    {"class": "sport", "sentence": "Johannes Bö porazil Lägreida. Potřetí v řadě je velký glóbus jeho"})

training_data.append(
    {"class": "domácí",
     "sentence": "Muž se vyhýbal vězení. Když na sebe upozornil přestupkem, zkusil hlídce ujet"})
training_data.append(
    {"class": "domácí", "sentence": "VIDEO: Jezevčík se zřítil do sklepa, strážnice ho vylákala až díky šunce"})
training_data.append(
    {"class": "domácí", "sentence": "Třicet dnů rozhodně ne, řekl Bartoš. Opozici svorně vadí uzávěra okresů"})
training_data.append({"class": "domácí", "sentence": "Zemřela první dáma českého šansonu Hana Hegerová"})

"""
print("%s sentences in training data" % len(training_data))


# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to list of words
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add list of classes
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# lower each word and remove duplicates
words = [cz_stem(w.lower()) for w in words if w not in ignore_words and not w.isdigit()]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)

training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # finding root of each word by czech language word stemmer
    pattern_words = [cz_stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # 0 for each word and 1 when words matches with other words in training data
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
print([cz_stem(word.lower()) for word in w])
print(training[i])
print(output[i])

# sigmoid function
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to derivative
# derivative of sigmoid function for backpropagation: using gradient descent in order to find an optimal set of model
# parameters in order to minimize a loss function
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

# input clean up
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [cz_stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))


def think(sentence, show_details=False):

    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


def train(X, y, hidden_neurons=10, alpha=2, epochs=50000, dropout=False, dropout_percent=0.5):
    print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
        hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))

    print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs + 1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        if (dropout):
            layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                1.0 / (1 - dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j % 10000) == 0 and j > 5000:
            # if this iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                break

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if (j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
               }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True, ensure_ascii=False)
    print("saving synapses to json file...", synapse_file)


X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=2, epochs=10000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print("Processing time:", elapsed_time, "seconds")

# probability threshold
ERROR_THRESHOLD = 0.2

# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])


def classify(sentence, show_details=False):

    results = think(sentence, show_details)

    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    print("%s \n classification: %s" % (sentence, return_results))
    return return_results


classify("FAČR o Slavii a Rangers: Věříme v objektivní vyšetření, odmítáme rasismus")
classify("Leicester ve čtvrtfinále Anglického poháru zdolal United, postoupily i City a Chelsea")
classify("Real má po půli dvoubrankový náskok nad Liverpoolem, City vede o gól")
classify("Na co zírá mašinfíra: projeďte si nádhernou trať v údolí nespoutané Opavy")
classify("Nová značka se zvláštním jménem koupila největší smartphonový propadák")
classify("Čína zveřejnila své smělé cíle. Plánuje letos ekonomický růst o šest procent")
classify("Respirátory či dvě roušky povinně od čtvrtka. Odbory chtěly lockdown")
