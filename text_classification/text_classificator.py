import csv
import datetime
import itertools
import time
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from cz_stemmer.czech_stemmer import cz_stem

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


def think(sentence, synapse_0, synapse_1, show_details=False):

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

"""
X = np.array(training)
y = np.array(output)

start_time = time.time()
train(X, y, hidden_neurons=20, alpha=2, epochs=10000, dropout=False, dropout_percent=0.2)
elapsed_time = time.time() - start_time
print("Processing time:", elapsed_time, "seconds")
"""
# probability threshold
ERROR_THRESHOLD = 0.2

def load_synapse_file():
    # load our calculated synapse values
    synapse_file = 'synapses.json'
    with open(synapse_file) as data_file:
        synapse = json.load(data_file)
        synapse_0 = np.asarray(synapse['synapse0'])
        synapse_1 = np.asarray(synapse['synapse1'])

    return synapse_0, synapse_1

def classify(sentence, synapse_0, synapse_1,show_details=False):

    results = think(sentence, show_details, synapse_0, synapse_1)

    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    print("%s \n classification: %s" % (sentence, return_results))
    return return_results

synapse_0, synapse_1 = load_synapse_file()

classify("Popularita Netflixu a dalších služeb prospívá i Česku, míní šéfka fondu. Vítězi pandemie jsou streamovací platformy Netflix, Amazon nebo Apple TV, konstatuje šéfka Státního fondu kinematografie Helena Bezděk Fraňková v Rozstřelu. Poptávka předplatitelů po jejich původních seriálech i filmech stoupá, což se následně zrcadlí i ve zvýšeném zájmu zahraničních produkcí natáčet je v Česku. Osm set milionů korun, které má fond k dispozici na filmové pobídky, by tak letos stejně jako v roce 2019 nemuselo stačit. „Všichni jsou zavření doma, nemohou do kina ani do divadla, tak stoupá počet předplatitelů streamovacích služeb. A když už si je někdo předplatí, vyžaduje neustále nové seriály a filmy,“ uvádí ředitelka fondu. / "
        "Zahraniční filmaře proto pandemie v důsledku nezastavila, ale spíše ještě nakopla. Zájem o filmové pobídky poskytované Státním fondem kinematografie je proto velký. Filmovou pobídku Helena Bezděk Fraňková zjednodušeně vysvětluje jako „dvacetiprocentní slevu na všechno české“. „Poskytujeme ji tak, že ještě než sem vůbec produkce přijede, producent si najde tuzemskou servisní producentskou firmu, která výrobu filmu nebo seriálu zajistí a současně pošle Fondu scénář daného projektu. Odborná komise si jej přečte, zhodnotí, a buď mu dá, nebo nedá značku kulturního produktu. / "
        "„Na základě toho pak producent předloží výrobní rozpočet, my se podíváme, co je z předložených nákladů české, a dáme na to dvacet procent slevu. Podstatné je však říct, že už v tu chvíli musíme mít dané finanční prostředky ze státního rozpočtu na fondu připraveny, abychom je konkrétnímu projektu zavázali. Pokud finanční prostředky nemáme, zákonem je zakázáno cokoli filmové produkci slibovat. Vyplatit je pak můžeme v momentě, kdy produkce dotočí a producent předloží audit toho, co za výrobu v ČR utratil,“ vysvětluje /"
        "se Carnival Row. A zatímco loni 800 milionů na filmové pobídky Fondu stačilo, letos tomu tak být nemusí a podle Bezděk Fraňkové ani nebude. Znovu se chce do Česka vrátit štáb seriálu Carnival Row, který zde utratil už přes tři miliardy korun a pokračovat v natáčení plánuje i producent seriálu The Wheel of Time, ve kterém hraje herečka Rosamund Pike. Ze svého dočasného pražského domova dokonce nedávno na dálku převzala Zlatý glóbus za roli ve filmu Jako v bavlnce. / "
        "Šéfku fondu tak čekají vyjednávání s ministerstvem kultury a financí o navýšení rozpočtu filmových pobídek. Odmítat nové projekty by Bezděk Fraňková činila nerada. „Odmítáte tím totiž zároveň práci pro místní, a to je ještě teď v době pandemie zcela špatné. Firmy i lidi potřebují práci. Většina filmů se navíc natáčí v regionech, takže štáby bydlí po hotelech a pro ty je výhodné, že mohou alespoň někoho ubytovat. Do toho samozřejmě produkce odebírají jídlo z místních restaurací, art departmenty zaměstnávají malé firmy pro výrobu různých rekvizit z kovu či dřeva, nakupují se věci z kosmetického a textilního průmyslu a podobně,“ vysvětluje. / "
        "V roce 2019 zahraniční filmaři v Česku utratili přes devět miliard korun. Hlavním konkurentem České republiky v rámci filmových pobídek je Maďarsko. Poskytuje pobídky ve výši 30 procent a činí tak odlišným způsobem než Státní fond kinematografie. „Je to něco jako daňová asignace, forma praktičtější pro jejich fond, trochu komplikovanější pro producenta, ale dává tím Maďarsku v podstatě neomezený rozpočet,“ podotýká Fraňková. Sama by v Česku ráda prosadila zvýšení pobídek na 25 procent, s tím však přímo souvisí i nutné navýšení rozpočtu. / "
        "V tuto chvíli jsou pro Helenu Bezděk Fraňkovou nejpodstatnějšími tématy Národní plán obnovy, ze kterého by fond mohl získat dostatek nových financí na přeměnu fondu kinematografie na fond audiovize. „Začínají se totiž zcela stírat hranice mezi tím, jestli jdeme do kina, nebo se doma díváme na obrazovku. Podporovat bychom navíc chtěli i gaming,“ prozrazuje.",
         synapse_0, synapse_1)
"""
classify("Leicester ve čtvrtfinále Anglického poháru zdolal United, postoupily i City a Chelsea")
classify("Real má po půli dvoubrankový náskok nad Liverpoolem, City vede o gól")
classify("Na co zírá mašinfíra: projeďte si nádhernou trať v údolí nespoutané Opavy")
classify("Nová značka se zvláštním jménem koupila největší smartphonový propadák")
classify("Čína zveřejnila své smělé cíle. Plánuje letos ekonomický růst o šest procent")
classify("Respirátory či dvě roušky povinně od čtvrtka. Odbory chtěly lockdown")
"""