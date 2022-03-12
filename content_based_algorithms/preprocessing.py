def load_stopwords():
    filename = "cz_stemmer/czech_stopwords.txt"
    with open(filename, encoding="utf-8") as file:
        global cz_stopwords
        cz_stopwords = file.readlines()
        cz_stopwords = [line.rstrip() for line in cz_stopwords]