from content_based_algorithms.word2vec import Word2VecClass

word2vecClass = Word2VecClass()
# 1. Create Dictionary
word2vecClass.create_dictionary_from_dataframe(force_update=False, filter_extremes=False)
# preprocess corpus and save it to mongo
word2vecClass.preprocess_idnes_corpus()