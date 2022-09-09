from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass, \
    create_dictionary_from_dataframe, preprocess_idnes_corpus

word2vecClass = Word2VecClass()
# 1. Create Dictionary
create_dictionary_from_dataframe(force_update=False)
# preprocess train_corpus and save it to mongo
preprocess_idnes_corpus()