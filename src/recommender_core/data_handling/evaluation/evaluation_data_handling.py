import gensim
import pandas as pd

from src.recommender_core.recommender_algorithms.content_based_algorithms.gensim_methods import preprocess


def save_wordsim(path_to_cropped_wordsim_file):
    df = pd.read_csv('research/word2vec/similarities/WordSim353-cs.csv',
                     usecols=['cs_word_1', 'cs_word_2', 'cs mean'])
    df['cs_word_1'] = df['cs_word_1'].apply(lambda x: gensim.utils.deaccent(preprocess(x)))
    df['cs_word_2'] = df['cs_word_2'].apply(lambda x: gensim.utils.deaccent(preprocess(x)))

    df.to_csv(path_to_cropped_wordsim_file, sep='\t', encoding='utf-8', index=False)


