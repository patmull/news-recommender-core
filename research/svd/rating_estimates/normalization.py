import sys

import pandas as pd
from matplotlib import pyplot as plt


def load_data():
    categories_views = pd.read_csv('research/svd/rating_estimates/news-readers-behaviour-mapped-to-idnes-categories.csv',
                                   delimiter=';')
    print(categories_views.to_string())
    original_df = categories_views
    categories_views = categories_views.fillna(0)
    num_cols = categories_views.columns.values.tolist()
    num_cols.remove('kategorie')
    categories_views = categories_views[num_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    categories_views = categories_views * 5.0
    normalized_df = categories_views.round(2)
    normalized_df['kategorie'] = original_df['kategorie']
    print(normalized_df.to_string())
    normalized_df = normalized_df.set_index('kategorie')
    print(normalized_df.to_string())
    for num_column in num_cols:
        normalized_df[num_column].plot(kind="bar")
        plt.title(num_column)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()

class Normalizer:
    pass
