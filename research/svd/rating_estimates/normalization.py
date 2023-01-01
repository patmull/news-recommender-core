import pandas as pd


def load_data():
    categories_views = pd.read_csv('research/svd/rating_estimates/news-readers-behaviour-mapped-to-idnes-categories.csv',
                                   on_bad_lines='skip')
    print(categories_views.info)


class Normalizer:
    pass
