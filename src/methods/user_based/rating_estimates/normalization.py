import random
import sys

import gensim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.recommender_core.data_handling.data_queries import RecommenderMethods


def load_rating_estimation():
    categories_views = pd.read_csv(
        'stats/svd/rating_estimates/news-readers-behaviour-mapped-to-idnes-categories.csv',
        delimiter=';')
    print(categories_views.to_string())
    original_df = categories_views
    # categories_views = categories_views.fillna()
    print("Normalizing...")
    print("Removing non-numerical categories column...")
    num_cols = categories_views.columns.values.tolist()
    num_cols.remove('kategorie')
    categories_views = categories_views[num_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    categories_views = categories_views * 5.0
    normalized_df = categories_views.round(2)
    print("Adding back categories...")
    normalized_df['kategorie'] = original_df['kategorie']
    print(normalized_df.to_string())
    normalized_df = normalized_df.set_index('kategorie')
    normalized_df = normalized_df.where(normalized_df > 1.0, 1.0)
    print("Adding 1.0 to dataset where value < 1.0")
    print(normalized_df.to_string())
    filter_cols = [col for col in normalized_df if col.startswith('Reuters')]
    for filter_col in filter_cols:
        normalized_df[filter_col]['moda'] = np.nan
    print("Adding back NaN...")
    print(normalized_df.to_string())

    return normalized_df


def show_categories_plots(df):
    num_cols = df.columns.values.tolist()
    num_cols.remove('kategorie')
    for num_column in num_cols:
        df[num_column].plot(kind="bar")
        plt.title(num_column)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()


def choose_value_for_random_post():
    rating_estimation_df = load_rating_estimation()
    recommender_methods = RecommenderMethods()
    posts_categories_df = recommender_methods.get_posts_categories_dataframe()

    while True:
        random_post = posts_categories_df.sample()
        # preprocessing categories names to macth the dataset
        random_post_slug = random_post['slug'].iloc[0]
        random_post_category_title = random_post['category_title'].iloc[0]
        random_post_category_title = gensim.utils.deaccent(random_post_category_title.lower())

        print("random_post_category_title")
        print(random_post_category_title)

        print("rating_estimation_df.index")
        print(rating_estimation_df.index)

        if random_post_category_title in rating_estimation_df.index.tolist():
            break

    print("rating_estimation_df without random noise:")
    print(rating_estimation_df)

    rating_estimation_df = generate_random_noise_for_df(rating_estimation_df)
    rating_estimation_df = normalize(rating_estimation_df)
    rating_estimation_df = rating_estimation_df.round(2)

    print("rating_estimation_df with random noise:")
    print(rating_estimation_df)

    demographic_groups = rating_estimation_df.columns.to_list()
    random_demographic_group = random.choice(demographic_groups)
    chosen_rating_for_post = rating_estimation_df[random_demographic_group][random_post_category_title]

    print("Random post slug:")
    print(random_post_slug)

    print("Random post category:")
    print(random_post_category_title)

    print("random_demographic_group:")
    print(random_demographic_group)

    print("chosen_rating_for_post")
    print(chosen_rating_for_post)

    categories_df = recommender_methods.get_categories_dataframe()
    # changing categories titles to preprocessed variants to match the loaded results data
    categories_df['title'] = categories_df.apply(lambda x: gensim.utils.deaccent(x['title'].lower()), axis=1)
    print(categories_df)
    selected_category = categories_df.loc[categories_df['title'] == random_post_category_title]['id'].iloc[0]

    print("Random category ID:")
    print(selected_category)

    random_post_id = recommender_methods.find_post_by_slug(random_post_slug)['id'].iloc[0]

    print("Random post ID:")
    print(random_post_id)
    try:
        rounded_rating_value = round(chosen_rating_for_post)
        if rounded_rating_value > 5:
            rounded_rating_value = 5
        if rounded_rating_value < 1:
            rounded_rating_value = 1
        print("Rounded rating:")
        print(rounded_rating_value)

        recommender_methods.database.insert_rating(random_post_id, rounded_rating_value)
    except ValueError as e:
        print(e)
        print("Value is NaN. Skipping")


def normalize(df):
    normalized_df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    normalized_df = normalized_df * 5
    print("Adding 1.0 to dataset where value < 1.0")
    normalized_df = normalized_df.where(normalized_df > 1.0, 1.0)
    filter_cols = [col for col in normalized_df if col.startswith('Reuters')]
    for filter_col in filter_cols:
        normalized_df[filter_col]['moda'] = np.nan
    return normalized_df


def generate_random_noise_for_df(clean_df):

    mu, sigma = clean_df.mean(), clean_df.std()
    # creating a noise with the same dimension as the dataset (2,2)
    noise = np.random.normal(mu, sigma, clean_df.shape)

    signal = clean_df + noise

    return signal


class Normalizer:
    pass
