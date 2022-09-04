import time
import random

import pandas as pd

from src.core.recommender_algorithms.content_based_algorithms.word2vec import CzPreprocess
from src.core.data_handling.data_manipulation import Database


class PreFillerAdditional():

    # universal common method
    # TODO:
    def fill_preprocessed(self, column_for_prefilling, skip_already_filled, reversed, random_order, db="pgsql"):
        database = Database()
        if skip_already_filled is False:
            pandas_df = pd.DataFrame()
            posts = database.get_all_posts()
            categories = database.get_all_categories()
            posts_categories = database.join_posts_ratings_categories()
        else:
            posts_categories = database.get_not_preprocessed_posts()

        number_of_inserted_rows = 0

        if reversed is True:
            print("Reversing list of posts...")
            posts_categories.reverse()

        if random_order is True:
            print("Starting random_order iteration...")
            time.sleep(5)
            random.shuffle(posts_categories)

        cz_lemma = CzPreprocess()

        for post in posts_categories:
            if len(posts_categories) < 1:
                break
            post_id = post[0]
            slug = post[3]
            article_category = post[2]
            article_title = post[1]
            article_excerpt = post[3]
            article_body = post[4]
            article_keywords = post[5]
            article_all_features_preprocessed = post[6]
            article_full_text = post[7]
            article_body_preprocessed = post[8]

            print("Prefilling body preprocessd in article: " + slug)
    
            if skip_already_filled is True:
                if article_body_preprocessed is None:
                    if db == "redis":
                        preprocessed_body = cz_lemma.preprocess(" ")  # post[????] + post[????] ...
                        database.insert_preprocessed_body(preprocessed_body=article_body_preprocessed,
                                                          article_id=post_id)
                    elif db == "pgsql":
                        preprocessed_text = cz_lemma.preprocess(article_full_text)
                        print("article_full_text:")
                        print(article_full_text)
                        print("preprocessed_text:")
                        print(preprocessed_text)

                        database.insert_preprocessed_body(preprocessed_body=preprocessed_text, article_id=post_id)
                    else:
                        raise Exception
                else:
                    print("Skipping.")
            else:
                if db == "redis":
                    keywords = self.extract_keywords(article_title + " " + article_excerpt, from_api=True)
                    preprocessed_text = cz_lemma.preprocess(article_title + " " + article_excerpt)
                    print("article_full_text:")
                    print(article_full_text)
                    print("preprocessed_text:")
                    print(preprocessed_text)
                    preprocessed_text = preprocessed_text + " " + keywords

                    database.insert_preprocessed_body(preprocessed_body=preprocessed_text,
                                                      article_id=post_id)
                elif db == "pgsql":
                    preprocessed_text = cz_lemma.preprocess(article_full_text)
                    print("article_full_text")
                    print(article_full_text)
                    print("preprocessed_text:")
                    print(preprocessed_text)

                    database.insert_preprocessed_body(preprocessed_body=preprocessed_text, article_id=post_id)
                else:
                    raise Exception
                number_of_inserted_rows += 1
                if number_of_inserted_rows > 20:
                    print("Refreshing list of posts for finding only not prefilled posts.")
                    self.fill_all_features_preprocessed(db=db, skip_already_filled=True, reversed=reversed,
                                                        random_order=random_order)

    def fill_body_preprocessed(self, skip_already_filled, reversed, random_order, db="pgsql"):
        database = Database()
        if skip_already_filled is False:
            posts = database.get_all_posts()
        else:
            posts = database.get_not_preprocessed_posts()

        number_of_inserted_rows = 0

        if reversed is True:
            print("Reversing list of posts...")
            posts.reverse()

        if random_order is True:
            print("Starting random_order iteration...")
            time.sleep(5)
            random.shuffle(posts)

        cz_lemma = CzPreprocess()

        for post in posts:
            if len(posts) < 1:
                break
            post_id = post[0]
            slug = post[3]
            article_title = post[2]
            article_excerpt = post[4]
            article_full_text = post[20]
            current_body_preprocessed = post[21]

            print("Prefilling body preprocessd in article: " + slug)

            if skip_already_filled is True:
                if current_body_preprocessed is None:
                    if db == "pgsql":
                        preprocessed_text = cz_lemma.preprocess(article_full_text)
                        print("article_full_text:")
                        print(article_full_text)
                        print("preprocessed_text:")
                        print(preprocessed_text)

                        database.insert_preprocessed_body(preprocessed_body=preprocessed_text, article_id=post_id)
                    else:
                        raise NotImplementedError
                else:
                    print("Skipping.")
            else:
                if db == "pgsql":
                    preprocessed_text = cz_lemma.preprocess(article_full_text)
                    print("article_full_text")
                    print(article_full_text)
                    print("preprocessed_text:")
                    print(preprocessed_text)

                    database.insert_preprocessed_body(preprocessed_body=preprocessed_text, article_id=post_id)
                else:
                    raise NotImplementedError
                number_of_inserted_rows += 1
                if number_of_inserted_rows > 20:
                    print("Refreshing list of posts for finding only not prefilled posts.")
                    self.fill_all_features_preprocessed(db=db, skip_already_filled=True, reversed=reversed,
                                                        random_order=random_order)

    def fill_keywords(self, skip_already_filled, reversed, random_order):
        pass

    def fill_all_features_preprocessed(self, skip_already_filled, reversed, random_order, db="pgsql"):
        database = Database()
        if skip_already_filled is False:
            posts = database.get_all_posts()
        else:
            posts = database.get_not_preprocessed_posts()

        number_of_inserted_rows = 0

        if reversed is True:
            print("Reversing list of posts...")
            posts.reverse()

        if random_order is True:
            print("Starting random_order iteration...")
            time.sleep(5)
            random.shuffle(posts)

        cz_lemma = CzPreprocess()

        for post in posts:
            if len(posts) < 1:
                break
            post_id = post[0]
            slug = post[3]
            article_title = post[2]
            article_excerpt = post[4]
            article_full_text = post[20]
            current_body_preprocessed = post[21]
            # TODO: Category should be there too

            print("Prefilling body preprocessd in article: " + slug)

            if skip_already_filled is True:
                if current_body_preprocessed is None:
                    if db == "pgsql":
                        preprocessed_text = cz_lemma.preprocess(article_full_text)
                        print("article_full_text:")
                        print(article_full_text)
                        print("preprocessed_text:")
                        print(preprocessed_text)

                        database.insert_preprocessed_combined(preprocessed_body=preprocessed_text, article_id=post_id)
                    else:
                        raise NotImplementedError
                else:
                    print("Skipping.")
            else:
                if db == "pgsql":
                    preprocessed_text = cz_lemma.preprocess(article_full_text)
                    print("article_full_text")
                    print(article_full_text)
                    print("preprocessed_text:")
                    print(preprocessed_text)

                    database.insert_preprocessed_body(preprocessed_body=preprocessed_text, article_id=post_id)
                else:
                    raise NotImplementedError
                number_of_inserted_rows += 1
                if number_of_inserted_rows > 20:
                    print("Refreshing list of posts for finding only not prefilled posts.")
                    self.fill_all_features_preprocessed(db=db, skip_already_filled=True, reversed=reversed,
                                                        random_order=random_order)
