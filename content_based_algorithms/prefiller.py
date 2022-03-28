import json
import random
import time as t

import psycopg2

from content_based_algorithms.doc2vec import Doc2VecClass
from content_based_algorithms.lda import Lda
from content_based_algorithms.tfidf import TfIdf
from data_conenction import Database

val_error_msg_db = "Not allowed DB method was passed for prefilling. Choose 'pgsql' or 'redis'."
val_error_msg_algorithm = "Selected algorithm does not correspondent with any implemented algorithm."

class PreFiller():

    def prefilling_job(self, algorithm, db, full_text, reverse, random=False):
        for i in range(100):
            while True:
                try:
                    self.fill_recommended_for_all_posts(algorithm, db, skip_already_filled=True, full_text=full_text, reversed=reverse, random_order=random)
                except psycopg2.OperationalError:
                    print("DB operational error. Waiting few seconds before trying again...")
                    t.sleep(30)  # wait 30 seconds then try again
                    continue
                break

    def fill_recommended_for_all_posts(self, algorithm, db, skip_already_filled, full_text=True, random_order=False, reversed=False):

        list_of_allowed_algorithms = ["word2vec", "lda", "doc2vec", "tfidf"]

        if algorithm not in list_of_allowed_algorithms:
            ValueError(val_error_msg_algorithm)

        database = Database()
        database.connect()
        if skip_already_filled is False:
            posts = database.get_all_posts()
        else:
            posts = database.get_not_prefilled_posts(full_text, algorithm)

        number_of_inserted_rows = 0

        if reversed is True:
            print("Reversing list of posts...")
            posts.reverse()

        if random_order is True:
            print("Starting random iteration...")
            t.sleep(5)
            random.shuffle(posts)

        for post in posts:
            if len(posts) < 1:
                break
            post_id = post[0]
            slug = post[3]
            if algorithm == "word2vec":
                if full_text is False:
                    current_recommended = post[22]
                else:
                    current_recommended = post[23]
            elif algorithm == "tfidf":
                if full_text is False:
                    current_recommended = post[24]
                else:
                    current_recommended = post[25]
            elif algorithm == "doc2vec":
                if full_text is False:
                    current_recommended = post[26]
                else:
                    current_recommended = post[27]
            elif algorithm == "lda":
                if full_text is False:
                    current_recommended = post[28]
                else:
                    current_recommended = post[29]

            print("Finding similar articles for article: " + slug)

            if skip_already_filled is True:
                if current_recommended is None:
                    if full_text is False:
                        if algorithm == "tfidf":
                            tfidf = TfIdf()
                            actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed(slug)
                        elif algorithm == "doc2vec":
                            doc2vec = Doc2VecClass()
                            actual_recommended_json = doc2vec.get_similar_doc2vec(slug)
                        elif algorithm == "lda":
                            lda = Lda()
                            actual_recommended_json = lda.get_similar_lda(slug)
                    else:
                        if algorithm == "tfidf":
                            tfidf = TfIdf()
                            actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(slug)
                        elif algorithm == "doc2vec":
                            doc2vec = Doc2VecClass()
                            actual_recommended_json = doc2vec.get_similar_doc2vec_with_full_text(slug)
                        elif algorithm == "lda":
                            lda = Lda()
                            actual_recommended_json = lda.get_similar_lda_full_text(slug)
                        else:
                            raise ValueError(val_error_msg_algorithm)
                    actual_recommended_json = json.dumps(actual_recommended_json)

                    # inserts
                    if full_text is False:
                        try:
                            if db == "redis":
                                database.insert_recommended_json(algorithm=algorithm, full_text=full_text,
                                                                 articles_recommended_json=actual_recommended_json,
                                                                 article_id=post_id, db="redis")
                            elif db == "pgsql":
                                database.insert_recommended_json(algorithm=algorithm, full_text=full_text,
                                                                 articles_recommended_json=actual_recommended_json,
                                                                 article_id=post_id, db="pgsql")
                            else:
                                raise ValueError(val_error_msg_db)
                        except Exception as e:
                            print("Error in DB insert. Skipping." + str(e))
                            pass
                    else:
                        try:
                            if db == "redis":
                                database.insert_recommended_json(algorithm=algorithm, full_text=full_text,
                                    articles_recommended_json=actual_recommended_json,
                                    article_id=post_id, db="redis")
                            elif db == "pgsql":
                                database.insert_recommended_json(algorithm=algorithm, full_text=full_text,
                                    articles_recommended_json=actual_recommended_json,
                                    article_id=post_id, db="pgsql")
                            else:
                                raise ValueError(val_error_msg_db)
                        except Exception as e:
                            print("Error in DB insert. Skipping." + str(e))
                            pass
                    number_of_inserted_rows += 1
                    if number_of_inserted_rows > 20:
                        print("Refreshing list of posts for finding only not prefilled posts.")
                        if full_text is False:
                            self.fill_recommended_for_all_posts(algorithm, db, skip_already_filled=True,
                                                                full_text=full_text, reversed=reversed,
                                                                random_order=random_order)
                    # print(str(number_of_inserted_rows) + " rows insertd.")
                else:
                    print("Skipping.")
            else:
                if full_text is False:
                    if algorithm == "tfidf":
                        tfidf = TfIdf()
                        actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed(slug)
                    elif algorithm == "doc2vec":
                        doc2vec = Doc2VecClass()
                        actual_recommended_json = doc2vec.get_similar_doc2vec(slug)
                    elif algorithm == "lda":
                        lda = Lda()
                        actual_recommended_json = lda.get_similar_lda(slug)
                else:
                    if algorithm == "tfidf":
                        tfidf = TfIdf()
                        actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(slug)
                    elif algorithm == "doc2vec":
                        doc2vec = Doc2VecClass()
                        actual_recommended_json = doc2vec.get_similar_doc2vec_with_full_text(slug)
                    elif algorithm == "lda":
                        lda = Lda()
                        actual_recommended_json = lda.get_similar_lda_full_text(slug)
                    else:
                        raise ValueError(val_error_msg_algorithm)
                actual_recommended_json = json.dumps(actual_recommended_json)
                if full_text is False:
                    try:
                        if db == "redis":
                            database.insert_recommended_json(algorithm=algorithm, full_text=full_text,
                                articles_recommended_json=actual_recommended_json,
                                                                   article_id=post_id, db="redis")
                        elif db == "pgsql":
                            database.insert_recommended_json(algorithm=algorithm, full_text=full_text,
                                articles_recommended_json=actual_recommended_json,
                                article_id=post_id, db="pgsql")
                        else:
                            raise ValueError(val_error_msg_db)
                    except:
                        print("Error in DB insert. Skipping.")
                        pass
                else:
                    if algorithm == "tfidf":
                        try:
                            if db == "redis":
                                database.insert_recommended_json(
                                    articles_recommended_json=actual_recommended_json,
                                    article_id=post_id, db="redis")
                            elif db == "pgsql":
                                database.insert_recommended_json(
                                    articles_recommended_json=actual_recommended_json,
                                    article_id=post_id, db="pgsql")
                            else:
                                raise ValueError(val_error_msg_db)
                        except:
                            print("Error in DB insert. Skipping.")
                            pass
                    elif algorithm == "doc2vec":
                        try:
                            if db == "redis":
                                database.insert_recommended_doc2vec_full_json(
                                    articles_recommended_json=actual_recommended_json,
                                    article_id=post_id, db="redis")
                            elif db == "pgsql":
                                database.insert_recommended_doc2vec_full_json(
                                    articles_recommended_json=actual_recommended_json,
                                    article_id=post_id, db="pgsql")
                            else:
                                raise ValueError(val_error_msg_db)
                        except:
                            print("Error in DB insert. Skipping.")
                            pass
                    elif algorithm == "lda":
                        try:
                            if db == "redis":
                                database.insert_recommended_lda_full_json(
                                    articles_recommended_json=actual_recommended_json,
                                    article_id=post_id, db="redis")
                            elif db == "pgsql":
                                database.insert_recommended_lda_full_json(
                                    articles_recommended_json=actual_recommended_json,
                                    article_id=post_id, db="pgsql")
                            else:
                                raise ValueError(val_error_msg_db)
                        except:
                            print("Error in DB insert. Skipping.")
                            pass
                    else:
                        raise ValueError(val_error_msg_algorithm)
                number_of_inserted_rows += 1
                if number_of_inserted_rows > 20:
                    print("Refreshing list of posts for finding only not prefilled posts.")
                    if full_text is False:
                        self.fill_recommended_for_all_posts(algorithm, db, skip_already_filled=True,
                                                            full_text=full_text, reversed=reversed,
                                                            random_order=random_order)
                # print(str(number_of_inserted_rows) + " rows insertd.")