import json
import random
import sys
import time as t

import numpy as np
import psycopg2
from gensim.models import Doc2Vec

from content_based_algorithms.data_queries import RecommenderMethods
from content_based_algorithms.doc2vec import Doc2VecClass
from content_based_algorithms.lda import Lda
from content_based_algorithms.tfidf import TfIdf
from content_based_algorithms.word2vec import Word2VecClass
from data_connection import Database
from learn_to_rank import LightGBM

val_error_msg_db = "Not allowed DB method was passed for prefilling. Choose 'pgsql' or 'redis'."
val_error_msg_algorithm = "Selected algorithm does not correspondent with any implemented algorithm."


class PreFiller():

    def prefilling_job(self, algorithm, db, full_text, reverse, random=False):
        print("Running " + algorithm + ", full text is set to " + str(full_text))
        if algorithm == "doc2vec" or algorithm == "doc2vec_vectors":
            doc2vec = Doc2VecClass()
            doc2vec.load_model()
        else:
            doc2vec = None
            database = Database()
            not_prefilled_posts = database.get_not_prefilled_posts(algorithm=algorithm, full_text=full_text)
            while len(not_prefilled_posts) > 0:
                print(str(len(not_prefilled_posts)) + " not prefilled posts left.")
                try:
                    if algorithm == "doc2vec" or "doc2vec_vectors":
                        self.fill_recommended_for_all_posts(algorithm, db, doc2vec=doc2vec, skip_already_filled=True, full_text=full_text, reversed=reverse, random_order=random)
                        not_prefilled_posts = database.get_not_prefilled_posts(algorithm=algorithm, full_text=full_text)
                    else:
                        self.fill_recommended_for_all_posts(algorithm, db, doc2vec=doc2vec, skip_already_filled=True, full_text=full_text, reversed=reverse, random_order=random)
                except psycopg2.OperationalError as e:
                    print(e)
                    print("DB operational error. Waiting few seconds before trying again...")
                    t.sleep(30)  # wait 30 seconds then try again
                    continue

        if algorithm == "word2vec":
            word2vec = Word2VecClass()
            word2vec.fill_recommended_for_all_posts(skip_already_filled=False, random_order=False, reversed=True)

    def fill_recommended_for_all_posts(self, algorithm, db, doc2vec, skip_already_filled, full_text, random_order=False, reversed=False):

        list_of_allowed_algorithms = ["word2vec", "lda", "doc2vec", "tfidf", "doc2vec_vectors"]

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
                    current = post[22]
                else:
                    current = post[23]
            elif algorithm == "tfidf":
                if full_text is False:
                    current = post[24]
                else:
                    current = post[25]
            elif algorithm == "doc2vec":
                if full_text is False:
                    current = post[26]
                else:
                    current = post[27]
            elif algorithm == "lda":
                if full_text is False:
                    current = post[28]
                else:
                    current = post[29]
            elif algorithm == "doc2vec_vectors":
                current = post[30]

            print("Finding similar articles for article: " + slug)

            if skip_already_filled is True:
                if current is None:
                    if algorithm != "doc2vec_vectors":
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
                            elif algorithm == "word2vec":
                                word2vec = Word2VecClass()
                                actual_recommended_json = word2vec.get_similar_word2vec(slug)
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
                            elif algorithm == "word2vec":
                                word2vec = Word2VecClass()
                                actual_recommended_json = word2vec.get_similar_word2vec(slug)
                            else:
                                raise ValueError(val_error_msg_algorithm)
                        actual_recommended_json = json.dumps(actual_recommended_json)
                    elif algorithm == "doc2vec_vectors":
                        doc2vec_vector = doc2vec.get_vector_representation(slug)
                    # inserts
                    if full_text is False:
                        if algorithm == "doc2vec_vectors":
                            database.insert_doc2vec_vector(json.dumps(doc2vec_vector.tolist()), post_id)
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
                                database.disconnect()
                                raise ValueError(val_error_msg_db)
                        except Exception as e:
                            print("Error in DB insert. Skipping." + str(e))
                            pass
                    number_of_inserted_rows += 1
                    if number_of_inserted_rows > 20:
                        print("Refreshing list of posts for finding only not prefilled posts.")
                        if full_text is False:
                            self.fill_recommended_for_all_posts(algorithm, db, doc2vec, skip_already_filled=True,
                                                                full_text=full_text, reversed=reversed,
                                                                random_order=random_order)
                    # print(str(number_of_inserted_rows) + " rows insertd.")
                else:
                    print("Skipping.")
            else:
                if algorithm == "doc2vec_vectors":
                    doc2vec_vector = doc2vec.get_vector_representation(slug)
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
                        elif algorithm == "doc2vec_vectors":
                            doc2vec_vector = doc2vec.get_vector_representation(slug)
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
                        elif algorithm == "doc2vec_vectors":
                            doc2vec_vector = doc2vec.get_vector_representation(slug)
                        else:
                            raise ValueError(val_error_msg_algorithm)
                    actual_recommended_json = json.dumps(actual_recommended_json)
                    if full_text is False:
                        try:
                            if db == "redis":
                                database.insert_recommended_json(algorithm=algorithm, full_text=full_text,
                                    articles_recommended_json=actual_recommended_json,
                                                                       article_id=post_id, db="redis")
                                database.disconnect()
                            elif db == "pgsql":
                                database.insert_recommended_json(algorithm=algorithm, full_text=full_text,
                                    articles_recommended_json=actual_recommended_json,
                                    article_id=post_id, db="pgsql")
                                database.disconnect()
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
                                    database.disconnect()
                                elif db == "pgsql":
                                    database.insert_recommended_json(
                                        articles_recommended_json=actual_recommended_json,
                                        article_id=post_id, db="pgsql")
                                    database.disconnect()
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
                                    database.disconnect()
                                elif db == "pgsql":
                                    database.insert_recommended_doc2vec_full_json(
                                        articles_recommended_json=actual_recommended_json,
                                        article_id=post_id, db="pgsql")
                                    database.disconnect()
                                else:
                                    raise ValueError(val_error_msg_db)
                            except:
                                print("Error in DB insert. Skipping.")
                                pass
                        elif algorithm == "doc2vec_vectors":
                            try:
                                if db == "redis":
                                    database.insert_doc2vec_vector(
                                        json.dumps(doc2vec_vector.tolist()),
                                        article_id=post_id, db="redis")
                                    database.disconnect()
                                elif db == "pgsql":
                                    database.insert_doc2vec_vector(
                                        json.dumps(doc2vec_vector.tolist()),
                                        article_id=post_id, db="pgsql")
                                    database.disconnect()
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
                                    database.disconnect()
                                elif db == "pgsql":
                                    database.insert_recommended_lda_full_json(
                                        articles_recommended_json=actual_recommended_json,
                                        article_id=post_id, db="pgsql")
                                    database.disconnect()
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

        database.disconnect()

