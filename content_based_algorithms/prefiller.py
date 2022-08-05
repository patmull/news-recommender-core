import json
import random
import time as t
import psycopg2

from content_based_algorithms.doc2vec import Doc2VecClass
from content_based_algorithms.lda import Lda
from content_based_algorithms.tfidf import TfIdf
from content_based_algorithms.word2vec import Word2VecClass
from data_connection import Database

val_error_msg_db = "Not allowed DB method was passed for prefilling. Choose 'pgsql' or 'redis'."
val_error_msg_algorithm = "Selected method does not correspondent with any implemented method."


class PreFiller:

    def prefilling_job(self, method, full_text, random_order=False, reversed=True, database="pgsql"):
        while True:
            try:
                self.fill_recommended(method=method, full_text=full_text, skip_already_filled=True,
                                      random_order=random_order, reversed=reversed)
            except psycopg2.OperationalError:
                print("DB operational error. Waiting few seconds before trying again...")
                t.sleep(30)  # wait 30 seconds then try again
                continue
            break

    def fill_recommended(self, method, skip_already_filled, full_text=True, random_order=False, reversed=False):

        database = Database()
        database.connect()
        if skip_already_filled is False:
            posts = database.get_all_posts()
        else:
            posts = database.get_not_prefilled_posts(full_text, method=method)

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

            if full_text is False:
                if method == "tfidf":
                    current_recommended = post[24]
                elif method == "word2vec":
                    current_recommended = post[22]
                elif method == "doc2vec":
                    current_recommended = post[26]
                elif method == "lda":
                    current_recommended = post[28]
                else:
                    current_recommended = None
            else:
                if method == "tfidf":
                    current_recommended = post[25]
                elif method == "word2vec":
                    current_recommended = post[23]
                elif method == "doc2vec":
                    current_recommended = post[27]
                elif method == "lda":
                    current_recommended = post[29]
                elif method == "word2vec_eval_1":
                    current_recommended = post[33]
                else:
                    current_recommended = None

            print("Searching similar articles for article: ")
            print(slug)

            if skip_already_filled is True:
                if current_recommended is None:
                    print("Post:")
                    print(slug)
                    print("Has currently no recommended posts.")
                    print("Trying to find recommended...")
                    if full_text is False:
                        if method == "tfidf":
                            tfidf = TfIdf()
                            actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed(slug)
                        elif method == "word2vec":
                            word2vec = Word2VecClass()
                            actual_recommended_json = word2vec.get_similar_word2vec(slug)
                        elif method == "doc2vec":
                            doc2vec = Doc2VecClass()
                            actual_recommended_json = doc2vec.get_similar_doc2vec(slug)
                        elif method == "lda":
                            lda = Lda()
                            actual_recommended_json = lda.get_similar_lda(slug)
                        else:
                            actual_recommended_json = None
                    else:
                        if method == "tfidf":
                            tfidf = TfIdf()
                            actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(
                                slug)
                        elif method == "word2vec":
                            word2vec = Word2VecClass()
                            actual_recommended_json = word2vec.get_similar_word2vec_full_text(slug)
                        elif method == "doc2vec":
                            doc2vec = Doc2VecClass()
                            actual_recommended_json = doc2vec.get_similar_doc2vec_with_full_text(slug)
                        elif method == "lda":
                            lda = Lda()
                            actual_recommended_json = lda.get_similar_lda_full_text(slug)
                        else:
                            actual_recommended_json = None
                    if len(actual_recommended_json) == 0:
                        print("No recommended post found. Skipping.")
                        continue
                    else:
                        actual_recommended_json = json.dumps(actual_recommended_json)

                    if full_text is False:
                        try:
                            database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                             article_id=post_id, full_text=False, db="pgsql",
                                                             method=method)
                        except:
                            print("Error in DB insert. Skipping.")
                            pass
                    else:
                        try:
                            database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                             article_id=post_id, full_text=True, db="pgsql",
                                                             method=method)
                            number_of_inserted_rows += 1
                            print("Inserted rows in current prefilling round: " + str(number_of_inserted_rows))
                        except:
                            print("Error in DB insert. Skipping.")
                            pass
                else:
                    print("Skipping.")
