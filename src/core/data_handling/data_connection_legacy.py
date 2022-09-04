# import psycopg2.connector
import os
from pathlib import Path

import psycopg2
import pandas as pd

DB_USER = os.environ.get('DB_RECOMMENDER_USER')
DB_PASSWORD = os.environ.get('DB_RECOMMENDER_PASSWORD')
DB_HOST = os.environ.get('DB_RECOMMENDER_HOST')
DB_NAME = os.environ.get('DB_RECOMMENDER_NAME')


class Database:
    cnx = None
    cursor = None
    df = None

    def connect(self):
        self.cnx = psycopg2.connect(user=DB_USER,
                                    password=DB_PASSWORD,
                                    host=DB_HOST,
                                    dbname=DB_NAME)

        self.cursor = self.cnx.cursor()

    def disconnect(self):
        self.cursor.close()
        self.cnx.close()

    def get_cnx(self):
        return self.cnx

    def set_row_var(self):
        sql_set_var = """SET @row_number = 0;"""
        self.cursor.execute(sql_set_var)

    def get_all_posts(self):

        sql = """SELECT * FROM posts ORDER BY id;"""

        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_all_categories(self):
        sql = """SELECT * FROM categories ORDER BY id;"""

        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_all_posts_and_categories(self):
        sql = """SELECT * FROM categories ORDER BY id;"""

        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def join_posts_ratings_categories(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[['id_x','post_title','slug','excerpt','body','views','keywords','category_title','description','all_features_preprocessed']]

    def get_posts_join_categories(self):

        sql = """SELECT posts.slug, posts.title, categories.title, posts.excerpt, body, keywords, all_features_preprocessed, full_text, body_preprocessed, posts.recommended_tfidf, posts.recommended_word2vec, posts.recommended_doc2vec, posts.recommended_lda, posts.recommended_tfidf_full_text, posts.recommended_word2vec_full_text, posts.recommended_doc2vec_full_text, posts.recommended_lda_full_text FROM posts JOIN categories ON posts.category_id = categories.id;;"""

        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_all_users(self):
        sql = """SELECT * FROM users ORDER BY id;"""
        query = (sql)
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        return rs

    def get_users_with_ratings(self):
        database = Database()
        database.connect()
        sql_select_all_users = """SELECT DISTINCT u.id AS user_id, u.name FROM users u JOIN ratings r ON r.user_id = u.id;"""
        # LOAD INTO A DATAFRAME
        self.df_users = pd.read_sql_query(sql_select_all_users, database.get_cnx())
        return self.df_users

    def get_all_users_ids(self):
        database = Database()
        database.connect()
        sql_select_all_users = """SELECT u.id AS user_id, u.name FROM users u;"""
        # LOAD INTO A DATAFRAME
        self.df_users = pd.read_sql_query(sql_select_all_users, database.get_cnx())
        return self.df_users

    def get_all_users(self):
        sql = """SELECT * FROM users ORDER BY id;"""

        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_post_by_id(self, post_id):

        query = ("SELECT * FROM posts WHERE id = '%s'" % (post_id))
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        return rs

    def get_posts_dataframe_from_sql(self, pd):
        print("Getting posts from SQL...")
        sql = """SELECT * FROM posts ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return df

    def get_posts_dataframe(self, from_cache=True):
        # self.database.insert_posts_dataframe_to_cache() # uncomment for UPDATE of DB records
        if from_cache is True:
            self.posts_df = Database().get_posts_dataframe_from_cache()
        else:
            self.posts_df = self.get_posts_dataframe_from_sql(pd)
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df


    def insert_posts_dataframe_to_cache(self):
        sql = """SELECT * FROM posts ORDER BY id;"""
        folder_name = 'db_cache/'
        p = Path(folder_name)
        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())

        outfile = 'cached_posts_dataframe.pkl'

        outdir = './db_cache'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fullpath = os.path.join(outdir, outfile)
        df.to_pickle(fullpath)  # will be stored in current directory
        return df

    def get_posts_dataframe_from_cache(self):
        print("Reading cache file...")
        try:
            df = pd.read_pickle('../db_cache/cached_posts_dataframe.pkl')  # read from current directory
            return df
        except Exception as e:
            print("Exception occured when reading file:")
            print(e)
            raise e
    def get_categories_dataframe(self):
        sql = """SELECT * FROM categories ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        self.category_df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return self.category_df

    def get_ratings_dataframe(self, pd):
        sql = """SELECT * FROM ratings ORDER BY id;"""

        # LOAD INTO A DATAFRAME

        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return df

    def get_users_dataframe(self):
        sql = """SELECT * FROM users ORDER BY id;"""
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return df

    def get_ratings_dataframe(self, pd):
        sql = """SELECT * FROM ratings ORDER BY id;"""
        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return df

    def get_user_categories(self, pd):
        sql = """SELECT * FROM user_categories ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return df

    def insert_keywords(self,keyword_all_types_splitted,article_id):
        # PREPROCESSING
        try:
            query = """UPDATE posts SET keywords=%s WHERE id=%s;"""
            inserted_values = (keyword_all_types_splitted, article_id)
            self.cursor.execute(query, inserted_values)
            self.cnx.commit()

        except psycopg2.OperationalError as e:
            print("NOT INSERTED")

        except psycopg2.connector.Error as e:
            print("NOT INSERTED")
            print("Error code:", e.errno)  # error number
            print("SQLSTATE value:", e.sqlstate)  # SQLSTATE value
            print("Error message:", e.msg)  # error message

            print("Error:", e)  # errno, sqlstate, msg values
            s = str(e)
            print("Error:", s)  # errno, sqlstate, msg values
            self.cnx.rollback()

    @DeprecationWarning
    def insert_recommended_tfidf_json(self, articles_recommended_json, article_id, db):
        if db == "pgsql":
            try:
                query = """UPDATE posts SET recommended_tfidf=%s WHERE id=%s;"""
                inserted_values = (articles_recommended_json, article_id)
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()
                print("Inserted")
            except psycopg2.connector.Error as e:
                print("NOT INSERTED")
                print("Error code:", e.errno)  # error number
                print("SQLSTATE value:", e.sqlstate)  # SQLSTATE value
                print("Error message:", e.msg)  # error message
                print("Error:", e)  # errno, sqlstate, msg values
                s = str(e)
                print("Error:", s)  # errno, sqlstate, msg values
                self.cnx.rollback()
                pass
        elif db == "redis":
            raise Exception("Redis is not implemented yet.")
        else:
            raise ValueError("Not allowed DB method passed.")

    def insert_recommended_json(self, algorithm, full_text, articles_recommended_json, article_id, db):
        if db == "pgsql":
            try:
                if algorithm == "tfidf" and full_text is False:
                    query = """UPDATE posts SET recommended_tfidf=%s WHERE id=%s;"""
                elif algorithm == "tfidf" and full_text is True:
                    query = """UPDATE posts SET recommended_tfidf_full_text=%s WHERE id=%s;"""
                elif algorithm == "doc2vec" and full_text is False:
                    query = """UPDATE posts SET recommended_doc2vec=%s WHERE id=%s;"""
                elif algorithm == "doc2vec" and full_text is True:
                    query = """UPDATE posts SET recommended_doc2vec_full_text=%s WHERE id=%s;"""
                elif algorithm == "lda" and full_text is False:
                    query = """UPDATE posts SET recommended_lda=%s WHERE id=%s;"""
                elif algorithm == "lda" and full_text is True:
                    query = """UPDATE posts SET recommended_lda_full_text=%s WHERE id=%s;"""

                inserted_values = (articles_recommended_json, article_id)
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()
                print("Inserted")

            except psycopg2.connector.Error as e:
                print("NOT INSERTED")
                print("Error code:", e.errno)  # error number
                print("SQLSTATE value:", e.sqlstate)  # SQLSTATE value
                print("Error message:", e.msg)  # error message
                print("Error:", e)  # errno, sqlstate, msg values
                s = str(e)
                print("Error:", s)  # errno, sqlstate, msg values
                self.cnx.rollback()
                pass
        elif db == "redis":
            raise Exception("Redis is not implemented yet.")
        else:
            raise ValueError("Not allowed DB method passed.")

    @DeprecationWarning
    def insert_doc2vec_vector(self, doc2vec_vector, article_id):
        query = """UPDATE posts SET doc2vec_representation=%s WHERE id=%s;"""
        inserted_values = (doc2vec_vector, article_id)
        self.cursor.execute(query, inserted_values)
        self.cnx.commit()
        print("Inserted")

    def insert_preprocessed_combined(self, preprocessed_all_features, post_id):
        try:
            query = """UPDATE posts SET all_features_preprocessed=%s WHERE id=%s;"""
            inserted_values = (preprocessed_all_features, post_id)
            self.cursor.execute(query, inserted_values)
            self.cnx.commit()

        except psycopg2.connector.Error as e:
            print("NOT INSERTED")
            print("Error code:", e.errno)  # error number
            print("SQLSTATE value:", e.sqlstate)  # SQLSTATE value
            print("Error message:", e.msg)  # error message
            print("Error:", e)  # errno, sqlstate, msg values
            s = str(e)
            print("Error:", s)  # errno, sqlstate, msg values
            self.cnx.rollback()


    def get_not_prefilled_posts(self, full_text, method):
        self.connect()
        if full_text is False:
            if method == "tfidf":
                sql = """SELECT * FROM posts AS p WHERE p.recommended_tfidf IS NULL ORDER BY id DESC;"""
            elif method == "doc2vec":
                sql = """SELECT * FROM posts AS p WHERE p.recommended_doc2vec IS NULL ORDER BY id DESC;"""
            elif method == "word2vec":
                sql = """SELECT * FROM posts AS p WHERE p.recommended_word2vec IS NULL ORDER BY id DESC;"""
            elif method == "lda":
                sql = """SELECT * FROM posts AS p WHERE p.recommended_lda IS NULL ORDER BY id DESC;"""
            elif method == "doc2vec_vectors":
                sql = """SELECT * FROM posts AS p WHERE p.doc2vec_representation IS NULL ORDER BY id DESC;"""
            else:
                raise ValueError("Selected method not implemented.")
        else:
            if method == "tfidf":
                sql = """SELECT * FROM posts WHERE recommended_tfidf_full_text IS NULL ORDER BY id DESC;"""
            elif method == "doc2vec":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec_full_text IS NULL ORDER BY id DESC;"""
            elif method == "word2vec":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_full_text IS NULL ORDER BY id DESC;"""
            elif method == "lda":
                sql = """SELECT * FROM posts WHERE recommended_lda_full_text IS NULL ORDER BY id DESC;"""
            elif method == "word2vec_eval_1":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_1 IS NULL ORDER BY id DESC"""
            elif method == "word2vec_eval_2":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_2 IS NULL ORDER BY id DESC"""
            elif method == "word2vec_eval_3":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_3 IS NULL ORDER BY id DESC"""
            elif method == "word2vec_eval_4":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_4 IS NULL ORDER BY id DESC"""
            else:
                raise ValueError("Selected method not implemented.")

        query = (sql)
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        self.disconnect()
        return rs

    def get_not_prefilled_posts(self, full_text, algorithm):
        if full_text is False:
            if algorithm == "tfidf":
                sql = """SELECT * FROM posts WHERE recommended_tfidf IS NULL ORDER BY id DESC;"""
            elif algorithm == "doc2vec":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec IS NULL ORDER BY id DESC;"""
            elif algorithm == "lda":
                sql = """SELECT * FROM posts WHERE recommended_lda IS NULL ORDER BY id DESC;"""
            elif algorithm == "doc2vec_vectors":
                sql = """SELECT * FROM posts WHERE doc2vec_representation IS NULL ORDER BY id DESC;"""
            else:
                raise ValueError("Selected algorithm not implemented.")
        else:
            if algorithm == "tfidf":
                sql = """SELECT * FROM posts WHERE recommended_tfidf_full_text IS NULL ORDER BY id DESC;"""
            elif algorithm == "doc2vec":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec_full_text IS NULL ORDER BY id DESC;"""
            elif algorithm == "lda":
                sql = """SELECT * FROM posts WHERE recommended_lda_full_text IS NULL ORDER BY id DESC;"""

        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_dataframe_from_database(self):
        sql = """SELECT * FROM posts ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_results_dataframe(self, pd):
        sql = """SELECT * FROM relevance_testings ORDER BY id;"""
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def insert_preprocessed_body(self, preprocessed_body, article_id):
        try:
            query = """UPDATE posts SET body_preprocessed=%s WHERE id=%s;"""
            inserted_values = (preprocessed_body, article_id)
            self.cursor.execute(query, inserted_values)
            self.cnx.commit()

        except psycopg2.Error as e:
            print("NOT INSERTED")
            print("Error code:", e.pgcode)  # error number
            print("SQLSTATE value:", e.pgerror)  # SQLSTATE value
            print("Error:", e)  # errno, sqlstate, msg values
            s = str(e)
            print("Error:", s)  # errno, sqlstate, msg values
            self.cnx.rollback()

    def get_posts_with_no_body_preprocessed(self):
        sql = """SELECT * FROM posts WHERE body_preprocessed IS NULL ORDER BY id;"""
        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_all_features_preprocessed(self):
        sql = """SELECT * FROM posts WHERE all_features_preprocessed IS NULL ORDER BY id;"""
        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_keywords(self):
        sql = """SELECT * FROM posts WHERE keywords IS NULL ORDER BY id;"""
        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_prefilled_tfidf(self, full_text):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE recommended_tfidf IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE recommended_tfidf_full_text IS NULL ORDER BY id;"""
        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_prefilled_word2vec(self, full_text):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE recommended_word2vec IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE recommended_word2vec_full_text IS NULL ORDER BY id;"""
        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_prefilled_doc2vec(self, full_text):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE recommended_doc2vec IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE recommended_doc2vec_full_text IS NULL ORDER BY id;"""
        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_prefilled_lda(self, full_text):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE recommended_lda IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE recommended_lda_full_text IS NULL ORDER BY id;"""
        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs