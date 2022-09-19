# import psycopg2.connector
import os

import psycopg2
import pandas as pd

DB_USER = os.environ.get('DB_RECOMMENDER_USER')
DB_PASSWORD = os.environ.get('DB_RECOMMENDER_PASSWORD')
DB_HOST = os.environ.get('DB_RECOMMENDER_HOST')
DB_NAME = os.environ.get('DB_RECOMMENDER_NAME')


def get_posts_dataframe_from_cache():
    print("Reading cache file...")
    try:
        df = pd.read_pickle('../db_cache/cached_posts_dataframe.pkl')  # read from current directory
        return df
    except Exception as e:
        print("Exception occurred when reading file:")
        print(e)
        raise e


@DeprecationWarning
class Database:

    def __init__(self):
        self.category_df = None
        self.cnx = None
        self.cursor = None
        self.df = None

    def connect(self):
        self.cnx = psycopg2.connect(user=DB_USER,
                                    password=DB_PASSWORD,
                                    host=DB_HOST,
                                    dbname=DB_NAME)

        self.cursor = self.cnx.cursor()

    def disconnect(self):
        if self.cursor is not None:
            self.cursor.close()
        self.cnx.close()

    def get_cnx(self):
        return self.cnx

    def set_row_var(self):
        sql_set_var = """SET @row_number = 0;"""
        self.cursor.execute(sql_set_var)

    def get_all_posts(self):

        sql = """SELECT * FROM posts ORDER BY id;"""

        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_all_categories(self):
        sql = """SELECT * FROM categories ORDER BY id;"""

        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_all_posts_and_categories(self):
        sql = """SELECT * FROM categories ORDER BY id;"""

        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_join_categories(self):

        sql = """SELECT posts.slug, posts.title, categories.title, posts.excerpt, body, keywords, 
        all_features_preprocessed, full_text, body_preprocessed, posts.recommended_tfidf, posts.recommended_word2vec, 
        posts.recommended_doc2vec, posts.recommended_lda, posts.recommended_tfidf_full_text, 
        posts.recommended_word2vec_full_text, posts.recommended_doc2vec_full_text, posts.recommended_lda_full_text 
        FROM posts JOIN categories ON posts.category_id = categories.id;; """

        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_all_users(self):
        sql = """SELECT * FROM users ORDER BY id;"""
        query = sql
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        return rs

    def get_post_by_id(self, post_id):

        query = ("SELECT * FROM posts WHERE id = '%s'" % post_id)
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        return rs

    def insert_posts_dataframe_to_cache(self):
        sql = """SELECT * FROM posts ORDER BY id;"""
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

    def get_categories_dataframe(self):
        sql = """SELECT * FROM categories ORDER BY id;"""
        self.category_df = pd.read_sql_query(sql, self.get_cnx())
        # LOAD INTO A DATAFRAME
        # df = pd.read_sql_query(results, database.get_cnx())
        return self.category_df

    def get_users_dataframe(self):
        sql = """SELECT * FROM users ORDER BY id;"""
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return df

    @DeprecationWarning
    def insert_doc2vec_vector(self, doc2vec_vector, article_id):
        query = """UPDATE posts SET doc2vec_representation = %s WHERE id = %s;"""
        inserted_values = (doc2vec_vector, article_id)
        self.cursor.execute(query, inserted_values)
        self.cnx.commit()
        print("Inserted")

    # noinspection DuplicatedCode
    @DeprecationWarning
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

        query = sql
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        self.disconnect()
        return rs

    def get_posts_dataframe_from_database(self):
        sql = """SELECT * FROM posts ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    # noinspection DuplicatedCode
    @DeprecationWarning
    def insert_preprocessed_body(self, preprocessed_body, article_id):
        try:
            query = """UPDATE posts SET body_preprocessed = %s WHERE id = %s;"""
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
        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_all_features_preprocessed(self):
        sql = """SELECT * FROM posts WHERE all_features_preprocessed IS NULL ORDER BY id;"""
        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_keywords(self):
        sql = """SELECT * FROM posts WHERE keywords IS NULL ORDER BY id;"""
        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_prefilled_tfidf(self, full_text):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE recommended_tfidf IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE recommended_tfidf_full_text IS NULL ORDER BY id;"""
        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_prefilled_word2vec(self, full_text):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE recommended_word2vec IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE recommended_word2vec_full_text IS NULL ORDER BY id;"""
        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_prefilled_doc2vec(self, full_text):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE recommended_doc2vec IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE recommended_doc2vec_full_text IS NULL ORDER BY id;"""
        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_prefilled_lda(self, full_text):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE recommended_lda IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE recommended_lda_full_text IS NULL ORDER BY id;"""
        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs
