# import psycopg2.connector
import os

import psycopg2
import pandas as pd

DB_HOST = os.environ.get('DB_RECOMMENDER_HOST')
DB_USER = os.environ.get('DB_RECOMMENDER_USER')
DB_PASSWORD = os.environ.get('DB_RECOMMENDER_PASSWORD')
DB_NAME = os.environ.get('DB_RECOMMENDER_NAME')


class Database:
    cnx = None
    cursor = None
    df = None

    def connect(self):
        self.cnx = psycopg2.connect(user="azuzazlsunperc" ,
                                    password="04f3582a0c5ea6074d9e3c4ed16d2152594b2f76f2c0768e05db2c037fb65cd3",
                                    host="ec2-34-250-16-127.eu-west-1.compute.amazonaws.com",
                                    dbname="dfjbkqmu3imv9u")

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

    def join_posts_ratings_categories(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[['id_x','title_x','slug_x','excerpt','body','views','keywords','title_y','description','all_features_preprocessed']]

    def get_posts_join_categories(self):

        sql = """SELECT posts.id, posts.title, posts.slug, posts.excerpt, posts.keywords, categories.title, categories.description, posts.all_features_preprocessed, posts."excerptPreprocessed", posts."titlePreprocessed" FROM posts JOIN categories ON posts.category_id=categories.id;"""

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

    def get_post_by_id(self, post_id):

        query = ("SELECT * FROM posts WHERE id = '%s'" % (post_id))
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        return rs

    def get_posts_dataframe_from_sql(self, pd):
        sql = """SELECT * FROM posts ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return df

    def get_posts_dataframe(self):
        # self.database.insert_posts_dataframe_to_cache() # uncomment for UPDATE of DB records
        self.posts_df = Database().get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df


    def insert_posts_dataframe_to_cache(self):
        sql = """SELECT * FROM posts ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        df.to_pickle('db_cache/cached_posts_dataframe.pkl')  # will be stored in current directory
        return df

    def get_posts_dataframe_from_cache(self):
        df = pd.read_pickle('db_cache/cached_posts_dataframe.pkl')  # read from current directory
        return df

    def get_categories_dataframe(self, pd):
        sql = """SELECT * FROM categories ORDER BY id;"""

        # LOAD INTO A DATAFRAME
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

        except psycopg2.connector.Error as e:
            print("NOT INSERTED")
            print("Error code:", e.errno)  # error number
            print("SQLSTATE value:", e.sqlstate)  # SQLSTATE value
            print("Error message:", e.msg)  # error message
            print("Error:", e)  # errno, sqlstate, msg values
            s = str(e)
            print("Error:", s)  # errno, sqlstate, msg values
            self.cnx.rollback()

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

    def get_not_prefilled_posts(self, full_text, algorithm):
        if full_text is False:
            if algorithm == "tfidf":
                sql = """SELECT * FROM posts WHERE recommended_tfidf IS NULL ORDER BY id;"""
            elif algorithm == "doc2vec":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec IS NULL ORDER BY id;"""
            elif algorithm == "lda":
                sql = """SELECT * FROM posts WHERE recommended_lda IS NULL ORDER BY id;"""
            else:
                raise ValueError("Selected algorithm not implemented.")
        else:
            if algorithm == "tfidf":
                sql = """SELECT * FROM posts WHERE recommended_tfidf_full_text IS NULL ORDER BY id;"""
            elif algorithm == "doc2vec":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec_full_text IS NULL ORDER BY id;"""
            elif algorithm == "lda":
                sql = """SELECT * FROM posts WHERE recommended_lda_full_text IS NULL ORDER BY id;"""

        query = (sql)
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs
