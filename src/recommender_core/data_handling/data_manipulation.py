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

    def __init__(self):
        self.categories_df = None
        self.posts_df = None
        self.df = None
        self.cursor = None
        self.cnx = None

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

    def join_posts_ratings_categories(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'post_title', 'slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
             'all_features_preprocessed']]

    def get_posts_join_categories(self):

        sql = """SELECT posts.slug, posts.title, categories.title, posts.excerpt, body, 
        keywords, all_features_preprocessed, full_text, body_preprocessed, posts.recommended_tfidf, 
        posts.recommended_word2vec, posts.recommended_doc2vec, posts.recommended_lda, posts.recommended_tfidf_full_text, 
        posts.recommended_word2vec_full_text, posts.recommended_doc2vec_full_text, posts.recommended_lda_full_text 
        FROM posts JOIN categories ON posts.category_id = categories.id;"""

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

    def get_posts_dataframe_from_sql(self):
        print("Getting posts from SQL...")
        sql = """SELECT * FROM posts ORDER BY id;"""
        # NOTICE: Connection is ok here. Need to stay here due to calling from function that's executing thread
        # operation
        self.connect()
        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        self.disconnect()
        return df

    def get_posts_dataframe(self, from_cache=True):
        # self.database.insert_posts_dataframe_to_cache() # uncomment for UPDATE of DB records
        if from_cache is True:
            self.posts_df = self.get_posts_dataframe_from_cache()
        else:
            self.posts_df = self.get_posts_dataframe_from_sql()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    def insert_posts_dataframe_to_cache(self, cached_file_path=None):

        if cached_file_path is None:
            print("Cached file path is None. Using default location.")
            cached_file_path = 'db_cache/cached_posts_dataframe.pkl'

        # Connection needs to stay here, otherwise does not make any sense due to threading of
        # cache insert
        self.connect()
        sql = """SELECT * FROM posts ORDER BY id;"""
        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        self.disconnect()

        splitted_cache_file = cached_file_path.split('/')

        outfile = splitted_cache_file[1]
        outdir = './' + splitted_cache_file[0]

        if not os.path.exists(outdir):
            os.makedirs(Path(outdir), exist_ok=True)

        fullpath = os.path.join(Path(outdir), Path(outfile))
        df.to_pickle(fullpath)  # will be stored in current directory
        print("fullpath")
        print(fullpath)
        return df

    def get_posts_dataframe_from_cache(self):
        print("Reading cache file...")
        try:
            path_to_df = Path('db_cache/cached_posts_dataframe.pkl')
            df = pd.read_pickle(path_to_df)
            # read from current directory
        except Exception as e:
            print("Exception occured when reading cached file:")
            print(e)
            print("Getting posts from SQL.")
            df = self.get_posts_dataframe_from_sql()
        return df

    def get_categories_dataframe(self):
        sql = """SELECT * FROM categories ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        self.categories_df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return self.categories_df

    def get_ratings_dataframe(self):
        sql = """SELECT * FROM ratings ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return df

    def get_users_dataframe(self):
        sql = """SELECT * FROM users ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_user_categories(self, user_id=None):
        if user_id is None:
            sql = """SELECT * FROM user_categories ORDER BY id;"""

            # LOAD INTO A DATAFRAME
            df_user_categories = pd.read_sql_query(sql, self.get_cnx())
            # df = pd.read_sql_query(results, database.get_cnx())
        else:
            sql_user_categories = """SELECT c.slug AS "category_slug" FROM user_categories uc 
            JOIN categories c ON c.id = uc.category_id WHERE uc.user_id = (%(user_id)s);"""
            query_params = {'user_id': user_id}
            df_user_categories = pd.read_sql_query(sql_user_categories, self.get_cnx(),
                                                   params=query_params)

            print("df_user_categories:")
            print(df_user_categories)
            return df_user_categories
        return df_user_categories

    def insert_keywords(self, keyword_all_types_splitted, article_id):
        # PREPROCESSING
        try:
            query = """UPDATE posts SET keywords = %s WHERE id = %s;"""
            inserted_values = (keyword_all_types_splitted, article_id)
            self.cursor.execute(query, inserted_values)
            self.cnx.commit()

        except psycopg2.OperationalError as e:
            print("NOT INSERTED")
            print("Error:", e)  # errno, sqlstate, msg values
            s = str(e)
            print("Error:", s)  # errno, sqlstate, msg values
            self.cnx.rollback()

    def get_user_rating_categories(self):

        # EXTRACT RESULTS FROM CURSOR

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug AS post_slug, r.value AS rating_value, c.title AS category_title, c.slug AS category_slug, p.created_at AS post_created_at
        FROM posts p
        JOIN ratings r ON r.post_id = p.id
        JOIN users u ON r.user_id = u.id
        JOIN categories c ON c.id = p.category_id
        LEFT JOIN user_categories uc ON uc.category_id = c.id;"""

        df_ratings = pd.read_sql_query(sql_rating, self.get_cnx())
        print("Loaded those ratings from DB.")
        print(df_ratings)
        print(df_ratings.columns)

        if 'slug_y' in df_ratings.columns:
            df_ratings = df_ratings.rename(columns={'slug': 'category_slug'})

        return df_ratings

    def get_user_keywords(self, user_id):
        sql_user_keywords = """SELECT multi_dimensional_list.name AS "keyword_name" FROM tag_user tu JOIN tags 
        multi_dimensional_list ON multi_dimensional_list.id = tu.tag_id WHERE tu.user_id = (%(user_id)s); """
        query_params = {'user_id': user_id}
        df_user_categories = pd.read_sql_query(sql_user_keywords, self.get_cnx(), params=query_params)
        print("df_user_categories:")
        print(df_user_categories)
        return df_user_categories

    @DeprecationWarning
    def insert_recommended_tfidf_json(self, articles_recommended_json, article_id, db):
        if db == "pgsql":
            try:
                query = """UPDATE posts SET recommended_tfidf = %s WHERE id = %s;"""
                inserted_values = (articles_recommended_json, article_id)
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()
                print("Inserted")
            except psycopg2.Error as e:
                print("NOT INSERTED")
                print(e.pgcode)
                print(e.pgerror)
                print("Full error: ", e)  # errno, sqlstate, msg values
                self.cnx.rollback()
                pass
        elif db == "redis":
            raise Exception("Redis is not implemented yet.")
        else:
            raise ValueError("Not allowed DB model_variant passed.")

    def insert_recommended_json(self, method, full_text, articles_recommended_json, article_id, db):
        if db == "pgsql":
            try:
                if method == "tfidf" and full_text is False:
                    query = """UPDATE posts SET recommended_tfidf = %s WHERE id = %s;"""
                elif method == "tfidf" and full_text is True:
                    query = """UPDATE posts SET recommended_tfidf_full_text = %s WHERE id = %s;"""
                elif method == "word2vec" and full_text is False:
                    query = """UPDATE posts SET recommended_word2vec = %s WHERE id = %s;"""
                elif method == "word2vec" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_full_text = %s WHERE id = %s;"""
                elif method == "doc2vec" and full_text is False:
                    query = """UPDATE posts SET recommended_doc2vec = %s WHERE id = %s;"""
                elif method == "doc2vec" and full_text is True:
                    query = """UPDATE posts SET recommended_doc2vec_full_text = %s WHERE id = %s;"""
                elif method == "lda" and full_text is False:
                    query = """UPDATE posts SET recommended_lda = %s WHERE id = %s;"""
                elif method == "lda" and full_text is True:
                    query = """UPDATE posts SET recommended_lda_full_text = %s WHERE id = %s;"""
                elif method == "word2vec_eval_idnes_1" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_1 = %s WHERE id = %s;"""
                elif method == "word2vec_eval_idnes_2" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_2 = %s WHERE id = %s;"""
                elif method == "word2vec_eval_idnes_3" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_3 = %s WHERE id = %s;"""
                elif method == "word2vec_eval_idnes_4" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_4 = %s WHERE id = %s;"""
                elif method == "word2vec_eval_cswiki_1" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_cswiki_1 = %s WHERE id = %s;"""
                elif method == "doc2vec_eval_cswiki_1" and full_text is True:
                    query = """UPDATE posts SET recommended_doc2vec_eval_cswiki_1 = %s WHERE id = %s;"""
                else:
                    raise Exception("Method not implemented.")
                inserted_values = (articles_recommended_json, article_id)
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()
                print("Inserted")
            except psycopg2.Error as e:
                print("NOT INSERTED")
                print(e.pgcode)
                print(e.pgerror)
                s = str(e)
                print("Full Error: ", s)  # errno, sqlstate, msg values
                self.cnx.rollback()
                pass
        elif db == "redis":
            raise Exception("Redis is not implemented yet.")
        else:
            raise ValueError("Not allowed DB method passed.")

    @DeprecationWarning
    def insert_doc2vec_vector(self, doc2vec_vector, article_id):
        query = """UPDATE posts SET doc2vec_representation = %s WHERE id = %s;"""
        inserted_values = (doc2vec_vector, article_id)
        self.cursor.execute(query, inserted_values)
        self.cnx.commit()
        print("Inserted")

    def insert_preprocessed_combined(self, preprocessed_all_features, post_id):
        try:
            query = """UPDATE posts SET all_features_preprocessed = %s WHERE id = %s;"""
            inserted_values = (preprocessed_all_features, post_id)
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

    def insert_phrases_text(self, bigram_text, article_id, full_text):
        try:
            if full_text is False:
                query = """UPDATE posts SET trigrams_short_text = %s WHERE id = %s;"""
            else:
                query = """UPDATE posts SET trigrams_full_text = %s WHERE id = %s;"""
            inserted_values = (bigram_text, article_id)
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

    def get_not_preprocessed_posts(self):
        sql = """SELECT * FROM posts WHERE body_preprocessed IS NULL ORDER BY id;"""
        # TODO: can be added also keywords, all_features_preprocecessed to make sure they were already added in
        #  Parser module
        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_not_prefilled_posts(self, full_text, method):
        if full_text is False:
            if method == "tfidf":
                sql = """SELECT * FROM posts WHERE recommended_tfidf IS NULL ORDER BY id;"""
            elif method == "word2vec":
                sql = """SELECT * FROM posts WHERE recommended_word2vec IS NULL ORDER BY id;"""
            elif method == "doc2vec":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec IS NULL ORDER BY id;"""
            elif method == "lda":
                sql = """SELECT * FROM posts WHERE recommended_lda IS NULL ORDER BY id;"""
            else:
                raise ValueError("Selected model_variant not implemented.")
        else:
            if method == "tfidf":
                sql = """SELECT * FROM posts WHERE recommended_tfidf_full_text IS NULL ORDER BY id;"""
            elif method == "word2vec":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_full_text IS NULL ORDER BY id;"""
            elif method == "doc2vec":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec_full_text IS NULL ORDER BY id;"""
            elif method == "lda":
                sql = """SELECT * FROM posts WHERE recommended_lda_full_text IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_idnes_1":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_1 IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_idnes_2":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_2 IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_idnes_3":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_3 IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_idnes_4":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_4 IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_cswiki_1":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_cswiki_1 IS NULL ORDER BY id;"""
            elif method == "doc2vec_eval_cswiki_1":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec_eval_cswiki_1 IS NULL ORDER BY id;"""
            else:
                raise ValueError("Selected model_variant not implemented.")

        query = sql
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        return rs

    def get_posts_dataframe_from_database(self):
        sql = """SELECT * FROM posts ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_relevance_testing_dataframe(self):
        sql = """SELECT * FROM relevance_testings ORDER BY id;"""
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_thumbs_dataframe(self):
        sql = """SELECT * FROM thumbs ORDER BY id;"""
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_posts_with_no_body_preprocessed(self):
        sql = """SELECT * FROM posts WHERE body_preprocessed IS NULL ORDER BY id;"""
        # TODO: can be added also keywords, all_features_preprocecessed to make sure they were already added in
        #  Parser module
        query = sql
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_keywords(self):
        sql = """SELECT * FROM posts WHERE posts.keywords IS NULL ORDER BY id;"""
        # TODO: can be added also keywords, all_features_preprocecessed to make sure they were already added in
        #  Parser module
        query = sql
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_no_all_features_preprocessed(self):
        sql = """SELECT * FROM posts WHERE posts.all_features_preprocessed IS NULL ORDER BY id;"""
        # TODO: can be added also keywords, all_features_preprocecessed to make sure they were already added in
        #  Parser module
        query = sql
        self.cursor.execute(query)
        rs = self.cursor.fetchall()
        return rs

    def get_posts_with_not_prefilled_ngrams_text(self, full_text=True):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE trigrams_short_text IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE trigrams_full_text IS NULL ORDER BY id;"""

        # TODO: can be added also: keywords, all_features_preprocecessed to make sure they were already added in
        #  Parser module
        query = sql
        self.cursor.execute(query)

        rs = self.cursor.fetchall()
        return rs

    def get_posts_users_categories_ratings(self):
        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug AS post_slug, r.value AS rating_value, c.title AS category_title, c.slug AS category_slug, p.created_at AS post_created_at
        FROM posts p
        JOIN ratings r ON r.post_id = p.id
        JOIN users u ON r.user_id = u.id
        JOIN categories c ON c.id = p.category_id
        LEFT JOIN user_categories uc ON uc.category_id = c.id;"""

        df_ratings = pd.read_sql_query(sql_rating, self.get_cnx())
        print("Loaded those ratings from DB.")
        print(df_ratings)

        return df_ratings

    def get_posts_users_categories_thumbs(self):
        sql_thumbs = """SELECT DISTINCT t.id AS thumb_id, p.id AS post_id, u.id AS user_id, p.slug AS post_slug,
        t.value AS thumbs_value, c.title AS category_title, c.slug AS category_slug,
        p.created_at AS post_created_at, t.created_at AS thumbs_created_at
        FROM posts p
        JOIN thumbs t ON t.post_id = p.id
        JOIN users u ON t.user_id = u.id
        JOIN categories c ON c.id = p.category_id;"""

        df_thumbs = pd.read_sql_query(sql_thumbs, self.get_cnx())

        print("df_thumbs")
        print(df_thumbs.to_string())

        # ### Keep only newest records of same post_id + user_id combination
        # Order by date of creation
        df_thumbs = df_thumbs.sort_values(by='thumbs_created_at')
        df_thumbs = df_thumbs.drop_duplicates(['post_id', 'user_id'], keep='last')

        print("df_thumbs after drop_duplicates")
        print(df_thumbs.to_string())

        # TODO: This can cause potential bug!!! Values are sorted by rating creation
        # but thumbs an be possible duplicated too...
        print("df_thumbs_ratings.columns")
        print(df_thumbs.columns)
        df_thumbs = df_thumbs.sort_values(by='thumbs_created_at')
        df_thumbs = df_thumbs.drop_duplicates(['post_id', 'user_id'], keep='last')

        return df_thumbs

