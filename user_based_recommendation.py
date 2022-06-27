import json

from content_based_algorithms.data_queries import RecommenderMethods
from data_connection import Database
import pandas as pd

class UserBasedRecommendation:

    def __init__(self):
        self.user_id = None
        self.database = Database()

    def get_user_id(self):
        return self.user_id

    @DeprecationWarning
    def set_database(self, database):
        self.database = database

    def get_database(self):
        return self.database

    def load_ratings(self):

        # EXTRACT RESULTS FROM CURSOR

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug AS post_slug, r.value AS rating_value, c.title AS category_title, c.slug AS category_slug, p.created_at AS post_created_at
        FROM posts p
        JOIN ratings r ON r.post_id = p.id
        JOIN users u ON r.user_id = u.id
        JOIN categories c ON c.id = p.category_id
        LEFT JOIN user_categories uc ON uc.category_id = c.id;"""

        df_ratings = pd.read_sql_query(sql_rating, self.get_database().get_cnx())
        print("Loaded those ratings from DB.")
        print(df_ratings)

        return df_ratings


    def load_user_categories(self, user_id):

        sql_user_categories = """SELECT c.slug AS "category_slug" FROM user_categories uc JOIN categories c ON c.id = uc.category_id WHERE uc.user_id = (%(user_id)s);"""
        queryParams = {'user_id': user_id}
        df_user_categories = pd.read_sql_query(sql_user_categories, self.get_database().get_cnx(), params=queryParams)

        print("df_user_categories:")
        print(df_user_categories)
        return df_user_categories

    def get_user_keywords(self, user_id):
        self.database.connect()
        sql_user_keywords = """SELECT t.name AS "keyword_name" FROM tag_user tu JOIN tags t ON t.id = tu.tag_id WHERE tu.user_id = (%(user_id)s);"""
        queryParams = {'user_id': user_id}
        df_user_categories = pd.read_sql_query(sql_user_keywords, self.get_database().get_cnx(), params=queryParams)
        self.database.disconnect()
        print("df_user_categories:")
        print(df_user_categories)
        return df_user_categories

    # loads posts for user based on his id and favourite categories
    def load_recommended_posts_for_user(self, user_id, num_of_recommendations=5):
        self.database.connect()
        df_posts_users_of_categories = self.load_ratings()[self.load_ratings().category_slug.isin(self.load_user_categories(user_id)['category_slug'].tolist())]
        df_filter_current_user = df_posts_users_of_categories[df_posts_users_of_categories.rating_id != self.get_user_id()]
        self.database.disconnect()
        df_sorted_results = df_filter_current_user[['post_id','post_slug','rating_value','post_created_at']].sort_values(['rating_value','post_created_at'], ascending=[False, False])
        df_sorted_results = df_sorted_results.drop_duplicates(subset=['post_id'])
        print("df_sorted_results[['post_slug']]")
        print(df_sorted_results[['post_id','post_slug']])
        return self.convert_to_json(df_sorted_results.head(num_of_recommendations))

    def load_user_keywords(self, user_id):
        self.database.connect()
        recommenderMethods = RecommenderMethods()
        recommenderMethods.get_user_keywords(user_id)
        self.database.disconnect()

    def convert_to_json(self, df):
        predictions_json = df.to_json(orient="split")
        predictions_json_parsed = json.loads(predictions_json)
        return predictions_json_parsed

def main():
    #Testing
    user_based_recommendation = UserBasedRecommendation()
    print(user_based_recommendation.load_recommended_posts_for_user(211,4))

if __name__ == "__main__": main()
