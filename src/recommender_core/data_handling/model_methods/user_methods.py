from src.recommender_core.data_handling.data_queries import RecommenderMethods

class UserMethods(RecommenderMethods):

    def __init__(self):
        super().__init__()

    def get_user_dataframe(self, user_id):
        self.database.connect()
        user_df = self.database.get_user_dataframe(user_id)
        self.database.disconnect()
        return user_df

    def get_users_dataframe(self):
        self.database.connect()
        user_df = self.database.get_users_dataframe()
        self.database.disconnect()
        return user_df

    def get_user_keywords(self, user_id):
        self.database.connect()
        df_user_keywords = self.database.get_user_keywords(user_id=user_id)
        self.database.disconnect()
        return df_user_keywords

    def get_user_rating_categories(self):
        self.database.connect()
        user_rating_categories_df = self.database.get_user_rating_categories()
        self.database.disconnect()
        return user_rating_categories_df

    def get_user_categories(self, user_id):
        self.database.connect()
        df_user_categories = self.database.get_user_categories(user_id)
        self.database.disconnect()
        return df_user_categories

    def get_posts_df_users_df_ratings_df(self):
        self.database.connect()
        df_posts, df_users, df_ratings = self.database.get_posts_df_users_df_ratings_df()
        self.database.disconnect()
        return df_posts, df_users, df_ratings
