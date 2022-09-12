import json

from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


def load_user_keywords(user_id):
    recommender_methods = RecommenderMethods()
    recommender_methods.get_user_keywords(user_id)


def convert_to_json(df):
    predictions_json = df.to_json(orient="split")
    predictions_json_parsed = json.loads(predictions_json)
    return predictions_json_parsed


def load_user_categories(user_id):
    recommender_methods = RecommenderMethods()
    df_user_categories = recommender_methods.get_user_categories(user_id)
    df_user_categories = df_user_categories.rename(columns={'title': 'category_title'})
    if 'slug_y' in df_user_categories.columns:
        df_user_categories = df_user_categories.rename(columns={'slug_y': 'category_slug'})
    elif 'slug' in df_user_categories.columns:
        df_user_categories = df_user_categories.rename(columns={'slug': 'category_slug'})
    return df_user_categories


def load_ratings():
    # EXTRACT RESULTS FROM CURSOR
    recommender_methods = RecommenderMethods()
    posts_users_categories_ratings_df = recommender_methods.get_posts_users_categories_ratings_df(
        only_with_bert_vectors=True)
    return posts_users_categories_ratings_df


class UserBasedRecommendation:

    def __init__(self):
        self.database = None
        self.user_id = None

    def get_user_id(self):
        return self.user_id

    @DeprecationWarning
    def set_database(self, database):
        # noinspection DuplicatedCod
        self.database = database

    def get_database(self):
        return self.database

    # loads posts for user based on his searched_id and favourite categories
    def load_recommended_posts_for_user(self, user_id, num_of_recommendations=5):
        self.database = DatabaseMethods()

        # noinspection PyPep8
        df_posts_users_of_categories = load_ratings()[load_ratings()
            .category_slug.isin(load_user_categories(user_id)['category_slug'].tolist())]
        df_filter_current_user = df_posts_users_of_categories[
            df_posts_users_of_categories.rating_id != self.get_user_id()]
        df_sorted_results = df_filter_current_user[['post_id', 'post_slug', 'rating_value', 'post_created_at']] \
            .sort_values(['rating_value', 'post_created_at'], ascending=[False, False])
        df_sorted_results = df_sorted_results.drop_duplicates(subset=['post_id'])
        print("df_sorted_results[['post_slug']]")
        print(df_sorted_results[['post_id', 'post_slug']])
        return convert_to_json(df_sorted_results.head(num_of_recommendations))

    def get_user_keywords(self, user_id):
        pass


def main():
    # Testing
    user_based_recommendation = UserBasedRecommendation()
    print(user_based_recommendation.load_recommended_posts_for_user(211, 4))


# noinspection PyPep8
if __name__ == "__main__": main()
