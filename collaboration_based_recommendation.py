import json

from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


class Svd:

    def __init__(self):
        self.df_ratings = None
        self.df_users = None
        self.df_posts = None
        self.user_ratings_mean = None
        self.user_item_table = None  # = R_df_

    def get_user_item_from_db(self):
        database = DatabaseMethods()
        # EXTRACT RESULTS FROM CURSOR

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id AS user_id, u.name, r.value 
        AS ratings_values
                    FROM posts p
                    JOIN ratings r ON r.post_id = p.id
                    JOIN users u ON r.user_id = u.id;"""
        # LOAD INTO A DATAFRAME
        self.df_ratings = pd.read_sql_query(sql_rating, database.get_cnx())

        sql_select_all_users = """SELECT u.id AS user_id, u.name FROM users u;"""
        # LOAD INTO A DATAFRAME
        self.df_users = pd.read_sql_query(sql_select_all_users, database.get_cnx())

        sql_select_all_posts = """SELECT p.id AS post_id, p.slug FROM posts p;"""
        # LOAD INTO A DATAFRAME
        self.df_posts = pd.read_sql_query(sql_select_all_posts, database.get_cnx())

        user_item_table = self.combine_user_item(self.df_ratings)

        R_demeaned = self.convert_to_matrix(user_item_table)

        return R_demeaned

    def combine_user_item(self, df_rating):
        self.user_item_table = df_rating.pivot(index='user_id', columns='post_id', values='ratings_values').fillna(0)

        return self.user_item_table

    def convert_to_matrix(self, R_df):
        """
        self.user_ratings_mean = np.array(R_df.mean(axis=1))
        R_demeaned = R_df.sub(R_df.mean(axis=1), axis=0)
        R_demeaned = R_demeaned.fillna(0).values # values = new version of deprecated ,as_matrix()
        """
        R = R_df.values
        self.user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - self.user_ratings_mean.reshape(-1, 1)

        return R_demeaned

    def run_svd(self, user_id, num_of_recommendations=20):
        U, sigma, Vt = svds(self.get_user_item_from_db(), k=5)

        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean.reshape(-1, 1)
        # type: ignore

        print("all_user_predicted_ratings")
        print(all_user_predicted_ratings)

        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=self.user_item_table.columns)

        print("preds_df")
        print(preds_df)

        preds_df['user_id'] = self.user_item_table.index.values.tolist()
        preds_df.set_index('user_id', drop=True, inplace=True)  # inplace for making change in callable  way

        already_rated, predictions = self.recommend_posts(preds_df, user_id, self.df_posts, self.df_ratings,
                                                          num_of_recommendations)

        print("already_rated.head(num_of_recommendations)")
        print(already_rated.head(num_of_recommendations))

        print("List of predictions based on already rated items:")
        print(predictions.head(num_of_recommendations))

        predictions_json = predictions.to_json(orient="split")

        predictions_json_parsed = json.loads(predictions_json)

        return predictions_json_parsed

    @staticmethod
    def recommend_posts(predictions_df, user_id, posts_df, original_ratings_df, num_recommendations):
        # Get and sort the user's predictions
        user_row_number = user_id  # UserID starts at 1, not # 0

        sorted_user_predictions = predictions_df.loc[user_row_number].sort_values(ascending=False).to_frame()

        print("sorted_user_predictions")
        print(sorted_user_predictions)

        # Get the user's data and merge in the post information.
        user_data = original_ratings_df[original_ratings_df.user_id == user_id]
        user_full = (
            user_data.merge(posts_df, how='left', left_on='post_id', right_on='post_id').sort_values(['ratings_values'],
                                                                                                     ascending=False))

        # Recommend the highest predicted rating posts that the user hasn't seen yet.
        recommendations = (posts_df[~posts_df['post_id'].isin(user_full['post_id'])]
                               .merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                                      how='left',
                                      left_on='post_id',
                                      right_on='post_id')
                               .rename(columns={user_row_number: 'Predictions'})
                               .sort_values('Predictions', ascending=False)
                               .iloc[:num_recommendations, :-1])

        return user_full, recommendations


def main():
    svd = Svd()
    print(svd.run_svd(431))


if __name__ == "__main__": main()
