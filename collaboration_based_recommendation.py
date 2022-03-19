import json

from sklearn.metrics import mean_squared_error

from content_based_algorithms.data_queries import RecommenderMethods
from data_conenction import Database
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd


class Svd:

    def __init__(self):
        self.df_ratings = None
        self.df_users = None
        self.df_posts = None
        self.user_ratings_mean = None
        self.user_item_table = None  # = R_df_

    def get_all_users_ids(self):
        database = Database()
        database.connect()
        sql_select_all_users = """SELECT u.id AS user_id, u.name FROM users u;"""
        # LOAD INTO A DATAFRAME
        self.df_users = pd.read_sql_query(sql_select_all_users, database.get_cnx())
        return self.df_users

    def get_user_item_from_db(self):
        database = Database()
        database.connect()
        ##Step 1
        # database.set_row_var()
        # EXTRACT RESULTS FROM CURSOR

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id AS user_id, u.name, r.value AS rating_value
                    FROM posts p
                    JOIN ratings r ON r.post_id = p.id
                    JOIN users u ON r.user_id = u.id;"""
        # LOAD INTO A DATAFRAME
        self.df_ratings = pd.read_sql_query(sql_rating, database.get_cnx())

        sql_select_all_users = """SELECT u.id AS user_id, u.name FROM users u;"""
        # LOAD INTO A DATAFRAME
        self.df_users = pd.read_sql_query(sql_select_all_users, database.get_cnx())

        # print("Users")
        # print(self.df_users)

        sql_select_all_posts = """SELECT p.id AS post_id, p.slug FROM posts p;"""
        # LOAD INTO A DATAFRAME
        self.df_posts = pd.read_sql_query(sql_select_all_posts, database.get_cnx())

        database.disconnect()
        # print("Posts:")
        # print(self.df_posts)

        user_item_table = self.combine_user_item(self.df_ratings)

        R_demeaned = self.convert_to_matrix(user_item_table)

        # print("R_demeaned:")
        # print(R_demeaned)

        return R_demeaned

    def combine_user_item(self, df_rating):
        # self.user_item_table = df_rating.pivot(index='user_id', columns='post_id', values='rating_value')
        self.user_item_table = df_rating.pivot(index='user_id', columns='post_id', values='rating_value').fillna(0)

        # print("User item matrix:")
        # print(self.user_item_table.to_string())

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

    def run_svd(self, user_id, num_of_recommendations=10):
        U, sigma, Vt = svds(self.get_user_item_from_db(), k=5)
        # print("U:")
        # print(U)
        # print("----------")
        # print("Sigma:")
        # print(sigma)
        # print("-----------")
        # print("Vt:")
        # print(Vt)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean.reshape(-1, 1)

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
        print(already_rated.head(num_of_recommendations).to_string())

        print("List of predictions based on already rated items:")
        print(predictions.head(num_of_recommendations).to_string())

        predictions_json = predictions.to_json(orient="split")

        predictions_json_parsed = json.loads(predictions_json)

        return predictions_json_parsed

    def recommend_posts(self, predictions_df, user_id, posts_df, original_ratings_df, num_recommendations):
        # Get and sort the user's predictions
        user_row_number = user_id  # UserID starts at 1, not # 0

        # print("predictions_df:")
        # print(predictions_df.to_string())
        sorted_user_predictions = predictions_df.loc[user_row_number].sort_values(ascending=False).to_frame()

        print("sorted_user_predictions")
        print(sorted_user_predictions)

        # Get the user's data and merge in the post information.
        user_data = original_ratings_df[original_ratings_df.user_id == (user_id)]
        user_full = (user_data.merge(posts_df, how='left', left_on='post_id', right_on='post_id').
                     sort_values(['rating_value'], ascending=False)
                     )

        # Recommend the highest predicted rating posts that the user hasn't rated yet.
        recommendations = (posts_df[~posts_df['post_id'].isin(user_full['post_id'])]
                               .merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                                      how='left',
                                      left_on='post_id',
                                      right_on='post_id')
                               .rename(columns={user_row_number: 'Predictions'})
                               .sort_values('Predictions', ascending=False)
                               .iloc[:num_recommendations, :-1])

        return user_full, recommendations

    def rmse(self):
        U, sigma, Vt = svds(self.get_user_item_from_db(), k=5)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean.reshape(-1, 1)

        print("all_user_predicted_ratings")
        print(all_user_predicted_ratings)

        predictions_df = pd.DataFrame(all_user_predicted_ratings, columns=self.user_item_table.columns)

        print("preds_df")
        print(predictions_df)

        predictions_df['user_id'] = self.user_item_table.index.values.tolist()
        predictions_df.set_index('user_id', drop=True, inplace=True)  # inplace for making change in callable  way
        # Get and sort the user's predictions
        #sorted_users_predictions = pd.DataFrame()
        already_rated_by_users = pd.DataFrame()
        user_ids = self.get_all_users_ids()
        predictions_results = pd.DataFrame()
        print(user_ids)
        print("predictions_df")
        print(predictions_df)

        all_users_predictions = pd.DataFrame()

        # loop is inefficient better to add multiple dataframes at once using same append
        for user_id in predictions_df.index.get_level_values('user_id'): # UserID starts at 1, not # 0
            sorted_users_predictions = predictions_df.loc[user_id].sort_values(ascending=False).to_frame()
            sorted_users_predictions.reset_index(inplace=True)
            sorted_users_predictions = sorted_users_predictions.rename(columns={user_id: 'rating_value'})
            print("sorted_users_predictions")
            print(sorted_users_predictions)
            print(sorted_users_predictions[['post_id']])
            print(sorted_users_predictions[['rating_value']])
            # print(sorted_user_predictions[["post_id"]])
            # print(sorted_user_predictions[["rating_value"]])
            all_users_predictions = all_users_predictions.append(sorted_users_predictions)
            print("all_users_predictions:")
            print(all_users_predictions)
            already_rated, predictions = self.recommend_posts(predictions_df, user_id, self.df_posts, self.df_ratings,
                                                              10)
            already_rated_by_users.append(already_rated)
        # print("predictions_df:")
        # print(predictions_df.to_string())

        print("all_users_predictions:")
        print(all_users_predictions)

        already_rated = already_rated_by_users[['post_id', 'rating_value']]
        print("already_rated")
        print(already_rated[["post_id"]])
        print(already_rated[["rating_value"]])

        dataframe_predicted_actual = pd.merge(sorted_users_prediction, already_rated, how='inner', on=['post_id'])
        print("dataframe_predicted_actual")
        print(dataframe_predicted_actual)

        print("RMSE:")
        print(np.sqrt(mean_squared_error(dataframe_predicted_actual[['rating_value_y']],
                                         dataframe_predicted_actual[['rating_value_x']])))


def main():
    svd = Svd()
    # print(svd.run_svd(431))
    print(svd.rmse())


if __name__ == "__main__": main()
