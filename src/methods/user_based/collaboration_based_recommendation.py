import json

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from src.data_handling.data_manipulation import DatabaseMethods
from src.data_handling.model_methods.user_methods import UserMethods


def recommend_posts(predictions_df, user_id, posts_df, original_ratings_df, num_recommendations):
    # Get and sort the evalutation's predictions
    user_row_number = user_id  # UserID starts at 1, not # 0

    if user_id not in original_ratings_df['user_id'].values:
        raise ValueError("User id not found dataframe of original ratings.")
    sorted_user_predictions = predictions_df.loc[user_row_number].sort_values(ascending=False).to_frame()

    # Get the evalutation's data and merge in the post information.
    user_data = original_ratings_df[original_ratings_df.user_id == user_id]
    user_full = (
        user_data.merge(posts_df, how='left', left_on='post_id', right_on='post_id').
        sort_values(['ratings_values'], ascending=False)
    )
    # Recommend the highest predicted rating posts that the evalutation hasn't rated yet.
    # noinspection PyPep8
    recommendations = (posts_df[~posts_df['post_id'].isin(user_full['post_id'])]
                       .merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left', left_on='post_id',
                              right_on='post_id')
                       .rename(columns={user_row_number: 'ratings_values'})
                       .sort_values('ratings_values', ascending=False).iloc[:num_recommendations, :])
    return user_full, recommendations


# *** HERE were tuning methods, e.g., RMSE. ABANDONED DUE TO: no labels to use

class SvdClass:

    def __init__(self):
        self.user_ratings_mean = None
        self.df_ratings = pd.DataFrame()
        self.df_users = pd.DataFrame()
        self.df_posts = pd.DataFrame()
        self.user_ratings_mean: np.ndarray
        self.user_item_table = pd.DataFrame()

    def get_all_users_ids(self):
        database = DatabaseMethods()
        sql_select_all_users = """SELECT u.id AS user_id, u.name FROM users u;"""
        # LOAD INTO A DATAFRAME
        self.df_users = pd.read_sql_query(sql_select_all_users, database.get_cnx())
        return self.df_users

    def get_user_item_from_db(self):

        user_methods = UserMethods()
        self.df_posts, self.df_users, self.df_ratings = user_methods.get_posts_df_users_df_ratings_df()
        user_item_table = self.combine_user_item(self.df_ratings)
        # noinspection PyPep8Naming
        R_demeaned = self.convert_to_matrix(user_item_table)
        return R_demeaned

    # noinspection DuplicatedCode
    def combine_user_item(self, df_rating):

        self.user_item_table = df_rating.pivot(index='user_id', columns='post_id', values='ratings_values').fillna(0)

        return self.user_item_table

    # noinspection PyPep8Naming
    def convert_to_matrix(self, R_df):
        """
        self.user_ratings_mean = np.array(R_df.mean(axis=1))
        R_demeaned = R_df.sub(R_df.mean(axis=1), axis=0)
        R_demeaned = R_demeaned.fillna(0).values # values = new version of deprecated ,as_matrix()
        """
        # noinspection PyPep8Naming
        R = R_df.values
        self.user_ratings_mean = np.mean(R, axis=1)
        # noinspection PyPep8Naming
        R_demeaned = R - self.user_ratings_mean.reshape(-1, 1)

        return R_demeaned

    def prepare_predictions(self, all_user_predicted_ratings):
        if self.user_item_table is not None:
            preds_df = pd.DataFrame(all_user_predicted_ratings, columns=self.user_item_table.columns)
        else:
            raise ValueError("user_item_table is None, cannot continue with next operation.")
        print("preds_df")
        print(preds_df)

        preds_df['user_id'] = self.user_item_table.index.values.tolist()
        preds_df.set_index('user_id', drop=True, inplace=True)  # inplace for making change in callable way

        return preds_df

    # @profile
    def run_svd(self, user_id: int, num_of_recommendations=10, dict_results=True):
        """

        @param dict_results: bool to determine whether you need JSON or rather Pandas Dataframe
        @param user_id: int corresponding to evalutation's id from DB
        @param num_of_recommendations: number of returned recommended items
        @return: Dict/JSON of posts recommended for a give evalutation or dataframe of recommenmded posrts according to
        json_results bool aram
        """
        all_user_predicted_ratings = self.get_all_users_predicted_ratings()
        preds_df = self.prepare_predictions(all_user_predicted_ratings)

        if self.df_posts is not None and self.df_ratings is not None:
            _, predictions = recommend_posts(preds_df, user_id, self.df_posts, self.df_ratings,
                                             num_of_recommendations)
        else:
            raise ValueError("Dataframe of posts is None. Cannot continue with next operation.")
        if dict_results is True:
            predictions_json = predictions.to_json(orient="split")
            predictions_json_parsed = json.loads(predictions_json)
            return predictions_json_parsed
        else:
            return predictions.head(num_of_recommendations)

    def get_all_users_predicted_ratings(self):
        # noinspection PyPep8Naming
        U, sigma, Vt = svds(self.get_user_item_from_db(), k=5)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean.reshape(-1, 1)
        return all_user_predicted_ratings
