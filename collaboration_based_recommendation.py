import json

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
        self.user_item_table = None # = R_df_

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
        preds_df.set_index('user_id', drop=True, inplace=True) # inplace for making change in callable  way

        already_rated, predictions = self.recommend_posts(preds_df, user_id, self.df_posts, self.df_ratings, num_of_recommendations)

        print("already_rated.head(num_of_recommendations)")
        print(already_rated.head(num_of_recommendations))

        print("List of predictions based on already rated items:")
        print(predictions.head(num_of_recommendations))

        predictions_json = predictions.to_json(orient="split")

        predictions_json_parsed = json.loads(predictions_json)

        return predictions_json_parsed

    def recommend_posts(self, predictions_df, user_id, posts_df, original_ratings_df, num_recommendations):
        # Get and sort the user's predictions
        user_row_number = user_id # UserID starts at 1, not # 0

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

        # print("user_data")
        # print(user_data)
        # print("user_full")
        # print(user_full)

        # print('User {0} has already rated {1} posts.'.format(user_id, user_full.shape[0]))
        # print('Recommending the highest {0} predicted ratings posts not already rated.'.format(num_recommendations))

        # print("sorted_user_predictions type:")
        # print(type(sorted_user_predictions))

        """
        arr = [np.array([posts_df[k] == v for k, v in x.items()]).all(axis=0) for x in sorted_user_predictions.to_dict('r')]
        recommendations = posts_df[np.array(arr).any(axis=0)]
        """
        """


        for predicted_post in sorted_user_predictions
            recommendations_own_version = post_df[predicted_post[0]]
        """
        # Recommend the highest predicted rating posts that the user hasn't seen yet.
        recommendations = (posts_df[~posts_df['post_id'].isin(user_full['post_id'])]
                               .merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                                     how='left',
                                     left_on='post_id',
                                     right_on='post_id')
                                .rename(columns={user_row_number: 'Predictions'})
                               .sort_values('Predictions', ascending=False)
                               .iloc[:num_recommendations, :-1])

        # print("recommendations:")
        # print(recommendations)

        return user_full, recommendations

def main():
    svd = Svd()
    print(svd.run_svd(431))

if __name__ == "__main__": main()
