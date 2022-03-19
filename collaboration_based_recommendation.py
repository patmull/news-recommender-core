import json
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from content_based_algorithms.data_queries import RecommenderMethods
from data_conenction import Database
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD, KNNBasic
from surprise.model_selection import cross_validate


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
        print("df_rating")
        print(df_rating)
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

    def rmse(self, user_id):

        recommenderMethods = RecommenderMethods()
        column_names = ['user_id','post_id','rating_value','slug']
        posts_df = recommenderMethods.get_posts_dataframe() # needs to be separated - posts, users, ratings
        users_df = recommenderMethods.get_users_dataframe()
        ratings = recommenderMethods.get_ratings_dataframe()
        # combined_posts_data = df[['user_id','post_id','rating_value','slug']]
        # print(combined_posts_data.head().to_string())

        X = ratings.copy()
        y = ratings['user_id']

        # combined_posts_data = combined_posts_data[['rating_value','user_id']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,
                                                            shuffle=True,
                                                            )
        df_ratings = X_train.pivot(index='user_id', columns='post_id', values='value')
        print("df_user_item.head()")
        print(df_ratings.to_string())

        df_ratings_dummy = df_ratings.copy().fillna(0)
        print("df_ratings_dummy.head()")
        print(df_ratings_dummy.head())

        similarity_matrix = cosine_similarity(df_ratings_dummy, df_ratings_dummy)
        similarity_matrix_df = pd.DataFrame(similarity_matrix, index=df_ratings.index, columns=df_ratings.index)

        print(self.calculate_ratings(user_id,704691,df_ratings_dummy,similarity_matrix_df))
        test_set_score = self.score_on_test_set(X_test, df_ratings_dummy, similarity_matrix_df)
        print(test_set_score)
        print(self.cross_validate_dataframe(ratings, user_id))


    # cosine similarity of the ratingssimilarity_matrix = cosine_similarity(df_ratings_dummy, df_ratings_dummy)similarity_matrix_df = pd.DataFrame(similarity_matrix, index=df_ratings.index, columns=df_ratings.index)#calculate ratings using weighted sum of cosine similarity#function to calculate ratings
    def calculate_ratings(self, id_post, id_user, df_ratings, similarity_matrix_df):
            if id_post in df_ratings:
                cosine_scores = similarity_matrix_df[id_user]  # similarity of id_user with every other user
                ratings_scores = df_ratings[id_post] #ratings of every other user for the post id_post won't consider users who havent rated id_post so drop similarity scores and ratings corresponsing to np.nan
                index_not_rated = ratings_scores[ratings_scores.isnull()].index
                ratings_scores = ratings_scores.dropna()
                cosine_scores = cosine_scores.drop(index_not_rated) #calculating rating by weighted mean of ratings and cosine scores of the users who have rated the post
                ratings_post = np.dot(ratings_scores, cosine_scores)/cosine_scores.sum()
            else:
                return 2.5
            return ratings_post

    def score_on_test_set(self, X_test, df_ratings, similarity_matrix_df):
        user_post_pairs = zip(X_test['post_id'], X_test['user_id'])
        predicted_ratings = np.array(
            [self.calculate_ratings(post, user, df_ratings, similarity_matrix_df) for (post, user) in user_post_pairs])
        true_ratings = np.array(X_test['value'])
        score = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
        return score

    def cross_validate_dataframe(self, ratings, users_id):
        print("ratings.to_string()")
        print(ratings.to_string())
        ratings = ratings.drop(columns='created_at')
        ratings = ratings.drop(columns='updated_at')
        ratings = ratings.drop(columns='id')
        reader = Reader()  # dataset creation
        data = Dataset.load_from_df(ratings, reader)
        knn = KNNBasic() #Evaluating the performance in terms of RMSE
        cross_validate(knn, data,  measures=['RMSE', 'mae'], cv=3)
        # Define the SVD algorithm object
        svd = SVD() #Evaluate the performance in terms of RMSE
        cross_validate(svd, data, measures=['RMSE'], cv=3)
        trainset = data.build_full_trainset()
        svd.fit(trainset)
        print("ratings[ratings['user_id'] == users_id]")
        print(ratings[ratings['user_id'] == users_id])
        print(svd.predict(users_id, 734909))

    def rmse_all_users(self):
        U, sigma, Vt = svds(self.get_user_item_from_db(), k=5)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean.reshape(-1, 1)
        print("all_user_predicted_ratings")
        print(all_user_predicted_ratings)
        print(all_user_predicted_ratings.size)

        predictions_df = pd.DataFrame(all_user_predicted_ratings, columns=self.user_item_table.columns)
        print("preds_df")
        print(predictions_df)
        predictions_df['user_id'] = self.user_item_table.index.values.tolist()
        predictions_df.set_index('user_id', drop=True, inplace=True)  # inplace for making change in callable  way
        # Get and sort the user's predictions
        # sorted_users_predictions = pd.DataFrame()
        user_ids = self.get_all_users_ids()
        predictions_results = pd.DataFrame()
        print(user_ids)
        print("predictions_df")
        print(predictions_df)

        recommenderMethods = RecommenderMethods()
        already_rated_by_users = recommenderMethods.get_ratings_dataframe()
        print("already_rated_by_users")
        print(already_rated_by_users)
        # already_rated_by_users.set_index('user_id', inplace=True)
        already_rated_by_users = already_rated_by_users.pivot(index='user_id', columns='post_id', values='value').fillna(0)
        print("already_rated_by_users")
        print(already_rated_by_users)
        print("predictions_df")
        print(predictions_df)

        already_rated_matrix = already_rated_by_users.to_numpy()
        predictions_matrix = predictions_df.to_numpy()

        rmse = mean_squared_error(already_rated_matrix, predictions_matrix, squared=True)

        print("RMSE with 0 on missing values:")
        print(rmse)

        # Any possibility to deal with missing  values???
        rmse = mean_squared_error(already_rated_matrix, predictions_matrix, squared=True)

        print("RMSE ignoring missing values:")
        print(rmse)

def main():

    svd = Svd()
    # print(svd.run_svd(431))
    print(svd.rmse_all_users())


if __name__ == "__main__": main()
