import json
import operator

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD, KNNBasic
from surprise.model_selection import cross_validate

from src.recommender_core.data_handling.data_queries import RecommenderMethods


def get_average_post_rating():
    database = DatabaseMethods()
    # # Step 1
    # database.set_row_var()
    # EXTRACT RESULTS FROM CURSOR

    sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id AS user_id, u.name, r.value AS rating_value
                       FROM posts p
                       JOIN ratings r ON r.post_id = p.id
                       JOIN users u ON r.user_id = u.id;"""
    # LOAD INTO A DATAFRAME
    recommender_methods = RecommenderMethods()
    all_posts_df = recommender_methods.get_posts_dataframe()
    df_ratings = pd.read_sql_query(sql_rating, database.get_cnx())
    sql_select_all_users = """SELECT u.id AS user_id, u.name FROM users u;"""
    # LOAD INTO A DATAFRAME
    df_users = pd.read_sql_query(sql_select_all_users, database.get_cnx())
    # print("Users")
    # print(self.df_users)
    sql_select_all_posts = """SELECT p.id AS post_id, p.slug FROM posts p;"""
    # LOAD INTO A DATAFRAME
    df_posts = pd.read_sql_query(sql_select_all_posts, database.get_cnx())
    # print("Posts:")
    # print(self.df_posts)
    print("df_ratings")
    print(df_ratings.to_string())
    ratings_means = df_ratings.groupby("slug")["rating_value"].mean()
    print("df_ratings_means")
    print(ratings_means)
    df_ratings_means = pd.DataFrame({'slug': ratings_means.index, 'rating_value': ratings_means.values}).set_index(
        'slug')
    df_ratings_means_list = []
    print("df_ratings_means")
    print(df_ratings_means.to_string())
    for slug_index, row in df_ratings_means.iterrows():
        df_ratings_means_list.append({'slug': slug_index, 'coefficient': row['rating_value']})
    df_ratings_means_list_sorted = sorted(df_ratings_means_list, key=lambda d: d['coefficient'], reverse=True)

    all_posts_df = all_posts_df.set_index("slug")
    print("all_posts_df")
    print(all_posts_df.head().to_string())
    print("df_ratings_means")
    print(df_ratings_means.to_string())
    print(all_posts_df.columns)
    print("all_posts_df.columns")
    print(all_posts_df.columns)
    all_posts_df = all_posts_df.reset_index()
    all_posts_df = all_posts_df[['slug']]
    all_posts_df = all_posts_df.set_index('slug')
    all_posts_df.to_csv("exports/all_posts_df.csv")
    df_ratings_means.to_csv("exports/df_ratings_means.csv")
    all_posts_df_means = pd.merge(all_posts_df, df_ratings_means, left_index=True, right_index=True, how="left")
    # noinspection PyTypeChecker
    all_posts_df_means.to_csv("exports/all_posts_df_means.csv")
    print("all_posts_df_means.columns")
    print(all_posts_df_means.columns)
    all_posts_df_means = all_posts_df_means[['rating_value']]
    all_posts_df_means = all_posts_df_means[['rating_value']].fillna(0)
    print(all_posts_df_means.to_string())
    all_posts_df_means_list = []
    for slug_index, row in all_posts_df_means.iterrows():
        all_posts_df_means_list.append({'slug': slug_index, 'coefficient': row['rating_value']})
    print("all_posts_df_means_list")
    print(all_posts_df_means_list)
    with open('../../../../datasets/exports/all_posts_df_means_list.txt', 'w') as f:
        f.write(str(all_posts_df_means_list))
    # all_posts_df_means_list_sorted = sorted(all_posts_df_means_list, key=lambda d: d['coefficient'], reverse=True)
    all_posts_df_means_list.sort(key=operator.itemgetter('coefficient'), reverse=True)
    print("all_posts_df_means_list")
    print(all_posts_df_means_list)
    return df_ratings_means_list_sorted, all_posts_df_means_list


def cross_validate_dataframe(ratings, users_id):
    print("ratings.to_string()")
    print(ratings.to_string())
    ratings = ratings.drop(columns='created_at')
    ratings = ratings.drop(columns='updated_at')
    ratings = ratings.drop(columns='id')
    reader = Reader()  # dataset creation
    data = Dataset.load_from_df(ratings, reader)
    knn = KNNBasic()  # Evaluating the performance in terms of RMSE
    cross_validate(knn, data, measures=['RMSE', 'mae'], cv=3)
    # Define the SVD method object
    svd = SVD()  # Evaluate the performance in terms of RMSE
    cross_validate(svd, data, measures=['RMSE'], cv=3)
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    print("ratings[ratings['user_id'] == users_id]")
    print(ratings[ratings['user_id'] == users_id])
    print(svd.predict(users_id, 734909))


def recommend_posts(predictions_df, user_id, posts_df, original_ratings_df, num_recommendations):
    # Get and sort the user's predictions
    user_row_number = user_id  # UserID starts at 1, not # 0

    print("predictions_df:")
    print(predictions_df.to_string())
    print("original_ratings_df:")
    print(original_ratings_df.to_string())
    if user_id not in original_ratings_df['user_id'].values:
        raise ValueError
    sorted_user_predictions = predictions_df.loc[user_row_number].sort_values(ascending=False).to_frame()

    print("sorted_user_predictions")
    print(sorted_user_predictions)

    # Get the user's data and merge in the post information.
    user_data = original_ratings_df[original_ratings_df.user_id == (user_id)]
    user_full = (user_data.merge(posts_df, how='left', left_on='post_id', right_on='post_id').
                 sort_values(['rating_value'], ascending=False)
                 )
    print("user_full.to_string()")
    print(user_full.to_string())
    # Recommend the highest predicted rating posts that the user hasn't rated yet.
    recommendations = (posts_df[~posts_df['post_id'].isin(user_full['post_id'])]
                           .merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                                  how='left',
                                  left_on='post_id',
                                  right_on='post_id')
                           .rename(columns={user_row_number: 'rating_value'})
                           .sort_values('rating_value', ascending=False)
                           .iloc[:num_recommendations, :]
                           )
    print("recommendations")
    print(recommendations.to_string())
    return user_full, recommendations


def calculate_ratings(id_post, id_user, df_ratings, similarity_matrix_df):
    if id_post in df_ratings:
        cosine_scores = similarity_matrix_df[id_user]  # similarity of id_user with every other user
        ratings_scores = df_ratings[
            id_post]  # ratings of every other user for the post id_post won't consider users who havent rated id_post so drop similarity scores and ratings corresponsing to np.nan
        index_not_rated = ratings_scores[ratings_scores.isnull()].index
        ratings_scores = ratings_scores.dropna()
        cosine_scores = cosine_scores.drop(
            index_not_rated)  # calculating rating by weighted mean of ratings and cosine scores of the users who have rated the post
        ratings_post = np.dot(ratings_scores, cosine_scores) / cosine_scores.sum()
    else:
        return 2.5
    return ratings_post


class SvdClass:

    def __init__(self):
        self.df_ratings = None
        self.df_users = None
        self.df_posts = None
        self.user_ratings_mean = None
        self.user_item_table = None  # = R_df_

    def get_all_users_ids(self):
        database = DatabaseMethods()
        sql_select_all_users = """SELECT u.id AS user_id, u.name FROM users u;"""
        # LOAD INTO A DATAFRAME
        self.df_users = pd.read_sql_query(sql_select_all_users, database.get_cnx())
        return self.df_users

    def get_user_item_from_db(self):
        database = DatabaseMethods()
        # Step 1
        # database.set_row_var()
        # EXTRACT RESULTS FROM CURSOR

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id AS user_id, u.name, 
        r.value AS rating_value FROM posts p JOIN ratings r ON r.post_id = p.id JOIN users u ON r.user_id = u.id;"""
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
        # print("Posts:")
        # print(self.df_posts)
        user_item_table = self.combine_user_item(self.df_ratings)
        R_demeaned = self.convert_to_matrix(user_item_table)
        return R_demeaned

    # noinspection DuplicatedCode
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

    # @profile
    def run_svd(self, user_id, num_of_recommendations=10):
        all_user_predicted_ratings = self.get_all_users_predicted_ratings()
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=self.user_item_table.columns)

        print("preds_df")
        print(preds_df)

        preds_df['user_id'] = self.user_item_table.index.values.tolist()
        preds_df.set_index('user_id', drop=True, inplace=True)  # inplace for making change in callable  way
        already_rated, predictions = recommend_posts(preds_df, user_id, self.df_posts, self.df_ratings,
                                                     num_of_recommendations)
        print("already_rated.head(num_of_recommendations)")
        print(already_rated.head(num_of_recommendations).to_string())
        print("List of predictions based on already rated items:")
        print(predictions.head(num_of_recommendations).to_string())
        predictions_json = predictions.to_json(orient="split")
        predictions_json_parsed = json.loads(predictions_json)
        return predictions_json_parsed

    def rmse(self, user_id):
        recommenderMethods = RecommenderMethods()
        column_names = ['user_id', 'post_id', 'rating_value', 'slug']
        posts_df = recommenderMethods.get_posts_dataframe()  # needs to be separated - posts, users, ratings
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

        print(calculate_ratings(user_id, 704691, df_ratings_dummy, similarity_matrix_df))
        test_set_score = self.score_on_test_set(X_test, df_ratings_dummy, similarity_matrix_df)
        print(test_set_score)
        print(cross_validate_dataframe(ratings, user_id))

    # noinspection DuplicatedCode

    # noinspection DuplicatedCode
    def score_on_test_set(self, X_test, df_ratings, similarity_matrix_df):
        user_post_pairs = zip(X_test['post_id'], X_test['user_id'])
        predicted_ratings = np.array(
            [calculate_ratings(post, user, df_ratings, similarity_matrix_df) for (post, user) in user_post_pairs])
        true_ratings = np.array(X_test['value'])
        score = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
        return score

    def get_all_users_predicted_ratings(self):
        U, sigma, Vt = svds(self.get_user_item_from_db(), k=5)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean.reshape(-1, 1)
        return all_user_predicted_ratings

    def rmse_all_users(self):
        all_user_predicted_ratings = self.get_all_users_predicted_ratings()
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
        already_rated_by_users = already_rated_by_users \
            .pivot(index='user_id', columns='post_id', values='value').fillna(0)
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
        cols = already_rated_by_users.columns
        bt = already_rated_by_users.apply(lambda x: x > 0)
        bt.apply(lambda x: list(cols[x.values]), axis=1)
        print("bt")
        print(bt)

        predicted_array = []
        actual_array = []

        for index, row in already_rated_by_users.iterrows():
            print("index")
            print(index)
            print("row")
            print(row)
            for column, value in row.items():
                # print(column)
                if value != 0:
                    print(predictions_df.at[index, column])
                    predicted_array.append(round(predictions_df.at[index, column], 1))
                    actual_array.append(value)

        print(predicted_array)
        print(actual_array)

        rmse_without_zero = mean_squared_error(actual_array, predicted_array, squared=True)

        print("RMSE ignoring missing values:")
        print(rmse_without_zero)


def main():
    # svd_class = SvdClass()
    # print(svd_class.run_svd(431))
    # print(svd.rmse_all_users())
    """
    rated, all = svd.get_average_post_rating()
    print("rated:")
    print(rated)
    print("all:")
    print(all)
    """


if __name__ == "__main__": main()
