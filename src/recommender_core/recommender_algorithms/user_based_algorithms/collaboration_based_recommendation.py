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
from src.recommender_core.data_handling.model_methods.user_methods import UserMethods


def get_average_post_rating():
    database = DatabaseMethods()
    # # Step 1
    # database.set_row_var()
    # EXTRACT RESULTS FROM CURSOR

    sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id AS user_id, u.name, 
    r.value AS ratings_values FROM posts p JOIN ratings r ON r.post_id = p.id JOIN users u ON r.user_id = u.id;"""
    # LOAD INTO A DATAFRAME
    recommender_methods = RecommenderMethods()
    all_posts_df = recommender_methods.get_posts_dataframe()
    df_ratings = pd.read_sql_query(sql_rating, database.get_cnx())

    print("df_ratings")
    print(df_ratings)
    ratings_means = df_ratings.groupby("slug")["ratings_values"].mean()
    print("df_ratings_means")
    print(ratings_means)
    df_ratings_means = pd.DataFrame({'slug': ratings_means.index, 'ratings_values': ratings_means.values}).set_index(
        'slug')
    df_ratings_means_list = []
    print("df_ratings_means")
    print(df_ratings_means)
    for slug_index, row in df_ratings_means.iterrows():
        df_ratings_means_list.append({'slug': slug_index, 'coefficient': row['ratings_values']})
    df_ratings_means_list_sorted = sorted(df_ratings_means_list, key=lambda d: d['coefficient'], reverse=True)

    all_posts_df = all_posts_df.set_index("slug")
    print("all_posts_df")
    print(all_posts_df.head())
    print("df_ratings_means")
    print(df_ratings_means)
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
    all_posts_df_means = all_posts_df_means[['ratings_values']]
    all_posts_df_means = all_posts_df_means[['ratings_values']].fillna(0)
    print(all_posts_df_means)
    all_posts_df_means_list = []
    for slug_index, row in all_posts_df_means.iterrows():
        all_posts_df_means_list.append({'slug': slug_index, 'coefficient': row['ratings_values']})
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
    print("ratings")
    print(ratings)
    ratings = ratings.drop(columns='created_at')
    ratings = ratings.drop(columns='updated_at')
    ratings = ratings.drop(columns='searched_id')
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
    print(predictions_df)
    print("original_ratings_df:")
    print(original_ratings_df)
    print("user_id:")
    print(user_id)
    print("original_ratings_df['user_id'].values:")
    print(original_ratings_df['user_id'].values)
    if user_id not in original_ratings_df['user_id'].values:
        raise ValueError("User id not found dataframe of original ratings.")
    sorted_user_predictions = predictions_df.loc[user_row_number].sort_values(ascending=False).to_frame()

    print("sorted_user_predictions")
    print(sorted_user_predictions)

    # Get the user's data and merge in the post information.
    user_data = original_ratings_df[original_ratings_df.user_id == user_id]
    user_full = (user_data.merge(posts_df, how='left', left_on='post_id', right_on='post_id').
                 sort_values(['ratings_values'], ascending=False)
                 )
    print("user_full")
    print(user_full)
    # Recommend the highest predicted rating posts that the user hasn't rated yet.
    # noinspection PyPep8
    recommendations = (posts_df[~posts_df['post_id'].isin(user_full['post_id'])]
                           .merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left', left_on='post_id',
                                  right_on='post_id')
                           .rename(columns={user_row_number: 'ratings_values'})
                           .sort_values('ratings_values', ascending=False).iloc[:num_recommendations, :])
    print("recommendations")
    print(recommendations)
    return user_full, recommendations


def calculate_ratings(id_post, id_user, df_ratings, similarity_matrix_df):
    if id_post in df_ratings:
        cosine_scores = similarity_matrix_df[id_user]  # similarity of id_user with every other user
        ratings_scores = df_ratings[id_post]
        # ratings of every other user for the post id_post won't consider users who havent rated
        # id_post so drop similarity scores and ratings corresponsing to np.nan
        index_not_rated = ratings_scores[ratings_scores.isnull()].index
        ratings_scores = ratings_scores.dropna()
        cosine_scores = cosine_scores.drop(index_not_rated)
        # calculating rating by weighted mean of ratings and cosine scores of the users who
        # have rated the post
        ratings_post = np.dot(ratings_scores, cosine_scores) / cosine_scores.sum()
    else:
        return 2.5
    return ratings_post


def score_on_test_set(X_test, df_ratings, similarity_matrix_df):
    user_post_pairs = zip(X_test['post_id'], X_test['user_id'])
    predicted_ratings = np.array(
        [calculate_ratings(post, user, df_ratings, similarity_matrix_df) for (post, user) in user_post_pairs])
    true_ratings = np.array(X_test['value'])
    score = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    return score


def rmse(user_id):
    recommender_methods = RecommenderMethods()
    ratings = recommender_methods.get_ratings_dataframe()
    # noinspection PyPep8Naming
    X = ratings.copy()
    y = ratings['user_id']

    # noinspection PyPep8Naming
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,
                                                        shuffle=True,
                                                        )
    df_ratings = X_train.pivot(index='user_id', columns='post_id', values='value')
    print("df_user_item.head()")
    print(df_ratings)

    df_ratings_dummy = df_ratings.copy().fillna(0)
    print("df_ratings_dummy.head()")
    print(df_ratings_dummy.head())

    similarity_matrix = cosine_similarity(df_ratings_dummy, df_ratings_dummy)
    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=df_ratings.index, columns=df_ratings.index)

    print(calculate_ratings(user_id, 704691, df_ratings_dummy, similarity_matrix_df))
    test_set_score = score_on_test_set(X_test, df_ratings_dummy, similarity_matrix_df)
    print(test_set_score)
    print(cross_validate_dataframe(ratings, user_id))


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

        user_methods = UserMethods()
        self.df_posts, self.df_users, self.df_ratings = user_methods.get_posts_df_users_df_ratings_df()
        user_item_table = self.combine_user_item(self.df_ratings)
        # noinspection PyPep8Naming
        R_demeaned = self.convert_to_matrix(user_item_table)
        return R_demeaned

    # noinspection DuplicatedCode
    def combine_user_item(self, df_rating):
        # self.user_item_table = df_rating.pivot(index='user_id', columns='post_id', values='ratings_values')
        print("df_rating")
        print(df_rating)
        self.user_item_table = df_rating.pivot(index='user_id', columns='post_id', values='ratings_values').fillna(0)

        # print("User item matrix:")
        # print(self.user_item_table)

        return self.user_item_table

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
    def run_svd(self, user_id : int, num_of_recommendations=10, dict_results=True):
        """

        @param dict_results: bool to determine whether you need JSON or rather Pandas Dataframe
        @param user_id: int corresponding to user's id from DB
        @param num_of_recommendations: number of returned recommended items
        @return: Dict/JSON of posts recommended for a give user or dataframe of recommenmded posrts according to
        json_results bool aram
        """
        all_user_predicted_ratings = self.get_all_users_predicted_ratings()
        preds_df = self.prepare_predictions(all_user_predicted_ratings)

        if self.df_posts is not None and self.df_ratings is not None:
            already_rated, predictions = recommend_posts(preds_df, user_id, self.df_posts, self.df_ratings,
                                                         num_of_recommendations)
        else:
            raise ValueError("Dataframe of posts is None. Cannot continue with next operation.")
        print("already_rated.head(num_of_recommendations)")
        print(already_rated.head(num_of_recommendations))
        print("List of predictions based on already rated items:")
        print(predictions.head(num_of_recommendations))
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

    def rmse_all_users(self):
        all_user_predicted_ratings = self.get_all_users_predicted_ratings()
        predictions_df = self.prepare_predictions(all_user_predicted_ratings)
        # Get and sort the user's predictions
        # sorted_users_predictions = pd.DataFrame()
        user_ids = self.get_all_users_ids()
        print(user_ids)
        print("predictions_df")
        print(predictions_df)

        recommender_methods = RecommenderMethods()
        already_rated_by_users = recommender_methods.get_ratings_dataframe()
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

        rmse_metric = mean_squared_error(already_rated_matrix, predictions_matrix, squared=True)

        print("RMSE with 0 on missing values:")
        print(rmse_metric)

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
    # svd_class = SvdFreshApi()
    # print(svd_class.run_svd(431))
    # print(svd.rmse_all_users())
    """
    rated, all = svd.get_average_post_rating()
    print("rated:")
    print(rated)
    print("all:")
    print(all)
    """


# noinspection  PyPep8
if __name__ == "__main__": main()
