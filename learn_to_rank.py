import itertools

import pandas as pd

from collaboration_based_recommendation import Svd
from content_based_algorithms.tfidf import TfIdf
from content_based_algorithms.doc2vec import Doc2VecClass
from content_based_algorithms.lda import Lda
from user_based_recommendation import UserBasedRecommendation


class LearnToRank:

    def linear_regression(self, user_id, post_slug):

        tfidf = TfIdf()
        doc2vec = Doc2VecClass()
        lda = Lda()
        user_based_recommendation = UserBasedRecommendation()
        svd = Svd()

        feature_list = []

        tfidf_posts = tfidf.recommend_posts_by_all_features_preprocessed(post_slug)

        user_keywords = user_based_recommendation.get_user_keywords(user_id)
        keyword_list = user_keywords['keyword_name'].tolist()
        tfidf_keywords = ''
        if len(keyword_list) > 0:
            keywords = ' '.join(keyword_list)
            print(keywords)
            tfidf_keywords = tfidf.keyword_based_comparison(keywords)

        doc2vec_posts = doc2vec.get_similar_doc2vec(post_slug)
        lda_posts = lda.get_similar_lda(post_slug)

        user_preferences_posts = user_based_recommendation.load_recommended_posts_for_user(user_id,
                                                                                           num_of_recommendations=20)
        print("user_preferences_posts")
        print(user_preferences_posts)
        df = pd.DataFrame.from_dict(user_preferences_posts, orient='index').transpose()
        user_preferences_posts_df = pd.DataFrame(df['data'].tolist(),
                                                 columns=['post_id', 'slug', 'rating_value', 'post_created_at'])
        user_preferences_posts_df = user_preferences_posts_df[['slug', 'rating_value']]
        user_preferences_posts_dict = user_preferences_posts_df.to_dict('records')
        print("user_preferences_posts_dict")
        print(user_preferences_posts_dict)

        user_collaboration_posts = svd.run_svd(user_id, num_of_recommendations=20)
        df = pd.DataFrame.from_dict(user_collaboration_posts, orient='index').transpose()
        user_collaboration_posts_df = pd.DataFrame(df['data'].tolist(),
                                                   columns=['post_id', 'slug', 'rating_value'])
        user_collaboration_posts_df = user_collaboration_posts_df[['slug', 'rating_value']]
        user_collaboration_posts_dict = user_collaboration_posts_df.to_dict('records')

        if len(keyword_list) > 0:
            feature_list.append([tfidf_posts, tfidf_keywords, doc2vec_posts, lda_posts, user_preferences_posts,
                                 user_collaboration_posts])
        else:
            feature_list.append([tfidf_posts, doc2vec_posts, lda_posts, user_preferences_posts,
                                 user_collaboration_posts])
        print("tfidf_posts")
        print(tfidf_posts)
        print("tfidf_keywords")
        print(tfidf_keywords)
        print("doc2vec_posts")
        print(doc2vec_posts)
        print("lda_posts")
        print(lda_posts)
        print("user_preferences_posts_dict")
        print(user_preferences_posts_dict)
        print("user_collaboration_posts")
        print(user_collaboration_posts_dict)

        print("feature_list")
        print(feature_list)

        tfidf_posts_df = pd.DataFrame(tfidf_posts)
        tfidf_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'coefficients_tfidf_posts'}, inplace=True)
        print("tfidf_posts_df:")
        print(tfidf_posts_df)

        tfidf_keywords_df = pd.DataFrame(tfidf_keywords)
        tfidf_keywords_df.rename(columns={'slug': 'slug', 'coefficient': 'coefficients_tfidf_keywords'}, inplace=True)
        print("tfidf_keywords_df:")
        print(tfidf_keywords_df)

        doc2vec_posts_df = pd.DataFrame(doc2vec_posts)
        doc2vec_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'coefficients_doc2vec_posts'}, inplace=True)
        print("tfidf_doc2vec_posts:")
        print(doc2vec_posts_df)

        lda_posts_df = pd.DataFrame(lda_posts)
        lda_posts_df.rename(columns={'slug': 'slug', 'coefficient': 'coefficients_lda_posts'}, inplace=True)
        print("tfidf_posts_df:")
        print(tfidf_posts_df)

        user_preferences_posts_dict_df = pd.DataFrame(user_preferences_posts_dict)
        user_preferences_posts_dict_df.rename(columns={'slug': 'slug', 'coefficient': 'coefficients_tf_idf_posts'},
                                              inplace=True)
        print("user_preferences_posts_dict_df:")
        print(user_preferences_posts_dict_df)

        user_collaboration_posts_df = pd.DataFrame(user_collaboration_posts_dict)
        user_collaboration_posts_df.rename(
            columns={'slug': 'slug', 'coefficient': 'coefficients_user_collaboration_posts_dict'}, inplace=True)
        print("user_collaboration_posts_dict_df:")
        print(user_collaboration_posts_df)

        # Convert to Dictionary to show also names of the dataframes?
        recommender_dataframes = [tfidf_posts_df, tfidf_keywords_df, doc2vec_posts_df, lda_posts_df,
                                  user_preferences_posts_dict_df, user_collaboration_posts_df]

        i, j = 0, 0
        for dataframe_i, dataframe_j in itertools.combinations(recommender_dataframes, 2):
                    dictionary_intersection = self.intersect(dataframe_i, dataframe_j)
                    print("dictionary_intersection")
                    print(dictionary_intersection)

        # predictions(tfidf,doc2vec,lda,wor2vec,user_r
        # ating,thumbs) = c0 + c1 * tfidf + c2 * doc2vec + c3 * lda + c4 * wor2vec + c5 * user_rating + c6 * thumbs

    def intersect(self, a, b):
        return pd.merge(a, b, how='inner', on=['slug'])


def main():
    user_id = 431
    post_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    learn_to_rank = LearnToRank()
    learn_to_rank.linear_regression(user_id, post_slug)


if __name__ == "__main__": main()
