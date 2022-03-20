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
        user_preferences_posts = user_based_recommendation.load_recommended_posts_for_user(user_id)
        user_collaboration_posts = svd.run_svd(user_id)

        if len(keyword_list) > 0:
            feature_list.append([tfidf_posts,tfidf_keywords,doc2vec_posts,lda_posts,user_preferences_posts,user_collaboration_posts])
        else:
            feature_list.append([tfidf_posts, doc2vec_posts, lda_posts, user_preferences_posts,
                                 user_collaboration_posts])

        # predictions(tfidf,doc2vec,lda,wor2vec,user_rating,thumbs) = c0 + c1 * tfidf + c2 * doc2vec + c3 * lda + c4 * wor2vec + c5 * user_rating + c6 * thumbs

def main():
    user_id = 431
    post_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    learn_to_rank = LearnToRank()
    learn_to_rank.linear_regression(user_id, post_slug)

if __name__ == "__main__": main()