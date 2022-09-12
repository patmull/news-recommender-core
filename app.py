import datetime
import os
import traceback

from src.constants.file_paths import get_cached_posts_file_path
from src.prefillers.preprocessing.cz_preprocessing import cz_lemma
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.lda import Lda
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.recommender_algorithms.learn_to_rank.learn_to_rank_methods import train_lightgbm_user_based, \
    linear_regression
from src.recommender_core.recommender_algorithms.user_based_algorithms\
    .collaboration_based_recommendation import SvdClass
from src.recommender_core.data_handling.data_queries import RecommenderMethods, preprocess_single_post_find_by_slug
from src.recommender_core.recommender_algorithms\
    .user_based_algorithms.user_based_recommendation import UserBasedRecommendation
from flask import Flask, request
from flask_restful import Resource, Api


def check_if_cache_exists_and_fresh():
    if os.path.exists(get_cached_posts_file_path()):
        today = datetime.datetime.today()
        modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(get_cached_posts_file_path()))
        duration = today - modified_date
        # if file older than 1 day
        if duration.total_seconds()/(24*60*60) > 1:
            return False
        else:
            recommender_methods = RecommenderMethods()
            cached_df = recommender_methods.get_posts_dataframe(force_update=False, from_cache=True)
            sql_columns = recommender_methods.get_sql_columns()
            if len(cached_df.columns) == len(sql_columns):
                if cached_df.columns == sql_columns:
                    return True
            else:
                return False
    else:
        return False


def create_app():
    # initializing files needed for the start of application
    # checking needed parts...

    if not check_if_cache_exists_and_fresh():
        print("Posts cache file does not exists, older than 1 day or columns do not match PostgreSQL columns.")
        print("Creating posts cache file...")
        recommender_methods = RecommenderMethods()
        recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)
    print("Crating flask app...")
    flask_app = Flask(__name__)
    print("FLASK APP READY TO START!")
    return flask_app


app = create_app()
api = Api(app)


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Moje články</h1><p>API pro doporučovací algoritmy.</p>'''


class GetPostsLearnToRank(Resource):

    def get(self):
        return train_lightgbm_user_based()

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostTfIdf(Resource):

    def get(self, param):
        tfidf = TfIdf()
        return tfidf.recommend_posts_by_all_features_preprocessed(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostWord2Vec(Resource):

    def get(self, param):
        word2vec_class = Word2VecClass()
        return word2vec_class.get_similar_word2vec(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostDoc2Vec(Resource):

    def get(self, param):
        doc2vec = Doc2VecClass()
        return doc2vec.get_similar_doc2vec(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostLda(Resource):

    def get(self, param):
        lda = Lda()
        return lda.get_similar_lda(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostTfIdfFullText(Resource):

    def get(self, param):
        tfidf = TfIdf()
        return tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostWord2VecFullText(Resource):

    def get(self, param):
        word2vec_class = Word2VecClass()
        return word2vec_class.get_similar_word2vec_full_text(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostDoc2VecFullText(Resource):

    def get(self, param):
        doc2vec = Doc2VecClass()
        return doc2vec.get_similar_doc2vec_with_full_text(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostLdaFullText(Resource):

    def get(self, param):
        lda = Lda()
        return lda.get_similar_lda_full_text(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByKeywords(Resource):

    def get(self):
        return {"data": "Posted"}

    def post(self):
        input_json_keywords = request.get_json(force=True)
        tfidf = TfIdf()
        return tfidf.keyword_based_comparison(input_json_keywords["keywords"])


class GetPostsByOtherUsers(Resource):

    def get(self, param1, param2):
        svd = SvdClass()
        return svd.run_svd(param1, param2)

    def post(self):
        return {"data": "Posted"}


class GetPostsByUserPreferences(Resource):

    def get(self, param1, param2):
        user_based_recommendation = UserBasedRecommendation()
        return user_based_recommendation.load_recommended_posts_for_user(param1, param2)

    def post(self):
        return {"data": "Posted"}


class GetPostsByLearnToRank(Resource):

    def get(self, param1, param2):
        return linear_regression(param1, param2)

    def post(self):
        return {"data": "Posted"}


class GetWordLemma(Resource):

    def get(self, word):
        return cz_lemma(word, json=True)

    def post(self):
        return {"data": "Posted"}


class Preprocess(Resource):

    def get(self, slug):
        return preprocess_single_post_find_by_slug(slug, supplied_json=True)

    def post(self):
        return {"data": "Posted"}


def set_global_exception_handler(flask_app):
    @app.errorhandler(Exception)
    def unhandled_exception():
        response = dict()
        error_message = traceback.format_exc()
        flask_app.logger.error("Caught Exception: {}".format(error_message))  # or whatever logger you use
        response["errorMessage"] = error_message
        return response, 500


api.add_resource(GetPostsByOtherUsers, "/api/user/<int:param1>/<int:param2>")
api.add_resource(GetPostsByUserPreferences, "/api/user-preferences/<int:param1>/<int:param2>")
api.add_resource(GetPostsByKeywords, "/api/user-keywords")

api.add_resource(GetPostsByLearnToRank, "/api/learn-to-rank/<int:param1>/<input_string:param2>")

api.add_resource(GetWordLemma, "/api/lemma/<input_string:word>")
api.add_resource(Preprocess, "/api/preprocess/<input_string:slug>")

api.add_resource(GetPostsByOtherPostTfIdf, "/api/post-tfidf/<input_string:param>")
api.add_resource(GetPostsByOtherPostWord2Vec, "/api/post-word2vec/<input_string:param>")
api.add_resource(GetPostsByOtherPostDoc2Vec, "/api/post-doc2vec/<input_string:param>")
api.add_resource(GetPostsByOtherPostLda, "/api/post-lda/<input_string:param>")

api.add_resource(GetPostsByOtherPostTfIdfFullText, "/api/post-tfidf-full-text/<input_string:param>")
api.add_resource(GetPostsByOtherPostWord2VecFullText, "/api/post-word2vec-full-text/<input_string:param>")
api.add_resource(GetPostsByOtherPostDoc2VecFullText, "/api/post-doc2vec-full-text/<input_string:param>")
api.add_resource(GetPostsByOtherPostLdaFullText, "/api/post-lda-full-text/<input_string:param>")

if __name__ == "__main__":
    app.run(debug=True)
