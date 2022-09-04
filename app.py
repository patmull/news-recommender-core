import traceback

from threading import Thread

from core.data_handling import RecommenderMethods
from core.recommender_algorithms.learn_to_rank.learn_to_rank import LightGBM, LearnToRank
from core.recommender_algorithms.user_based_algorithms.user_based_recommendation import UserBasedRecommendation
from flask import Flask, request
from flask_restful import Resource, Api
from core.recommender_algorithms.content_based_algorithms.lda import Lda
from core.recommender_algorithms.content_based_algorithms import Doc2VecClass
from core.recommender_algorithms.content_based_algorithms import TfIdf
from core.recommender_algorithms.content_based_algorithms import Word2VecClass
from core.recommender_algorithms.user_based_algorithms import SvdClass


def create_app():
    # initializing files needed for the start of application
    # checking needed parts...
    print("Creating posts cache file...")
    recommender_methods = RecommenderMethods()
    recommender_methods.database.connect()
    recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)
    recommender_methods.database.disconnect()
    print("Crating flask app...")
    app = Flask(__name__)
    print("FLASK APP READY TO START!")
    return app


app = create_app()
api = Api(app)

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Moje články</h1><p>API pro doporučovací algoritmy.</p>'''


class GetPostsLearnToRank(Resource):

    def get(self, param):
        lightgbm = LightGBM()
        return lightgbm.train_lightgbm_user_based(param)

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
        word2vecClass = Word2VecClass()
        return word2vecClass.get_similar_word2vec(param)

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
        word2vecClass = Word2VecClass()
        return word2vecClass.get_similar_word2vec_full_text(param)

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
        learn_to_rank = LearnToRank()
        return learn_to_rank.linear_regression(param1, param2)

    def post(self):
        return {"data": "Posted"}


class GetWordLemma(Resource):

    def get(self, word):
        tfIdf = TfIdf()
        return tfIdf.cz_lemma(word, json=True)

    def post(self):
        return {"data": "Posted"}


class GetWordStem(Resource):

    def get(self, word, aggressive):
        tfIdf = TfIdf()
        return tfIdf.cz_stem(word, aggressive, json=True)

    def post(self):
        return {"data": "Posted"}


class Preprocess(Resource):

    def get(self, slug):
        tfIdf = TfIdf()
        return tfIdf.preprocess_single_post(slug, json=True)

    def post(self):
        return {"data": "Posted"}


class PreprocessStemming(Resource):

    def get(self, slug):
        tfIdf = TfIdf()
        return tfIdf.preprocess_single_post(slug, json=True, stemming=True)

    def post(self):
        return {"data": "Posted"}


def set_global_exception_handler(app):
    @app.errorhandler(Exception)
    def unhandled_exception(e):
        response = dict()
        error_message = traceback.format_exc()
        app.logger.error("Caught Exception: {}".format(error_message)) # or whatever logger you use
        response["errorMessage"] = error_message
        return response, 500


@app.route("/api/refresh-cache")
def refresh_cache():
    def do_update_cache_work():
        # do something that takes a long time
        print("Calling cache refreshing method. This will take some time...")
        recommenderMethods = RecommenderMethods()
        recommenderMethods.refresh_cached_db_file()

    # thread = Thread(target=do_work, kwargs={'value': request.args.get('value', 20)})
    thread = Thread(target=do_update_cache_work)
    thread.start()
    return 'Request for cache refresh recieved. Started refreshing', 200


api.add_resource(GetPostsByOtherUsers, "/api/user/<int:param1>/<int:param2>")
api.add_resource(GetPostsByUserPreferences, "/api/user-preferences/<int:param1>/<int:param2>")
api.add_resource(GetPostsByKeywords, "/api/user-keywords")

api.add_resource(GetPostsByLearnToRank, "/api/learn-to-rank/<int:param1>/<string:param2>")

api.add_resource(GetWordLemma, "/api/lemma/<string:word>")
api.add_resource(GetWordStem, "/api/stem/<string:word>/<string:aggressive>")
api.add_resource(Preprocess, "/api/preprocess/<string:slug>")
api.add_resource(PreprocessStemming, "/api/preprocess-stemming/<string:slug>")

api.add_resource(GetPostsByOtherPostTfIdf, "/api/post-tfidf/<string:param>")
api.add_resource(GetPostsByOtherPostWord2Vec, "/api/post-word2vec/<string:param>")
api.add_resource(GetPostsByOtherPostDoc2Vec, "/api/post-doc2vec/<string:param>")
api.add_resource(GetPostsByOtherPostLda, "/api/post-lda/<string:param>")

api.add_resource(GetPostsByOtherPostTfIdfFullText, "/api/post-tfidf-full-text/<string:param>")
api.add_resource(GetPostsByOtherPostWord2VecFullText, "/api/post-word2vec-full-text/<string:param>")
api.add_resource(GetPostsByOtherPostDoc2VecFullText, "/api/post-doc2vec-full-text/<string:param>")
api.add_resource(GetPostsByOtherPostLdaFullText, "/api/post-lda-full-text/<string:param>")

if __name__ == "__main__":
    app.run(debug=True)
