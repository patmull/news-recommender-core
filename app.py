import traceback

from user_based_recommendation import UserBasedRecommendation
from flask import Flask, request
from flask_restful import Api, Resource
from content_based_recommendation import TfIdf, Doc2VecClass, Word2VecClass, Lda, word2vec_embedding, load_models, load_stopwords
from collaboration_based_recommendation import Svd
from content_based_old import TfIdfOld, Doc2VecOld, Word2VecOld, LdaOld

REDIS_URL = 'redis://default:RuTYjOqZZofckHAPTNqUMlg8XjT1nQdZ@redis-18878.c59.eu-west-1-2.ec2.cloud.redislabs.com:18878'

def create_app():
    app = Flask(__name__)
    #app.config['CELERY_BROKER_URL'] = REDIS_URL
    #app.config['CELERY_BACKEND'] = REDIS_URL

    return app

app = create_app()
# celery = make_celery(app)
api = Api(app)

"""
redis_url = urlparse(REDIS_URL)
r = redis.Redis(host=redis_url.hostname, port=redis_url.port, username=redis_url.username, password=redis_url.password)
"""

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Moje články</h1><p>API pro doporučovací algoritmy.</p>'''

class GetPostsByOtherPostTfIdf(Resource):

    def get(self, param):
        tfidf = TfIdf()
        return tfidf.recommend_posts_by_all_features_preprocessed(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostDoc2Vec(Resource):

    def get(self, param):
        doc2vec = Doc2VecClass()
        return doc2vec.get_similar_doc2vec(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostWord2Vec(Resource):

    def get(self, param):
        word2vecClass = Word2VecClass()
        return word2vecClass.get_similar_word2vec(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostLda(Resource):

    def get(self, param):
        lda = Lda()
        return lda.get_similar_lda(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostTfIdfOld(Resource):

    def get(self, param):
        tfidf = TfIdfOld()
        return tfidf.recommend_posts_by_all_features_preprocessed(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostTfIdfFullText(Resource):

    def get(self, param):
        tfidf = TfIdf()
        return tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostDoc2VecOld(Resource):

    def get(self, param):
        doc2vec = Doc2VecOld()
        return doc2vec.get_similar_doc2vec(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostDoc2VecFullText(Resource):

    def get(self, param):
        doc2vec = Doc2VecClass()
        return doc2vec.get_similar_doc2vec_with_full_text(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostWord2VecOld(Resource):

    def get(self, param):
        word2vecClass = Word2VecOld()
        return word2vecClass.get_similar_word2vec(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostWord2VecFullText(Resource):

    def get(self, param):
        word2vecClass = Word2VecClass()
        return word2vecClass.get_similar_word2vec_full_text(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostLdaOld(Resource):

    def get(self, param):
        lda = LdaOld()
        return lda.get_similar_lda(param)

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
        svd = Svd()
        return svd.run_svd(param1, param2)

    def post(self):
        return {"data": "Posted"}


class GetPostsByUserPreferences(Resource):

    def get(self, param1, param2):
        user_based_recommendation = UserBasedRecommendation()
        return user_based_recommendation.load_recommended_posts_for_user(param1, param2)

    def post(self):
        return {"data": "Posted"}


class GetWordLemma(Resource):

    def get(self, word):
        tfIdf = TfIdf()
        return tfIdf.cz_lemma(word,json=True)

    def post(self):
        return {"data": "Posted"}


class GetWordStem(Resource):

    def get(self, word, aggressive):
        tfIdf = TfIdf()
        return tfIdf.cz_stem(word, aggressive,json=True)

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
        app.logger.error("Caught Exception: {}".format(error_message)) #or whatever logger you use
        response["errorMessage"] = error_message
        return response, 500


api.add_resource(GetPostsByOtherPostTfIdf, "/api/post-tfidf/<string:param>")
api.add_resource(GetPostsByOtherPostWord2Vec, "/api/post-word2vec/<string:param>")
api.add_resource(GetPostsByOtherPostDoc2Vec, "/api/post-doc2vec/<string:param>")
api.add_resource(GetPostsByOtherPostLda, "/api/post-lda/<string:param>")

api.add_resource(GetPostsByOtherUsers, "/api/user/<int:param1>/<int:param2>")
api.add_resource(GetPostsByUserPreferences, "/api/user-preferences/<int:param1>/<int:param2>")
api.add_resource(GetPostsByKeywords, "/api/user-keywords")
api.add_resource(GetWordLemma, "/api/lemma/<string:word>")
api.add_resource(GetWordStem, "/api/stem/<string:word>/<string:aggressive>")
api.add_resource(Preprocess, "/api/preprocess/<string:slug>")
api.add_resource(PreprocessStemming, "/api/preprocess-stemming/<string:slug>")

api.add_resource(GetPostsByOtherPostTfIdfOld, "/api/post-tfidf-old/<string:param>")
api.add_resource(GetPostsByOtherPostWord2VecOld, "/api/post-word2vec-old/<string:param>")
api.add_resource(GetPostsByOtherPostDoc2VecOld, "/api/post-doc2vec-old/<string:param>")
api.add_resource(GetPostsByOtherPostLdaOld, "/api/post-lda-old/<string:param>")

api.add_resource(GetPostsByOtherPostTfIdfFullText, "/api/post-tfidf-full-text/<string:param>")
api.add_resource(GetPostsByOtherPostWord2VecFullText, "/api/post-word2vec-full-text/<string:param>")
api.add_resource(GetPostsByOtherPostDoc2VecFullText, "/api/post-doc2vec-full-text/<string:param>")
api.add_resource(GetPostsByOtherPostLdaFullText, "/api/post-lda-full-text/<string:param>")

if __name__ == "__main__":
    # print("Loading Gensim models...")
    # load_models()
    # print("Gensim model loaded.")
    # print("Loading stopwords")
    # load_stopwords()
    # print("Stopwords loaded")
    app.run(debug=True)
