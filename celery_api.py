from flask import Flask, request

from collaboration_based_recommendation import Svd
from flask_celery_config import make_celery
from flask_restful import Api, Resource
from content_based_old import LdaOld, TfIdfOld, Doc2VecOld, Word2VecOld
from content_based_recommendation import Lda, TfIdf, Doc2VecClass, Word2VecClass
from urllib.parse import urlparse
import json
import redis
import os

from user_based_recommendation import UserBasedRecommendation

REDIS_URL = os.environ.get('REDISCLOUD_URL')
RABBIT_URL = 'pyamqp://guest@localhost//'
redis_url = urlparse(REDIS_URL)
r = redis.Redis(host=redis_url.hostname, port=redis_url.port, username=redis_url.username, password=redis_url.password)


def create_app():
    application = app = Flask(__name__)
    app.config['CELERY_BROKER_URL'] = REDIS_URL
    app.config['CELERY_BACKEND'] = REDIS_URL

    """
    from celery.celery_api import bp
    app.register_blueprint(bp)
    """
    return app


app = create_app()
celery = make_celery(app)
# celery.conf.update(BROKER_URL=REDIS_URL,Cdef post_lda(slug_param):ELERY_RESULT_BACKEND=REDIS_URL)
api = Api(app)


# redis_url = urlparse(REDIS_URL)
# r = redis.Redis(host=redis_url.hostname, port=redis_url.port, username=redis_url.username, password=redis_url.password)


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

class GetPostsByOtherPostDoc2VecOld(Resource):

    def get(self, param):
        doc2vec = Doc2VecOld()
        return doc2vec.get_similar_doc2vec(param)

    def post(self):
        return {"data": "Posted"}

class GetPostsByOtherPostWord2VecOld(Resource):

    def get(self, param):
        word2vecClass = Word2VecOld()
        return word2vecClass.get_similar_word2vec(param)

    def post(self):
        return {"data": "Posted"}

"""
class GetPostsByOtherPostLdaOld(Resource):

    def get(self, param):
        lda = LdaOld()
        return lda.get_similar_lda(param)

    def post(self):
        return {"data": "Posted"}
"""

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

@app.route('/api/post-lda-old/<param>')
def process_post_lda(param):
    # post_lda.delay(param)
    post_lda.delay(param)
    return {"waiting_for": "celery"}


@celery.task(name="celery_api.post_lda")
def post_lda(slug_param):
    print("Processing LDA posts")
    # lda_old = LdaOld()
    lda_old = LdaOld()
    json_result = json.dumps(lda_old.get_similar_lda(slug_param))
    r.set(slug_param, json_result)
    print(r.get(slug_param))  # just for check


api.add_resource(GetPostsByOtherPostTfIdf, "/api/post-tfidf/<string:param>")
api.add_resource(GetPostsByOtherPostWord2Vec, "/api/post-word2vec/<string:param>")
api.add_resource(GetPostsByOtherPostDoc2Vec, "/api/post-doc2vec/<string:param>")
api.add_resource(GetPostsByOtherPostLda, "/api/post-lda/<string:param>")

api.add_resource(GetPostsByOtherUsers, "/api/user/<int:param1>/<int:param2aram>")
api.add_resource(GetPostsByUserPreferences, "/api/user-preferences/<int:param1>/<int:param2>")
api.add_resource(GetPostsByKeywords, "/api/user-keywords")
api.add_resource(GetWordLemma, "/api/lemma/<string:word>")
api.add_resource(GetWordStem, "/api/stem/<string:word>/<string:aggressive>")
api.add_resource(Preprocess, "/api/preprocess/<string:slug>")
api.add_resource(PreprocessStemming, "/api/preprocess-stemming/<string:slug>")

api.add_resource(GetPostsByOtherPostTfIdfOld, "/api/post-tfidf-old/<string:param>")
api.add_resource(GetPostsByOtherPostWord2VecOld, "/api/post-word2vec-old/<string:param>")
api.add_resource(GetPostsByOtherPostDoc2VecOld, "/api/post-doc2vec-old/<string:param>")
# api.add_resource(GetPostsByOtherPostLdaOld, "/api/post-lda-old/<string:param>")


if __name__ == "__main__":
    # print("Loading Gensim models...")
    # load_models()
    # print("Gensim model loaded.")
    # print("Loading stopwords")
    # load_stopwords()
    # print("Stopwords loaded")
    app.run(debug=True)
