import datetime
import logging
import os
import traceback

from src.constants.file_paths import get_cached_posts_file_path
from src.data_handling.data_queries import RecommenderMethods
from src.data_handling.dataframe_methods.preprocessing import preprocess_single_post_find_by_slug
from src.methods.content_based.doc2vec import Doc2VecClass
from src.methods.content_based.lda import Lda
from src.methods.content_based.tfidf import TfIdf
from src.methods.content_based.word2vec.word2vec import Word2VecClass
from src.methods.user_based.collaboration_based_recommendation import SvdClass
from src.methods.user_based.user_keywords_recommendation import UserBasedMethods
from src.prefillers.preprocessing.czech_preprocessing import cz_lemma

from flask import Flask, request
from flask_restful import Resource, Api
from flask_wtf.csrf import CSRFProtect

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's why using prints, although not ideal.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging.")


# TODO: Replace prints with debug logging
def check_if_cache_exists_and_fresh():
    if os.path.exists(get_cached_posts_file_path()):
        today = datetime.datetime.today()
        modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(get_cached_posts_file_path()))
        duration = today - modified_date
        # if file older than 1 day
        if duration.total_seconds() / (24 * 60 * 60) > 1:
            return False
        else:
            recommender_methods = RecommenderMethods()
            cached_df = recommender_methods.get_posts_dataframe(force_update=False, from_cache=True)
            sql_columns = recommender_methods.get_sql_columns().tolist()  # tolist() for converting Pandas index to list
            print("sql_columns:")
            print(sql_columns)
            sql_columns.remove('bert_vector_representation')
            if len(cached_df.columns) == len(sql_columns):
                # -1 because bert_vector_representation needs to be excluded from cache
                if set(cached_df.columns) == set(sql_columns):
                    return True
            else:
                return False
    else:
        return False


csrf = CSRFProtect()


def create_app():
    # initializing files needed for the start of application
    # checking needed parts...

    if not check_if_cache_exists_and_fresh():
        # NOTICE: Logging not working here, using prints
        print("Posts cache file does not exists, older than 1 day or columns do not match PostgreSQL columns.")
        print("Creating posts cache file...")
        recommender_methods = RecommenderMethods()
        recommender_methods.database.insert_posts_dataframe_to_cache(recommender_methods.cached_file_path)
    print("Crating flask app...")
    flask_app = Flask(__name__)
    print("FLASK APP READY TO START!")
    csrf.init_app(flask_app)
    return flask_app


app = create_app()
api = Api(app)


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Moje články</h1><p>API pro doporučovací algoritmy.</p>'''


# Here was GetPostByLearnToRiank using XGBoost

class GetPostsByOtherPostTfIdf(Resource):

    def get(self, param):
        tfidf = TfIdf()
        return tfidf.recommend_posts_by_all_features_preprocessed(param)

    def post(self):
        return {"data": "Posted"}


class GetPostsByOtherPostWord2Vec(Resource):

    def get(self, param):
        word2vec_class = Word2VecClass()
        return word2vec_class.get_similar_word2vec(param, 'idnes_3')

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
        user_based_recommendation = UserBasedMethods()
        return user_based_recommendation.load_best_rated_by_others_in_user_categories(param1, param2)

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


api.add_resource(GetPostsByOtherUsers, "/api/evalutation/<int:param1>/<int:param2>")
api.add_resource(GetPostsByUserPreferences, "/api/evalutation-preferences/<int:param1>/<int:param2>")
api.add_resource(GetPostsByKeywords, "/api/evalutation-keywords")

api.add_resource(GetWordLemma, "/api/lemma/<string:word>")
api.add_resource(Preprocess, "/api/preprocess/<string:slug>")

api.add_resource(GetPostsByOtherPostTfIdf, "/api/post-terms_frequencies/<string:param>")
api.add_resource(GetPostsByOtherPostWord2Vec, "/api/post-word2vec/<string:param>")
api.add_resource(GetPostsByOtherPostDoc2Vec, "/api/post-doc2vec/<string:param>")
api.add_resource(GetPostsByOtherPostLda, "/api/post-topics/<string:param>")

api.add_resource(GetPostsByOtherPostTfIdfFullText, "/api/post-terms_frequencies-full-text/<string:param>")
api.add_resource(GetPostsByOtherPostWord2VecFullText, "/api/post-word2vec-full-text/<string:param>")
api.add_resource(GetPostsByOtherPostDoc2VecFullText, "/api/post-doc2vec-full-text/<string:param>")
api.add_resource(GetPostsByOtherPostLdaFullText, "/api/post-topics-full-text/<string:param>")

if __name__ == "__main__":
    app.run(debug=True)
