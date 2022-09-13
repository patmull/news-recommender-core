import json
import random
import time as t
import psycopg2
from gensim.models import KeyedVectors

from src.recommender_core.recommender_algorithms.content_based_algorithms.doc2vec import Doc2VecClass
from src.recommender_core.recommender_algorithms.content_based_algorithms.doc_sim import DocSim
from src.recommender_core.recommender_algorithms.content_based_algorithms.lda import Lda
from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods

val_error_msg_db = "Not allowed DB model_variant was passed for prefilling. Choose 'pgsql' or 'redis'."
val_error_msg_algorithm = "Selected model_variant does not correspondent with any implemented model_variant."


def fill_recommended(method, skip_already_filled, full_text=True, random_order=False, reversed_order=False):

    source = w2v_model = selected_model_name = None
    docsim_index, dictionary = None, None
    database = DatabaseMethods()
    if skip_already_filled is False:
        posts = database.get_all_posts()
    else:
        database.connect()
        posts = database.get_not_prefilled_posts(full_text, method=method)
        database.disconnect()

    number_of_inserted_rows = 0

    if reversed_order is True:
        print("Reversing list of posts...")
        posts.reverse()

    if random_order is True:
        print("Starting random_order iteration...")
        t.sleep(5)
        random.shuffle(posts)

    if method.startswith("word2vec"):
        if method == "word2vec_eval_idnes_1":
            selected_model_name = "idnes_1"
            path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_1/"
        elif method == "word2vec_eval_idnes_2":
            selected_model_name = "idnes_2"
            path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_2_default_parameters/"
        elif method == "word2vec_eval_idnes_3":
            selected_model_name = "idnes_3"
            path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_3/"
        elif method == "word2vec_eval_idnes_4":
            selected_model_name = "idnes_4"
            path_to_folder = "full_models/idnes/evaluated_models/word2vec_model_4/"
        elif method == "word2vec_limited_fasttext":
            selected_model_name = "fasttext_limited"
            path_to_folder = "full_models/cswiki/word2vec_fassttext_model/"
        elif method == "word2vec_limited_fasttext_full_text":
            selected_model_name = "fasttext_full_text"
            path_to_folder = "full_models/cswiki/word2vec_fassttext_full_text_model/"
        elif method == "word2vec_eval_cswiki_1":
            selected_model_name = "cswiki"
            path_to_folder = "full_models/cswiki/evaluated_models/word2vec_model_cswiki_1/"
        else:
            raise ValueError("Wrong doc2vec_model name chosen.")

        if method.startswith("word2vec_eval_idnes_"):
            file_name = "w2v_idnes.model"
            path_to_model = path_to_folder + file_name
            w2v_model = KeyedVectors.load(path_to_model)
            source = "idnes"
        elif method.startswith("word2vec_eval_cswiki_"):
            file_name = "w2v_cswiki.model"
            path_to_model = path_to_folder + file_name
            w2v_model = KeyedVectors.load(path_to_model)
            source = "cswiki"
        else:
            ValueError("Wrong doc2vec_model name chosen.")

        ds = DocSim(w2v_model)
        docsim_index = ds.load_docsim_index(source=source, model_name=selected_model_name)
    elif method.startswith("doc2vec"):
        if method == "doc2vec_eval_cswiki_1":
            print("Similarities on FastText doc2vec_model.")
            print("Loading Dov2Vec cs.Wikipedia.org doc2vec_model...")

    for post in posts:
        if len(posts) < 1:
            break

        post_id = post[0]
        slug = post[3]

        if full_text is False:
            if method == "tfidf":
                current_recommended = post[24]
            elif method == "word2vec":
                current_recommended = post[22]
            elif method == "doc2vec":
                current_recommended = post[26]
            elif method == "lda":
                current_recommended = post[28]
            else:
                current_recommended = None
        else:
            if method == "tfidf":
                current_recommended = post[25]
            elif method == "word2vec":
                current_recommended = post[23]
            elif method == "doc2vec":
                current_recommended = post[27]
            elif method == "lda":
                current_recommended = post[29]
            elif method == "word2vec_eval_idnes_1":
                current_recommended = post[33]
            elif method == "word2vec_eval_idnes_2":
                current_recommended = post[34]
            elif method == "word2vec_eval_idnes_3":
                current_recommended = post[35]
            elif method == "word2vec_eval_idnes_4":
                current_recommended = post[36]
            elif method == "word2vec_limited_fasttext":
                current_recommended = post[37]
            elif method == "word2vec_limited_fasttext_full_text":
                current_recommended = post[38]
            elif method == "word2vec_eval_cswiki_1":
                current_recommended = post[39]
            elif method == "doc2vec_eval_cswiki_1":
                current_recommended = post[40]
            else:
                current_recommended = None

        print("Searching similar articles for article: ")
        print(slug)

        if skip_already_filled is True:
            if current_recommended is None:
                print("Post:")
                print(slug)
                print("Has currently no recommended posts.")
                print("Trying to find recommended...")
                if full_text is False:
                    if method == "tfidf":
                        tfidf = TfIdf()
                        actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed(slug)
                    elif method == "word2vec":
                        if docsim_index is None:
                            raise ValueError("DocSim index is not set.")
                        if dictionary is None:
                            raise ValueError("Dictionary is not set")
                        word2vec = Word2VecClass()
                        path_to_model = "models/w2v_model_limited"
                        w2v_model = KeyedVectors.load(path_to_model)
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model_name=w2v_model,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "doc2vec":
                        doc2vec = Doc2VecClass()
                        actual_recommended_json = doc2vec.get_similar_doc2vec(searched_slug=slug)
                    elif method == "lda":
                        lda = Lda()
                        actual_recommended_json = lda.get_similar_lda(slug)
                    else:
                        actual_recommended_json = None
                else:
                    if method == "tfidf":
                        tfidf = TfIdf()
                        actual_recommended_json = tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(
                            slug)
                    elif method == "word2vec":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec_full_text(searched_slug=slug)
                    elif method == "doc2vec":
                        doc2vec = Doc2VecClass()
                        actual_recommended_json = doc2vec.get_similar_doc2vec_with_full_text(slug)
                    elif method == "lda":
                        lda = Lda()
                        actual_recommended_json = lda.get_similar_lda_full_text(slug)
                    elif method == "word2vec_eval_idnes_1":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model_name=w2v_model,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_eval_idnes_2":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model_name=w2v_model,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_eval_idnes_3":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model_name=w2v_model,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_eval_idnes_4":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model_name=w2v_model,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_fasttext":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model_name=w2v_model,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_fasttext_full_text":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model_name=w2v_model,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "word2vec_eval_cswiki_1":
                        word2vec = Word2VecClass()
                        actual_recommended_json = word2vec.get_similar_word2vec(searched_slug=slug,
                                                                                model_name=w2v_model,
                                                                                docsim_index=docsim_index,
                                                                                dictionary=dictionary)
                    elif method == "doc2vec_eval_cswiki_1":
                        doc2vec = Doc2VecClass()
                        actual_recommended_json = doc2vec.get_similar_doc2vec(searched_slug=slug)
                    else:
                        actual_recommended_json = None
                if len(actual_recommended_json) == 0:
                    print("No recommended post found. Skipping.")
                    continue
                else:
                    actual_recommended_json = json.dumps(actual_recommended_json)

                if full_text is False:
                    try:
                        database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                         article_id=post_id, full_text=False, db="pgsql",
                                                         method=method)
                    except Exception as e:
                        print("Error in DB insert. Skipping.")
                        print(e)
                        pass
                else:
                    try:
                        database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                         article_id=post_id, full_text=True, db="pgsql",
                                                         method=method)
                        number_of_inserted_rows += 1
                        print("Inserted rows in current prefilling round: " + str(number_of_inserted_rows))
                    except Exception as e:
                        print("Error in DB insert. Skipping.")
                        print(e)
                        pass
            else:
                print("Skipping.")


def prefilling_job(method, full_text, random_order=False, reversed_order=True):
    while True:
        try:
            fill_recommended(method=method, full_text=full_text, skip_already_filled=True,
                             random_order=random_order, reversed_order=reversed_order)
        except psycopg2.OperationalError:
            print("DB operational error. Waiting few seconds before trying again...")
            t.sleep(30)  # wait 30 seconds then try again
            continue
        break
